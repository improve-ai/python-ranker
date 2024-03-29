from copy import deepcopy
import numpy as np
import json
from ksuid import Ksuid
import orjson
import pandas as pd
from scipy import stats
import string
from tqdm import tqdm
from warnings import warn

from improveai import Scorer, RewardTracker


LETTERS = [letter for letter in string.ascii_letters]
DIGITS = [digit for digit in string.digits]

DECISION_ID_KEY = 'decision_id'
MODEL_KEY = 'model'
REWARD_KEY = 'reward'
REWARDS_KEY = 'rewards'
ITEM_KEY = 'item'
CONTEXT_KEY = 'context'
COUNT_KEY = 'count'
SAMPLE_KEY = 'sample'

NUMERIC_COLUMNS_DTYPE = 'float64'

DF_SCHEMA = {
    DECISION_ID_KEY: 'object',
    ITEM_KEY: 'object',
    CONTEXT_KEY: 'object',
    COUNT_KEY: NUMERIC_COLUMNS_DTYPE,
    SAMPLE_KEY: 'object',
    REWARDS_KEY: 'object',
    REWARD_KEY: NUMERIC_COLUMNS_DTYPE,
}


# DATASET_NAME_KEY = 'dataset_name'
# TIMESPAN_KEY = 'timespan'
# ENCODING_NOISE_KEY = 'encoding_noise'
# MODEL_SEED_KEY = 'model_seed'
#
# VARIANTS_DEFINITION_KEY = 'variants_definition'
#
# GIVENS_DEFINITION_KEY = 'givens_definition'
# VARIANTS_TO_GIVENS_MAPPING_KEY = 'variants_to_givens_mapping'
# GIVENS_FRACTION_KEY = 'givens_fraction'
#
# REWARD_MAPPING_KEY = 'reward_mapping'
# RECORDS_PER_EPOCH_KEY = 'records_per_epoch'
# EPOCHS_KEY = 'epochs'
# DATA_SEED_KEY = 'data_seed'


class BasicSemiRandomDataGenerator:

    @property
    def data_definition_json_path(self):
        return self._data_definition_json_path

    @data_definition_json_path.setter
    def data_definition_json_path(self, value):
        self._data_definition_json_path = value

    @property
    def data_definition(self):
        return self._data_definition

    @data_definition.setter
    def data_definition(self, value):
        self._data_definition = value

    @property
    def reward_cache(self):
        return self._reward_cache

    @reward_cache.setter
    def reward_cache(self, value):
        self._reward_cache = value

    @property
    def max_allowed_regret_ratio(self):
        return self._max_allowed_regret_ratio

    @max_allowed_regret_ratio.setter
    def max_allowed_regret_ratio(self, value):
        self._max_allowed_regret_ratio = value

    @property
    def reward_distribution_def(self):
        return self._reward_distribution_def

    @reward_distribution_def.setter
    def reward_distribution_def(self, value):
        self._reward_distribution_def = value

    @property
    def candidates_vs_context_stats(self):
        return self._variants_vs_givens_stats

    @candidates_vs_context_stats.setter
    def candidates_vs_context_stats(self, value):
        self._variants_vs_givens_stats = value

    def __init__(self, data_definition_json_path: str, track_url: str):

        # init values - if applicable later refactor to properties
        self.dataset_name = None
        self.timespan = None
        self.reward_cache = None
        self.max_allowed_regret_ratio = None
        self.reward_distribution_def = None
        self.candidates_vs_context_stats = None

        self.special_values = ['#any#', '#other#']

        self.candidates_definition = None
        self.candidates = None
        self.context_definition = None
        self.all_contexts = None
        self.context_fraction = None
        self.reward_mapping = None
        self.records_per_epoch = None
        self.epochs = None
        self.data_seed = None
        self.epochs_timespans = None
        self.track_url = track_url

        self.encoding_noise = None
        self.model_seed = None

        self._data_definition_json_path = data_definition_json_path
        self._load_data_definition()
        self._unpack_data_definition()

        self.init_reward_cache()

        np.random.seed(self.data_seed)
        self._unpack_candidates()
        self._unpack_contexts()
        self._unpack_contexts_probabilities()

        # check and set rewards mapping
        self._process_rewards_mapping()
        # cache best variant per givens for test's sake
        # self._set_per_givens_summary()

        self._rewarded_records = None

        self._generate_epochs_timespans()

    def _process_rewards_mapping(self):

        if self.reward_mapping:
            return
        else:
            if not self.reward_distribution_def:
                raise ValueError('missing both reward mapping and reward distribution definition')

            distribution_processor_name = \
                self.reward_distribution_def.get('distribution_processor_name', None)
            if not distribution_processor_name:
                raise ValueError('no distribution processor specified')

            distribution_processor = getattr(self, distribution_processor_name)
            distribution_processor()

    def _set_per_givens_summary(self):

        givens_key_parts = \
            ['|#any#'] + (
                ['|{}'.format(el_idx) for el_idx, _ in enumerate(self.all_contexts)]
                if self.all_contexts is not None else [])

        print(givens_key_parts)

        self.candidates_vs_context_stats = {
            kp: {
                'min_reward': min(
                    [v for k, v in self.reward_mapping.items() if kp == '|{}'.format(k.split('|')[-1])] or 0.0),
                'max_reward': max(
                    [v for k, v in self.reward_mapping.items() if kp == '|{}'.format(k.split('|')[-1])] or 0.0),
                'mean_reward': np.mean(
                    [v for k, v in self.reward_mapping.items() if kp == '|{}'.format(k.split('|')[-1])] or 0.0),
                'median_reward': np.median(
                    [v for k, v in self.reward_mapping.items() if kp == '|{}'.format(k.split('|')[-1])] or 0.0),
                'best_variant':
                    self.candidates[np.argmax([v for k, v in self.reward_mapping.items() if kp == '|{}'.format(k.split('|')[-1])])]
            } for kp in givens_key_parts}

    def _load_data_definition(self):
        with open(self.data_definition_json_path, 'r') as tcjf:
            self.data_definition = json.loads(tcjf.read())

        print(self.data_definition)

    def get_dataset_name(self):
        return self.data_definition['dataset_name']

    def init_reward_cache(self):
        self.reward_cache = \
            {'epoch_{}'.format(epoch_index): {
                'max_reward': 0, 'achieved_reward': 0, 'regret': 0}
             for epoch_index in range(self.epochs)}

    def _unpack_data_definition(self):

        for data_def_key in self.data_definition.keys():
            setattr(self, data_def_key, self.data_definition.get(data_def_key, None))

    def _get_probabilities(self, collection: list, distribution_name: str):

        # to be used with np.random.choice([...], probabilities)
        if distribution_name == 'uniform':
            return np.full((len(collection), ), 1 / len(collection))

        # future -> support other distributions

    def _generate_epochs_timespans(self):
        np.random.seed(self.data_seed)
        start_datetime = pd.to_datetime(self.timespan.get('data_starts', None))
        end_datetime = pd.to_datetime(self.timespan.get('data_ends', None))

        total_timespan = end_datetime - start_datetime
        sorted_timedeltas = \
            sorted((np.random.rand(self.epochs - 1) * np.full((self.epochs - 1, ), total_timespan)).tolist())
        epochs_limits = \
            [start_datetime] + \
            (np.full((self.epochs - 1, ), start_datetime) + np.array(sorted_timedeltas)).tolist() + \
            [end_datetime]

        self.epochs_timespans = {
            'epoch_{}'.format(epoch_idx - 1): [
                epochs_limits[epoch_idx - 1], epochs_limits[epoch_idx]]
            for epoch_idx in range(1, self.epochs + 1)}

    def _generate_record_timestamps_for_epoch(self, epoch_index: int):

        np.random.seed(self.data_seed)
        if not self.epochs_timespans:
            print('making timespans')
            self._generate_epochs_timespans()

        epoch_starts = self.epochs_timespans['epoch_{}'.format(epoch_index)][0]
        epoch_ends = self.epochs_timespans['epoch_{}'.format(epoch_index)][1]
        epoch_duration = epoch_ends - epoch_starts

        np.random.seed(epoch_index)

        timedelta_fractions = np.random.rand(self.records_per_epoch - 1)

        epoch_timestamps = \
            sorted(
                (timedelta_fractions * np.full((self.records_per_epoch - 1,), epoch_duration) +
                 np.full((self.records_per_epoch - 1,), epoch_starts)).tolist())

        records_timestamps = [epoch_starts] + epoch_timestamps

        return records_timestamps

    def _unpack_candidates(self):
        # determine if candidates should be produced or are provided
        candidates = self.candidates_definition.get('values', None)
        if candidates is None:
            eval_call = self.candidates_definition.get('eval_call', None)
            print(eval_call)
            candidates = eval(eval_call)

        self.candidates = candidates

    def _unpack_contexts(self):
        all_contexts = self.context_definition.get('values', None)
        if all_contexts is None:
            eval_call = self.context_definition.get('eval_call', None)
            if eval_call is not None:
                all_contexts = eval(eval_call)

        self.all_contexts = all_contexts

    def _unpack_contexts_probabilities(self):
        contexts_distribution_name = \
            self.context_definition.get('distribution_name', None)

        if contexts_distribution_name is None or self.all_contexts is None:
            self.all_contexts_probabilities = None
            return
            # raise ValueError('Givens distributions is None')

        contexts_distribution_name = \
            contexts_distribution_name.replace('#', '').strip()

        self.all_contexts_probabilities = \
            self._get_probabilities(
                collection=self.all_contexts, distribution_name=contexts_distribution_name)

        # future -> code for givens dependent on variant

    def _choose_context_index(self):

        if self.all_contexts is None:
            return None

        if self.context_fraction < np.random.rand():
            return None

        all_contexts_count = len(self.all_contexts)
        return np.random.choice(range(all_contexts_count), p=self.all_contexts_probabilities)

    def _get_rewards_for_items_and_context_indices(self, candidates_indices, context_index):
        return [
            self._get_record_reward_from_dict(
                item_index=item_index, context_index=context_index)
            for item_index in candidates_indices]

    def _get_highest_reward_item_and_context(self):
        if self.reward_mapping:
            highest_reward = max(self.reward_mapping.values())

            highest_reward_items_and_contexts = \
                [rm_k for rm_k, rm_v in self.reward_mapping.items() if rm_v == highest_reward]

            chosen_item_context = np.random.choice(highest_reward_items_and_contexts).split('|')

            # return best variant and best givens
            return self.candidates[int(chosen_item_context[0])], \
                self.all_contexts[int(chosen_item_context[1])]

        return None

    def make_decisions_for_epoch(
            self, epoch_index: int, reward_tracker: RewardTracker, scorer: Scorer = None):

        request_body_container = {'body': None}
        records = []

        def custom_matcher(request):
            request_dict = deepcopy(request.json())
            request_body_container['body'] = request_dict
            return True

        np.random.seed(self.data_seed)

        # history_ids = self._get_history_ids()

        # get epoch's timestamps
        epoch_timestamps = \
            self._generate_record_timestamps_for_epoch(epoch_index=epoch_index)

        print('\n### GENERATING RECORDS ###')
        # for each timestamp:

        max_reward = 0
        min_reward = 0
        achieved_reward = 0

        for record_timestamp in tqdm(epoch_timestamps):
            #  - choose context randomly or according to strategy defined in json if
            #    context are provided; should candidates be 'trimmed' only to those
            #    which can occur with provided context?4

            context_index = self._choose_context_index()

            # does DecisionModel have chooser - if not shuffle candidates
            candidates = self.candidates
            candidates_indices = range(len(candidates))

            if scorer is None:
                candidates = self.candidates.copy()
                candidates_with_indices = \
                    np.array([[v_idx, v] for v_idx, v in enumerate(candidates)])
                np.random.shuffle(candidates_with_indices)

                candidates_indices, candidates = \
                    candidates_with_indices[:, 0].astype(int).tolist(), \
                    candidates_with_indices[:, 1].tolist()

            possible_rewards = \
                self._get_rewards_for_items_and_context_indices(
                    candidates_indices=candidates_indices, context_index=context_index)

            max_possible_reward = max(possible_rewards)
            min_possible_reward = min(possible_rewards)

            if scorer is None:
                current_regret = max_reward - achieved_reward
                max_possible_regret = max_reward - min_reward

                # possible only for the first epoch when there is no model yet
                if self.max_allowed_regret_ratio and epoch_index == 0 and max_possible_regret \
                        and current_regret / max_possible_regret > self.max_allowed_regret_ratio:
                    # print('boosting')

                    currently_best_item = candidates[np.argmax(possible_rewards)]
                    # make the best variant first variant
                    candidates.remove(currently_best_item)
                    candidates = [currently_best_item] + candidates

            #  - create a Decision object and call get() with mockup tracker endpoint
            context = self.all_contexts[context_index] \
                if self.all_contexts is not None and context_index is not None \
                else None

            # self.__scores = self.model.score(candidates=self.candidates, context=self.context)
            # if scorer is None choose random candidate
            if scorer is None:
                best_item_index = np.random.randint(len(candidates))
            else:
                best_item_index = np.argmax(scorer.score(items=candidates, context=context))

            chosen_item = candidates[best_item_index]

            reward = self._get_record_reward_from_dict(
                self.candidates.index(chosen_item), context_index=context_index)

            # {
            # 'message_id': '8f52aa37-3f44-45d0-9804-ea47779b1f03',
            # 'model': 'test-model',
            # 'variant': 'eZGj',
            # 'count': 1000,
            # 'sample': 'WqYI',
            # 'reward': 0.0}

            record = {
                'decision_id': str(Ksuid()),  # str(uuid4()),
                # 'type': 'decision',
                'model': reward_tracker.model_name,
                'item': chosen_item,
                'count': len(candidates)}

            if context is not None:
                record['context'] = context

            if len(candidates) > 1:
                record['sample'] = \
                    reward_tracker._get_sample(item=chosen_item, candidates=candidates)

            record['reward'] = reward

            max_reward += max_possible_reward
            min_reward += min_possible_reward
            # increment achieved reward
            achieved_reward += reward
            records.append(record)

        self.reward_cache['epoch_{}'.format(epoch_index)]['max_reward'] = max_reward
        self.reward_cache['epoch_{}'.format(epoch_index)]['achieved_reward'] = achieved_reward
        self.reward_cache['epoch_{}'.format(epoch_index)]['regret'] = max_reward - achieved_reward

        return records

    def _get_record_reward_from_dict(self, item_index: object, context_index: object):
        if context_index is None:
            context_index = "#any#"
        reward_key = '{}|{}'.format(item_index, context_index)
        reward = self.reward_mapping.get(reward_key, None)
        if reward is None:
            not_exact_reward_key = '{}|{}'.format(item_index, '#any#')
            reward = self.reward_mapping.get(not_exact_reward_key, 0.0)
        return reward

    def _get_rewards_keys(self, all_contexts_indices):
        return \
            ['{}|{}'.format(i_idx, c_idx) for c_idx in all_contexts_indices
             for i_idx, v in enumerate(self.candidates)]

    def _get_rewards_for_distribution(self, distribution, x_values):
        return [distribution.pdf(x) for x in x_values]

    def _set_uniform_fixed_rewards(self):

        max_reward = self.reward_distribution_def.get('max_reward', 1.0)

        all_givens_indices = \
            ['#any#'] + ([] if self.all_contexts is None else list(range(len(self.all_contexts))))

        rewards_keys = self._get_rewards_keys(
            all_contexts_indices=all_givens_indices)
        rewards_values = [max_reward for _ in range(len(rewards_keys))]
        reward_mapping = {k: v for k, v in zip(rewards_keys, rewards_values)}
        self.reward_mapping = reward_mapping

    def _set_staircase_fixed_rewards(self):
        #                           ---__--__----- ...
        #                           |   givens 1
        #           ___---___---_---|
        #           |   givens 0
        # -__--__---|
        # None givens

        max_reward = self.reward_distribution_def.get('max_reward', None)
        step_increment = self.reward_distribution_def.get('step_increment', 1.0)
        reward_shift = self.reward_distribution_def.get('reward_shift', 0.0)
        shuffle_rewards = self.reward_distribution_def.get('shuffle_rewards', False)

        all_givens_indices = \
            ['#any#'] + ([] if self.all_contexts is None else list(range(len(self.all_contexts))))

        rewards_keys = self._get_rewards_keys(
            all_contexts_indices=all_givens_indices)
        rewards = \
            [(g_idx + np.random.rand() / 20) * step_increment + reward_shift
             for g_idx, gi in enumerate(all_givens_indices) for v_idx, v in enumerate(self.candidates)]

        scaled_rewards = self._scale_rewards(rewards=rewards, max_reward=max_reward)
        rewards = \
            self._shift_rewards(rewards=scaled_rewards, reward_shift=reward_shift)
        self._shuffle_rewards(rewards=rewards, shuffle_rewards=shuffle_rewards)

        rewards_mapping = {k: v for k, v in zip(rewards_keys, rewards)}
        self.reward_mapping = rewards_mapping

    def _set_normal_fixed_rewards(self):

        max_reward = self.reward_distribution_def.get('max_reward', None)
        mean = self.reward_distribution_def.get('mean', 0.0)
        sd = self.reward_distribution_def.get('sd', 1.0)
        reward_shift = self.reward_distribution_def.get('reward_shift', 0.0)
        min_x = self.reward_distribution_def.get('min_x', -5.0)
        max_x = self.reward_distribution_def.get('max_x', 5.0)
        shuffle_rewards = self.reward_distribution_def.get('shuffle_rewards', False)

        all_givens_indices = \
            ['#any#'] + ([] if self.all_contexts is None else list(range(len(self.all_contexts))))

        rewards_keys = self._get_rewards_keys(
            all_contexts_indices=all_givens_indices)

        rewards_count = len(rewards_keys)
        # get all variant - givens pairs (including no givens)
        x_values = np.linspace(min_x, max_x, rewards_count)
        pdf_values = \
            self._get_rewards_for_distribution(
                distribution=stats.norm(mean, sd), x_values=x_values)

        scaled_rewards = self._scale_rewards(rewards=pdf_values, max_reward=max_reward)
        rewards = \
            self._shift_rewards(rewards=scaled_rewards, reward_shift=reward_shift)
        self._shuffle_rewards(rewards=rewards, shuffle_rewards=shuffle_rewards)

        rewards_mapping = {k: v for k, v in zip(rewards_keys, rewards)}
        self.reward_mapping = rewards_mapping

    def _set_inverted_normal_fixed_rewards(self):
        max_reward = self.reward_distribution_def.get('max_reward', None)
        mean = self.reward_distribution_def.get('mean', 0.0)
        sd = self.reward_distribution_def.get('sd', 1.0)
        reward_shift = self.reward_distribution_def.get('reward_shift', 0.0)
        min_x = self.reward_distribution_def.get('min_x', -5.0)
        max_x = self.reward_distribution_def.get('max_x', 5.0)
        shuffle_rewards = self.reward_distribution_def.get("shuffle_rewards", False)

        all_givens_indices = \
            ['#any#'] + ([] if self.all_contexts is None else list(range(len(self.all_contexts))))

        rewards_keys = self._get_rewards_keys(
            all_contexts_indices=all_givens_indices)

        rewards_count = len(rewards_keys)
        # get all variant - givens pairs (including no givens)
        x_values = np.linspace(min_x, max_x, rewards_count)
        inverse_pdf_values = \
            [1 / el for el in self._get_rewards_for_distribution(
                distribution=stats.norm(mean, sd), x_values=x_values)]

        scaled_rewards = self._scale_rewards(rewards=inverse_pdf_values, max_reward=max_reward)
        rewards = \
            self._shift_rewards(rewards=scaled_rewards, reward_shift=reward_shift)
        self._shuffle_rewards(rewards=rewards, shuffle_rewards=shuffle_rewards)

        self.reward_mapping = {k: v for k, v in zip(rewards_keys, rewards)}

    def _set_exponential_fixed_rewards(self):
        max_reward = self.reward_distribution_def.get('max_reward', None)
        reward_shift = self.reward_distribution_def.get('reward_shift', 0.0)
        min_x = self.reward_distribution_def.get('min_x', 0.0)
        max_x = self.reward_distribution_def.get('max_x', 10.0)
        shuffle_rewards = self.reward_distribution_def.get('shuffle_givens', False)

        all_givens_indices = \
            ['#any#'] + ([] if self.all_contexts is None else list(range(len(self.all_contexts))))

        rewards_keys = self._get_rewards_keys(
            all_contexts_indices=all_givens_indices)
        x_values = np.linspace(min_x, max_x, len(rewards_keys))
        exp_values = self._get_rewards_for_distribution(
                distribution=stats.expon(), x_values=x_values)

        scaled_rewards = self._scale_rewards(rewards=exp_values, max_reward=max_reward)
        rewards = \
            self._shift_rewards(rewards=scaled_rewards, reward_shift=reward_shift)
        self._shuffle_rewards(rewards=rewards, shuffle_rewards=shuffle_rewards)

        self.reward_mapping = {k: v for k, v in zip(rewards_keys, rewards)}

    def _set_half_normal_fixed_rewards(self):
        max_reward = self.reward_distribution_def.get('max_reward', None)
        mean = self.reward_distribution_def.get('mean', 0.0)
        sd = self.reward_distribution_def.get('sd', 1.0)
        reward_shift = self.reward_distribution_def.get('reward_shift', 0.0)
        min_x = self.reward_distribution_def.get('min_x', -5.0)
        max_x = self.reward_distribution_def.get('max_x', 5.0)
        shuffle_rewards = self.reward_distribution_def.get('shuffle_givens', False)

        all_givens_indices = \
            ['#any#'] + ([] if self.all_contexts is None else list(range(len(self.all_contexts))))

        rewards_keys = self._get_rewards_keys(
            all_contexts_indices=all_givens_indices)
        x_values = np.linspace(min_x, max_x, len(rewards_keys))
        pdf_values = self._get_rewards_for_distribution(
                distribution=stats.halfnorm(mean, sd), x_values=x_values)

        scaled_rewards = self._scale_rewards(rewards=pdf_values, max_reward=max_reward)
        rewards = \
            self._shift_rewards(rewards=scaled_rewards, reward_shift=reward_shift)
        self._shuffle_rewards(rewards=rewards, shuffle_rewards=shuffle_rewards)

        self.reward_mapping = {k: v for k, v in zip(rewards_keys, rewards)}

    def _set_large_symmetric_fixed_rewards_with_zero_mean(self):
        abs_reward = self.reward_distribution_def.get('max_reward', 1000)
        if abs_reward is None:
            abs_reward = 1000

        all_givens_indices = \
            ['#any#'] + ([] if self.all_contexts is None else list(range(len(self.all_contexts))))

        rewards_keys = self._get_rewards_keys(
            all_contexts_indices=all_givens_indices)

        rewards = \
            [abs_reward * (-1)**(1 if np.random.rand() > 0.5 else 2) for _ in rewards_keys]

        self.reward_mapping = {k: v for k, v in zip(rewards_keys, rewards)}

    def _set_inverted_gauss_fixed_rewards(self):

        max_reward = self.reward_distribution_def.get('max_reward', None)
        mu = self.reward_distribution_def.get('mu', 0.1)
        reward_shift = self.reward_distribution_def.get('reward_shift', 0.0)
        min_x = self.reward_distribution_def.get('min_x', 0.0)
        if min_x < 0:
            warn('`min_x` < 0 -> setting to 0>')
            min_x = 0
        max_x = self.reward_distribution_def.get('max_x', 5.0)
        shuffle_rewards = self.reward_distribution_def.get('shuffle_rewards', False)

        all_givens_indices = \
            ([] if self.all_contexts is None else list(range(len(self.all_contexts)))) + ['#any#']

        raw_rewards_keys = self._get_rewards_keys(
            all_contexts_indices=all_givens_indices)

        rewards_keys = \
            [el for el in raw_rewards_keys if '#any#' not in el] + \
            [el for el in raw_rewards_keys if '#any#' in el]

        x_values = np.linspace(min_x, max_x, len(rewards_keys))
        pdf_values = self._get_rewards_for_distribution(
            distribution=stats.invgauss(mu), x_values=x_values)

        scaled_rewards = self._scale_rewards(rewards=pdf_values, max_reward=max_reward)
        rewards = \
            self._shift_rewards(rewards=scaled_rewards, reward_shift=reward_shift)
        self._shuffle_rewards(rewards=rewards, shuffle_rewards=shuffle_rewards)

        self.reward_mapping = {k: v for k, v in zip(rewards_keys, rewards)}

    def _scale_rewards(self, rewards, max_reward):
        if max_reward is not None:
            max_exp_value = max(rewards)
            scaling_factor = max_reward / max_exp_value
            rewards = [r * scaling_factor for r in rewards]

        return rewards

    def _shift_rewards(self, rewards, reward_shift):
        if reward_shift:
            rewards = [r + reward_shift for r in rewards]
        return rewards

    def _shuffle_rewards(self, rewards, shuffle_rewards):
        if shuffle_rewards:
            np.random.shuffle(rewards)

    def _set_variants_reward_distributions(self):
        # get mean (master) rewards distribution of givens
        # sample mean reward per givens
        # create reward distribution per givens using sampled means
        # (and additional definitions if present otherwise normal)
        # prepare a dict of arrays:
        # {'<variant index>': \
        # [[<reward 0>, <probability of reward 0>],
        #  [<reward 1>, <probability of reward 1>],...]
        # this will allow to call np.random.choice(rewards_w_probas[:, 0], p=rewards_w_probas[:, 1])
        pass

    def _set_givens_reward_distributions(self):
        # get mean (master) rewards distribution of givens
        # sample mean reward per givens
        # create reward distribution per givens using sampled means
        # (and additional definitions if present otherwise normal)
        # prepare a dict of arrays:
        # {'<givens index>':  \
        # [[<reward 0>, <probability of reward 0>],
        #  [<reward 1>, <probability of reward 1>],...]
        # this will allow to call np.random.choice(rewards_w_probas[:, 0], p=rewards_w_probas[:, 1])
        pass

    def dump_data(self, data: list, path: str):
        df = pd.DataFrame(data, columns=list(DF_SCHEMA.keys())).astype(DF_SCHEMA)
        for cn in [ITEM_KEY, CONTEXT_KEY, SAMPLE_KEY]:
            df[cn] = df[cn].apply(lambda x: orjson.dumps(x).decode('utf-8'))

        df.to_parquet(path, index=False, compression='ZSTD')


if __name__ == '__main__':

    from pprint import pprint

    track_url = 'http://tesst.track.url'
    test_path = \
        '../artifacts/data/synthetic_models_data_definitions/datasets_definitions/0_and_nan.json'
    q = BasicSemiRandomDataGenerator(data_definition_json_path=test_path, track_url=track_url)
    # q.records_per_epoch = 5000
    # pprint(q.data_definition)
    pprint(q.timespan)
    # q._generate_epochs_timespans()
    # ts = q._generate_record_timestamps_for_epoch(24)
    # print(ts[-3:])

    rt = RewardTracker(track_url=track_url, model_name='dummy-model')

    # self, epoch_index: int, reward_tracker: RewardTracker, scorer: Scorer = None

    records_00 = q.make_decisions_for_epoch(0, reward_tracker=rt, scorer=None)
    q.dump_data(records_00, 'dummy_decisions_00_new.parquet')

    pprint(q.reward_cache)

    # records_01 = q.make_decision_for_epoch(1, dm)
    # q.dump_data(records_01, 'dummy_decisions_01')


    # print(res)
    # records_10 = q.make_decision_for_epoch(0, dm)
    # q.dump_data(records_10, 'dummy_decisions_10')
    #
    # records_11 = q.make_decision_for_epoch(1, dm)
    # q.dump_data(records_11, 'dummy_decisions_11')


    # for epoch_idx in range(q.epochs):
    #     print(q.epochs_timespans['epoch_{}'.format(epoch_idx)])
    #     if epoch_idx > 0:
    #         assert q.epochs_timespans['epoch_{}'.format(epoch_idx - 1)][1] == q.epochs_timespans['epoch_{}'.format(epoch_idx)][0]
