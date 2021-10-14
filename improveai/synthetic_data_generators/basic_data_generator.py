from copy import deepcopy
import numpy as np
import hashlib
import json
import pandas as pd
import requests_mock as rqm
import string
from tqdm import tqdm

from improveai import Decision, DecisionModel, DecisionTracker

LETTERS = [letter for letter in string.ascii_letters]

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


class NotSoRandomGenerator:

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

    def __init__(self, data_definition_json_path: str, track_url: str):

        # init values - if applicable later refactor to properties
        self.dataset_name = None
        self.timespan = None

        self.variants_definition = None
        self.variants = None
        self.givens_definition = None
        self.variants_to_givens_mapping = None
        self.givens_fraction = None
        self.reward_mapping = None
        self.records_per_epoch = None
        self.epochs = None
        self.data_seed = None
        self.epochs_timespans = None
        self.track_url = track_url

        self.variants_probabilities = None
        self._all_givens_probabilities = None

        # TODO make sure this can be set in trainer
        self.encoding_noise = None
        self.model_seed = None

        self._data_definition_json_path = data_definition_json_path
        self._load_data_definition()
        self._unpack_data_definition()

        self._unpack_variants()
        self._unpack_variants_probabilities()

        self._unpack_givens()
        self._unpack_givens_probabilities()

        self._variants_to_givens_mapping = \
            self.data_definition.get('variants_to_givens_mapping', None)
        self._rewarded_records = None

        self._generate_epochs_timespans()

    def _load_data_definition(self):
        with open(self.data_definition_json_path, 'r') as tcjf:
            self.data_definition = json.loads(tcjf.read())

    def _unpack_data_definition(self):

        for data_def_key in self.data_definition.keys():
            setattr(
                self, data_def_key, self.data_definition.get(data_def_key, None))

    def _get_probabilities(self, collection: list, distribution_name: str):

        # TODO to be used with np.random.choice([...], probabilities)
        if distribution_name == 'uniform':
            return np.full((len(collection), ), 1/len(collection))

        # TODO support other distributions

    def _generate_epochs_timespans(self):
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

        return start_datetime

    def _generate_record_timestamps_for_epoch(self, epoch_index: int):

        if not self.epochs_timespans:
            self._generate_epochs_timespans()

        epoch_starts = self.epochs_timespans['epoch_{}'.format(epoch_index)][0]
        epoch_ends = self.epochs_timespans['epoch_{}'.format(epoch_index)][1]
        epoch_duration = epoch_ends - epoch_starts

        epoch_timestamps = \
            sorted(
                (np.random.rand(self.records_per_epoch) *
                 np.full((self.records_per_epoch,), epoch_duration) +
                 np.full((self.records_per_epoch,), epoch_starts)).tolist())

        records_timestamps = [epoch_starts] + epoch_timestamps

        return records_timestamps

    def _unpack_variants(self):
        # determine if variants should be produced or are provided
        variants = self.variants_definition.get('values', None)
        if variants is None:
            eval_call = self.variants_definition.get('eval_call', None)
            variants = eval(eval_call)

        self.variants = variants

    def _unpack_variants_probabilities(self):

        variants_distribution_name = \
            self.variants_definition.get('distribution_name', None)

        if variants_distribution_name is None:
            raise ValueError('Variants distributions is None')

        variants_distribution_name = \
            variants_distribution_name.replace('#', '').strip()

        self.variants_probabilities = \
            self._get_probabilities(
                collection=self.variants,
                distribution_name=variants_distribution_name)

    def _unpack_givens(self):
        all_givens = self.givens_definition.get('values', None)
        if all_givens is None:
            eval_call = self.givens_definition.get('eval_call', None)
            all_givens = eval(eval_call)

        self.all_givens = all_givens

    def _unpack_givens_probabilities(self):
        givens_distribution_name = \
            self.givens_definition.get('distribution_name', None)

        if givens_distribution_name is None:
            raise ValueError('Variants distributions is None')

        givens_distribution_name = \
            givens_distribution_name.replace('#', '').strip()

        self.all_givens_probabilities = \
            self._get_probabilities(
                collection=self.all_givens, distribution_name=givens_distribution_name)

    def _choose_givens(self):

        if self.givens_fraction < np.random.rand():
            return None

        if not self.variants_to_givens_mapping:
            return np.random.choice(self.all_givens, p=self.all_givens_probabilities)

        # TODO code for givens dependent on variant

    def _get_history_ids(self):
        # records count / avergae files per history / average records per history file
        histories_count = int(self.records_per_epoch / 35)
        np.random.seed(self.data_seed)

        history_ids = \
            [hashlib.sha256(
             ''.join(np.random.choice(LETTERS).tolist()).encode()).hexdigest()
             for _ in range(histories_count)]

        return history_ids

    def make_decision_for_epoch(
            self, epoch_index: int, decision_model: object):

        request_body_container = {'body': None}
        records = []

        np.random.seed(self.data_seed)
        history_ids = self._get_history_ids()

        def custom_matcher(request):
            request_dict = deepcopy(request.json())
            request_body_container['body'] = request_dict
            return True

        # get epoch's timestamps
        epoch_timestamps = \
            self._generate_record_timestamps_for_epoch(epoch_index=epoch_index)

        print('\n### GENERATING RECORDS ###')
        # for each timestamp:
        for record_timestamp in tqdm(epoch_timestamps):
            #  - choose givens randomly or according to strategy defined in json if
            #    givens are provided; should variants be 'trimmed' only to those
            #    which can occur with provided givens?4
            givens = self._choose_givens()

            # does DecisionModel have chooser - if not shuffle variants
            variants = self.variants.copy()

            if dm.chooser is None:
                np.random.shuffle(variants)

            #  - create a Decision object and call get() with mockup tracker endpoint
            d = \
                Decision(decision_model=decision_model)\
                .choose_from(variants).given(givens=givens)
            #  - mockup track endpoint
            with rqm.Mocker() as m:
                m.post(self.track_url, text='success',
                       additional_matcher=custom_matcher)
                #  - intercept body from get() call and append it to all records
                d.get()
                request_body = request_body_container['body']

                reward = \
                    self._assign_record_reward(
                        variant=request_body['variant'], givens=givens)

                request_body['timestamp'] = \
                    np.datetime_as_string(record_timestamp.to_datetime64())
                request_body['reward'] = reward
                request_body['history_id'] = np.random.choice(history_ids)

                records.append(request_body)
        return records

    def _assign_record_reward(self, variant: object, givens: object):
        if isinstance(self.reward_mapping, list):

            for mapped_reward_dict in self.reward_mapping:
                mapped_variant = mapped_reward_dict.get('variant', None)
                # if this is a wrong variant - continue
                if mapped_variant != variant:
                    continue

                mapped_givens = mapped_reward_dict.get('givens', None)

                if mapped_givens != givens and isinstance(mapped_givens, str) \
                        and '#any#' in mapped_givens:
                    return mapped_reward_dict['reward']
                elif mapped_givens == givens:
                    return mapped_reward_dict['reward']
                else:
                    continue

            return None

    def dump_data(self, data: list, path: str, mode='w'):

        with open(path, mode) as f:

            json_lines = [json.dumps(line) + '\n' for line in data]
            f.writelines(json_lines)


if __name__ == '__main__':

    from pprint import pprint

    track_url = 'http://tesst.track.url'
    test_path = \
        '../artifacts/data/synthetic_models/datasets_definitions/synth_data_def_0.json'
    q = NotSoRandomGenerator(
        data_definition_json_path=test_path, track_url=track_url)
    # pprint(q.data_definition)
    pprint(q.timespan)
    # q._generate_epochs_timespans()
    # ts = q._generate_record_timestamps_for_epoch(24)
    # print(ts[-3:])

    dt = DecisionTracker(track_url=track_url, history_id='dummy-history')
    dm = DecisionModel('test-model')
    dm.track_with(dt)

    records_0 = q.make_decision_for_epoch(0, dm)
    q.dump_data(records_0, 'dummy_decisions_0')

    records_1 = q.make_decision_for_epoch(0, dm)
    q.dump_data(records_1, 'dummy_decisions_1')
    # print(res)

    # for epoch_idx in range(q.epochs):
    #     print(q.epochs_timespans['epoch_{}'.format(epoch_idx)])
    #     if epoch_idx > 0:
    #         assert q.epochs_timespans['epoch_{}'.format(epoch_idx - 1)][1] == q.epochs_timespans['epoch_{}'.format(epoch_idx)][0]
