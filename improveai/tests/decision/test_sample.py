import json
import numpy as np
import os
from pytest import fixture, raises
import sys
from unittest import TestCase

# sys.path.append(os.sep.join(str(os.path.abspath(__file__)).split(os.sep)[:-3]))
sys.path.append(
    os.sep.join(str(os.path.abspath(__file__)).split(os.sep)[:-3]))

from decisions.v6 import Decision
from models.v6 import DecisionModel
from trackers.decision_trackers import DecisionTracker


class TestDecisionTracker(TestCase):

    @property
    def model_kind(self) -> str:
        return self._model_kind

    @model_kind.setter
    def model_kind(self, new_val: str):
        self._model_kind = new_val

    @property
    def model_pth(self) -> str:
        return self._model_pth

    @model_pth.setter
    def model_pth(self, new_val: str):
        self._model_pth = new_val

    @property
    def artifacts_model(self) -> DecisionModel:
        return self._artifacts_model

    @artifacts_model.setter
    def artifacts_model(self, new_val: DecisionModel):
        self._artifacts_model = new_val

    @property
    def max_runners_up(self):
        return self._max_runners_up

    @max_runners_up.setter
    def max_runners_up(self, new_val: int):
        self._max_runners_up = new_val

    @property
    def artifacts_track_url(self) -> str:
        return self._artifacts_track_url

    @artifacts_track_url.setter
    def artifacts_track_url(self, new_val: str):
        self._artifacts_track_url = new_val

    @property
    def artifacts_api_key(self) -> str:
        return self._artifacts_api_key

    @artifacts_api_key.setter
    def artifacts_api_key(self, new_val: str):
        self._artifacts_api_key = new_val

    @property
    def decision_tracker(self) -> DecisionTracker:
        return self._decision_tracker

    @decision_tracker.setter
    def decision_tracker(self, new_val: DecisionTracker):
        self._decision_tracker = new_val

    @property
    def variants_keys(self) -> list:
        return self._variants_keys

    @variants_keys.setter
    def variants_keys(self, new_val: list):
        self._variants_keys = new_val

    @property
    def model_seed_not_tracks(self) -> int:
        return self._model_seed_not_tracks

    @model_seed_not_tracks.setter
    def model_seed_not_tracks(self, new_val: int):
        self._model_seed_not_tracks = new_val

    @property
    def model_seed_tracks(self) -> int:
        return self._model_seed_tracks

    @model_seed_tracks.setter
    def model_seed_tracks(self, new_val: int):
        self._model_seed_tracks = new_val

    @property
    def artifacts_no_runners_up_best_variant(self) -> dict:
        return self._artifacts_no_runners_up_best_variants_pth

    @artifacts_no_runners_up_best_variant.setter
    def artifacts_no_runners_up_best_variant(self, new_val: dict):
        self._artifacts_no_runners_up_best_variants_pth = new_val

    @property
    def artifacts_no_runners_up_chosen_sample(self) -> list:
        return self._artifacts_no_runners_up_chosen_sample

    @artifacts_no_runners_up_chosen_sample.setter
    def artifacts_no_runners_up_chosen_sample(self, new_val: list):
        self._artifacts_no_runners_up_chosen_sample = new_val

    @property
    def sample_method_call_seed(self) -> int:
        return self._sample_method_call_seed

    @sample_method_call_seed.setter
    def sample_method_call_seed(self, new_val: int):
        self._sample_method_call_seed = new_val

    @property
    def model_ranked_benchmark_variants(self) -> list:
        return self._model_ranked_benchmark_variants

    @model_ranked_benchmark_variants.setter
    def model_ranked_benchmark_variants(self, new_val: list):
        self._model_ranked_benchmark_variants = new_val

    @property
    def scores_seed(self) -> int:
        return self._scores_seed

    @scores_seed.setter
    def scores_seed(self, new_val: int):
        self._scores_seed = new_val

    @property
    def artifacts_samples_w_runners_up_w_model(self) -> list:
        return self._artifacts_samples_w_runners_up_w_model

    @artifacts_samples_w_runners_up_w_model.setter
    def artifacts_samples_w_runners_up_w_model(self, new_val: list):
        self._artifacts_samples_w_runners_up_w_model = new_val

    @property
    def artifacts_samples_w_runners_up_no_model(self) -> list:
        return self._artifacts_samples_w_runners_up_no_model

    @artifacts_samples_w_runners_up_no_model.setter
    def artifacts_samples_w_runners_up_no_model(self, new_val: list):
        self._artifacts_samples_w_runners_up_no_model = new_val

    @fixture(autouse=True)
    def prepare_env(self):
        self.model_kind = os.getenv('DECISION_TRACKER_TESTS_MODEL_KIND')
        self.model_pth = os.getenv('MODEL_PTH')

        self.artifacts_model = \
            DecisionModel(model_kind=self.model_kind, model_pth=self.model_pth)

        with open(os.getenv('CONTEXT_JSON_PTH'), 'r') as cjson:
            read_str = ''.join(cjson.readlines())
            self.context = json.loads(read_str)

        with open(os.getenv('DECISION_TRACKER_TESTS_VARIANTS_PTH'), 'r') as mjson:
            read_str = ''.join(mjson.readlines())
            self.variants = json.loads(read_str)

        with open(os.getenv('DECISION_TRACKER_TESTS_BEST_PTH'), 'r') as bjson:
            read_str = ''.join(bjson.readlines())
            self.artifacts_no_runners_up_best_variant = json.loads(read_str)

        with open(os.getenv('DECISION_TRACKER_TESTS_SAMPLE_PTH'), 'r') as sjson:
            read_str = ''.join(sjson.readlines())
            self.artifacts_no_runners_up_chosen_sample = json.loads(read_str)

        with open(os.getenv('RANKED_VARIANTS_FOR_MODEL_PTH')) as rvmjson:
            read_str = ''.join(rvmjson.readlines())
            self.model_ranked_benchmark_variants = json.loads(read_str)

        with open(os.getenv('DECISION_TRACKER_TESTS_SAMPLE_W_RUNNERS_UP_W_MODEL_PTH')) as srmjson:
            read_str = ''.join(srmjson.readlines())
            self.artifacts_samples_w_runners_up_w_model = json.loads(read_str)

        with open(os.getenv('DECISION_TRACKER_TESTS_SAMPLE_W_RUNNERS_UP_NO_MODEL_PTH')) as srnmjson:
            read_str = ''.join(srnmjson.readlines())
            self.artifacts_samples_w_runners_up_no_model = json.loads(read_str)

        # DECISION_TRACKER_TESTS_SAMPLE_W_RUNNERS_UP_W_MODEL_PTH

        self.artifacts_modeln = \
            self.artifacts_model.chooser._get_model_metadata()['model']

        self.max_runners_up = 5

        self.decisions_kwargs = {
            # 'plain': Decision(),
            'variants_model': {
                'variants': self.variants[0], 'model': self.artifacts_model,
                'max_runners_up': self.max_runners_up},
            'variants_model_context': {
                'variants': self.variants[1], 'model': self.artifacts_model,
                'givens': self.context, 'max_runners_up': self.max_runners_up},
            'variants_model_context_nulled': {
                'variants': self.variants[2], 'model': self.artifacts_model,
                'givens': None, 'max_runners_up': self.max_runners_up},
            'ranked_variants_modeln': {
                'ranked_variants': self.variants[3],
                'model_name': self.artifacts_modeln,
                'max_runners_up': self.max_runners_up},
            'ranked_variants_modeln_context': {
                'ranked_variants': self.variants[4],
                'model_name': self.artifacts_modeln,
                'givens': self.context, 'max_runners_up': self.max_runners_up},
            'ranked_variants_modeln_context_nulled':
                {'ranked_variants': self.variants[5],
                 'model_name': self.artifacts_modeln, 'givens': None,
                 'max_runners_up': self.max_runners_up}}

        self.variants_keys = \
            ['variants' for _ in range(3)] + \
            ['ranked_variants' for _ in range(3)]

        self.artifacts_track_url = os.getenv('DUMMY_TEST_TRACK_URL')
        self.artifacts_api_key = os.getenv('DUMMY_TEST_API_KEY')
        self.decision_tracker = \
            DecisionTracker(track_url=self.artifacts_track_url,
                            api_key=self.artifacts_api_key)

        self.model_seed_not_tracks = int(os.getenv('MODEL_SEED_NOT_TRACKS'))
        self.model_seed_tracks = int(os.getenv('MODEL_SEED_TRACKS'))
        self.sample_method_call_seed = int(os.getenv('SAMPLE_METHOD_CALL_SEED'))
        self.scores_seed = int(os.getenv('SCORES_SEED'))

    def test_one_variant_sample_best_calc(self):

        for (case, dec_kwargs), variants_key in \
                zip(self.decisions_kwargs.items(), self.variants_keys):

            curr_variants = dec_kwargs.get(variants_key, None)
            assert curr_variants is not None

            dec_kwargs[variants_key] = curr_variants[:1]
            curr_dec = Decision(**dec_kwargs)

            curr_best = curr_dec.best()
            assert curr_best is not None

            curr_sample = \
                self.decision_tracker._get_sample(decision=curr_dec)

            assert curr_sample is None

    def test_one_variant_sample_best_not_calc(self):

        for (case, dec_kwargs), variants_key in \
                zip(self.decisions_kwargs.items(), self.variants_keys):

            curr_variants = dec_kwargs.get(variants_key, None)
            assert curr_variants is not None

            dec_kwargs[variants_key] = curr_variants[:1]
            curr_dec = Decision(**dec_kwargs)

            assert curr_dec.memoized_best is None

            with raises(NotImplementedError):
                self.decision_tracker._get_sample(decision=curr_dec)

    def test_sample_no_runners_up(self):
        # TODO make sure sample has length 1- is this correct ?

        for case, dec_kwargs in self.decisions_kwargs.items():

            np.random.seed(self.model_seed_not_tracks)
            curr_dec = Decision(**dec_kwargs)

            np.random.seed(self.scores_seed)
            curr_best = curr_dec.best()

            assert curr_best is not None
            assert curr_best == self.artifacts_no_runners_up_best_variant

            np.random.seed(self.sample_method_call_seed)
            curr_sample = self.decision_tracker._get_sample(decision=curr_dec)

            assert len(curr_sample) == 1

            np.testing.assert_array_equal(
                curr_sample, self.artifacts_no_runners_up_chosen_sample)

    def test_no_sample_has_runners_up_w_model(self):

        # TODO this sets max_runners_up to 10 for current test only to make
        #  sure there are no variants to sample from left

        des_decision_kwgs_keys = \
            ['variants_model', 'variants_model_context',
             'variants_model_context_nulled']

        variants_keys = ['variants' for _ in range(3)]

        des_decision_kwgs = \
            dict(zip(
                des_decision_kwgs_keys,
                [self.decisions_kwargs.get(k) for k in des_decision_kwgs_keys]))

        for (case, dec_kwargs), variants_key, ranked_v in \
                zip(des_decision_kwgs.items(), variants_keys,
                    self.model_ranked_benchmark_variants):

            curr_dec_variants = dec_kwargs.get(variants_key, None)

            assert curr_dec_variants is not None

            curr_test_max_runners_up = len(curr_dec_variants) - 1
            dec_kwargs['max_runners_up'] = curr_test_max_runners_up

            np.random.seed(self.model_seed_tracks)
            curr_dec = Decision(**dec_kwargs)

            np.random.seed(self.scores_seed)
            curr_best = curr_dec.best()

            assert curr_best is not None
            assert curr_best == ranked_v[0]

            top_runners_up = curr_dec.top_runners_up()
            np.testing.assert_array_equal(top_runners_up, ranked_v[1:])

            np.random.seed(self.sample_method_call_seed)
            curr_sample = self.decision_tracker._get_sample(decision=curr_dec)

            assert curr_sample is None

    def test_no_sample_has_runners_up_no_model(self):

        # TODO this sets max_runners_up to 10 for current test only to make
        #  sure there are no variants to sample from left

        des_decision_kwgs_keys = \
            ['ranked_variants_modeln', 'ranked_variants_modeln_context',
             'ranked_variants_modeln_context_nulled']

        ranked_variants_keys = ['ranked_variants' for _ in range(3)]

        des_decision_kwgs = \
            dict(zip(
                des_decision_kwgs_keys,
                [self.decisions_kwargs.get(k) for k in des_decision_kwgs_keys]))

        for (case, dec_kwargs), variants_key in \
                zip(des_decision_kwgs.items(), ranked_variants_keys):

            curr_dec_variants = dec_kwargs.get(variants_key, None)

            assert curr_dec_variants is not None

            curr_test_max_runners_up = len(curr_dec_variants) - 1
            dec_kwargs['max_runners_up'] = curr_test_max_runners_up

            np.random.seed(self.model_seed_tracks)
            curr_dec = Decision(**dec_kwargs)

            np.random.seed(self.scores_seed)
            curr_best = curr_dec.best()

            assert curr_best is not None
            assert curr_best == curr_dec_variants[0]

            top_runners_up = curr_dec.top_runners_up()
            np.testing.assert_array_equal(top_runners_up, curr_dec_variants[1:])

            np.random.seed(self.sample_method_call_seed)
            curr_sample = self.decision_tracker._get_sample(decision=curr_dec)

            assert curr_sample is None

    def test_sample_has_runners_up_w_model(self):

        # TODO this sets max_runners_up to 10 for current test only to make
        #  sure there are no variants to sample from left

        des_decision_kwgs_keys = \
            ['variants_model', 'variants_model_context',
             'variants_model_context_nulled']

        variants_keys = ['variants' for _ in range(3)]

        des_decision_kwgs = \
            dict(zip(
                des_decision_kwgs_keys,
                [self.decisions_kwargs.get(k) for k in des_decision_kwgs_keys]))

        for (case, dec_kwargs), variants_key, ranked_v, bench_sample in \
                zip(des_decision_kwgs.items(), variants_keys,
                    self.model_ranked_benchmark_variants,
                    self.artifacts_samples_w_runners_up_w_model):

            curr_dec_variants = dec_kwargs.get(variants_key, None)

            assert curr_dec_variants is not None

            np.random.seed(self.model_seed_tracks)
            curr_dec = Decision(**dec_kwargs)

            np.random.seed(self.scores_seed)
            curr_best = curr_dec.best()

            assert curr_best is not None
            assert curr_best == ranked_v[0]

            top_runners_up = curr_dec.top_runners_up()

            np.testing.assert_array_equal(
                top_runners_up, ranked_v[1:dec_kwargs['max_runners_up'] + 1])

            np.random.seed(self.sample_method_call_seed)
            curr_sample = self.decision_tracker._get_sample(decision=curr_dec)
            # print(curr_sample)
            np.testing.assert_array_equal(curr_sample, bench_sample)

            # assert curr_sample is None
        # assert False

    def test_sample_has_runners_up_no_model(self):

        # TODO this sets max_runners_up to 10 for current test only to make
        #  sure there are no variants to sample from left

        des_decision_kwgs_keys = \
            ['ranked_variants_modeln', 'ranked_variants_modeln_context',
             'ranked_variants_modeln_context_nulled']

        ranked_variants_keys = ['ranked_variants' for _ in range(3)]

        des_decision_kwgs = \
            dict(zip(
                des_decision_kwgs_keys,
                [self.decisions_kwargs.get(k) for k in des_decision_kwgs_keys]))

        for (case, dec_kwargs), variants_key in \
                zip(des_decision_kwgs.items(), ranked_variants_keys):

            curr_dec_variants = dec_kwargs.get(variants_key, None)

            assert curr_dec_variants is not None

            np.random.seed(self.model_seed_tracks)
            curr_dec = Decision(**dec_kwargs)

            np.random.seed(self.scores_seed)
            curr_best = curr_dec.best()

            assert curr_best is not None
            assert curr_best == curr_dec_variants[0]

            top_runners_up = curr_dec.top_runners_up()
            np.testing.assert_array_equal(
                top_runners_up, curr_dec_variants[1:dec_kwargs['max_runners_up'] + 1])

            np.random.seed(self.sample_method_call_seed)
            curr_sample = self.decision_tracker._get_sample(decision=curr_dec)

            np.testing.assert_array_equal(
                curr_sample, self.artifacts_samples_w_runners_up_no_model)
