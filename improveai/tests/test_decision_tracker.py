from copy import deepcopy
import json
from ksuid import Ksuid
import math
import numpy as np
import requests_mock as rqm
import os
from pytest import fixture, raises
import sys
from warnings import warn, catch_warnings, simplefilter

sys.path.append(
    os.sep.join(str(os.path.abspath(__file__)).split(os.sep)[:-3]))

import improveai.decision_tracker as dtr
from improveai.utils.general_purpose_tools import is_valid_ksuid


class TestDecisionTracker:

    @property
    def tracks_seed(self) -> int:
        return self._tracks_seed

    @tracks_seed.setter
    def tracks_seed(self, value: int):
        self._tracks_seed = value

    @property
    def not_tracks_seed(self):
        return self._not_tracks_seed

    @not_tracks_seed.setter
    def not_tracks_seed(self, value):
        self._not_tracks_seed = value

    @property
    def track_url(self) -> str:
        return self._track_url

    @track_url.setter
    def track_url(self, value: str):
        self._track_url = value

    @property
    def api_key(self) -> str:
        return self._api_key

    @api_key.setter
    def api_key(self, value: str):
        self._api_key = value

    @property
    def variants_count(self):
        return self._variants_count

    @variants_count.setter
    def variants_count(self, value):
        self._variants_count = value

    @property
    def variants(self):
        return self._variants

    @variants.setter
    def variants(self, value):
        self._variants = value
        
    @property
    def max_runners_up(self):
        return self._max_runners_up

    @max_runners_up.setter
    def max_runners_up(self, value):
        self._max_runners_up = value

    @property
    def sample_seed(self):
        return self._sample_seed

    @sample_seed.setter
    def sample_seed(self, value):
        self._sample_seed = value

    @fixture(autouse=True)
    def prep_env(self):

        self.tracks_seed = \
            int(os.getenv('DECISION_TRACKER_TRACKS_SEED'))

        self.not_tracks_seed = \
            int(os.getenv('DECISION_TRACKER_NOT_TRACKS_SEED'))

        self.track_url = os.getenv('DECISION_TRACKER_TEST_URL')
        self.api_key = os.getenv('DECISION_TRACKER_TEST_API_KEY')

        self.variants_count = \
            int(os.getenv('DECISION_TRACKER_SHOULD_TRACK_RUNNERS_UP_VARIANTS_COUNT'))

        self.variants = list(range(100))
        self.max_runners_up = \
            int(os.getenv('DECISION_TRACKER_MAX_RUNNERS_UP'))

        self.sample_seed = int(os.getenv('DECISION_TRACKER_SAMPLE_SEED'))
        self.first_sample_identical_with_variant_seed = \
            int(os.getenv('DECISION_TRACKER_FIRST_SAMPLE_IDENTICAL_WITH_VARIANT_SEED'))

        self.dummy_variant = {'dummy0': 'variant'}
        self.dummy_ranked_variants = \
            np.array(
                [self.dummy_variant] +
                [{'dummy{}'.format(el): 'variant'}
                 for el in range(self.max_runners_up + 10)])

        self.dummy_model_name = 'dummy-model'
        self.dummy_message_id = 'dummy_message'

    def test_should_track_runners_up_single_variant(self):

        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        assert decision_tracker._should_track_runners_up(variants_count=1) is False

    def test_should_track_runners_up_0_max_runners_up(self):

        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = 0

        np.random.seed(self.tracks_seed)
        assert decision_tracker._should_track_runners_up(variants_count=100) is False

    def test_should_track_runners_up_true(self):
        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        np.random.seed(self.tracks_seed)
        assert decision_tracker._should_track_runners_up(variants_count=self.variants_count) is True

    def test_should_track_runners_up_false(self):
        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        np.random.seed(self.not_tracks_seed)
        assert decision_tracker._should_track_runners_up(variants_count=self.variants_count) is False

    def test_should_track_runners_up_2_variants(self):
        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        np.random.seed(self.not_tracks_seed)
        assert decision_tracker._should_track_runners_up(variants_count=2) is True

    def test_top_runners_up(self):

        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        top_runners_up = \
            decision_tracker._top_runners_up(
                ranked_variants=self.variants)

        assert len(top_runners_up) == self.max_runners_up
        np.testing.assert_array_equal(
            top_runners_up, self.variants[1:self.max_runners_up + 1])

    def test_top_runners_up_empty_variants(self):

        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        top_runners_up = \
            decision_tracker._top_runners_up(ranked_variants=[])

        assert top_runners_up is None

    def test_top_runners_up_single_variant(self):

        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        ranked_variants = self.variants[:1]

        top_runners_up = \
            decision_tracker._top_runners_up(ranked_variants=ranked_variants)

        assert top_runners_up is None

    def test_top_runners_up_none_variant(self):

        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        ranked_variants = [None]

        top_runners_up = \
            decision_tracker._top_runners_up(ranked_variants=ranked_variants)

        assert top_runners_up is None

    def test_top_runners_up_less_variants_than_runners_up(self):

        max_runners_up = len(self.variants) + 10

        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = max_runners_up

        top_runners_up = \
            decision_tracker._top_runners_up(
                ranked_variants=self.variants)

        assert len(top_runners_up) == len(self.variants) - 1
        np.testing.assert_array_equal(
            top_runners_up, self.variants[1:])

    def test__is_sample_available_1_variant_no_runners_up(self):
        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        runners_up = None

        has_sample = decision_tracker._is_sample_available(
            variants=self.variants[:1], runners_up=runners_up)

        assert has_sample is False

    def test__is_sample_available_2_variants_1_runner_up(self):
        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        runners_up = self.variants[1:2]

        has_sample = decision_tracker._is_sample_available(
            variants=self.variants[:2], runners_up=runners_up)

        assert has_sample is False

    def test__is_sample_available_10_variants_9_runners_up(self):
        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        variants = self.variants[:10]
        runners_up = self.variants[1:10]

        has_sample = decision_tracker._is_sample_available(
            variants=variants, runners_up=runners_up)

        assert has_sample is False

    def test__is_sample_available_10_variants_3_runners_up(self):
        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        variants = self.variants[:10]
        runners_up = self.variants[1:4]

        has_sample = decision_tracker._is_sample_available(
            variants=variants, runners_up=runners_up)

        assert has_sample is True

    def test__is_sample_available_10_variants_8_runners_up(self):
        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        variants = self.variants[:10]
        runners_up = self.variants[1:9]

        has_sample = decision_tracker._is_sample_available(
            variants=variants, runners_up=runners_up)

        assert has_sample is True

    def test_get_sample_raises_for_zero_len_variants(self):
        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        ranked_variants = []
        variant = None
        track_runners_up = False

        with raises(AssertionError) as aerr:
            decision_tracker.get_sample(
                variant=variant, variants=ranked_variants,
                track_runners_up=track_runners_up)

    def test_get_sample_raises_for_1_variant_not_tracks(self):
        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        variant = self.variants[0]
        variants = self.variants[:1]
        track_runners_up = False

        with raises(AssertionError) as aerr:
            decision_tracker.get_sample(
                variant=variant, variants=variants, track_runners_up=track_runners_up)

    def test_get_sample_raises_for_1_variant_tracks(self):
        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        variant = self.variants[0]
        variants = self.variants[:1]
        track_runners_up = True

        with raises(AssertionError) as aerr:
            decision_tracker.get_sample(
                variant=variant, variants=variants, track_runners_up=track_runners_up)

    def test_get_sample_raises_for_2_variants_no_runners_up_positive_max_runners_up(self):
        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        variant = self.variants[0]
        variants = self.variants[:2]
        track_runners_up = True

        with raises(AssertionError) as aerr:
            decision_tracker.get_sample(
                variant=variant, variants=variants, track_runners_up=track_runners_up)

    def test_get_sample_raises_for_2_variants_tracks_0_max_runners_up(self):

        max_runners_up = 10

        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = max_runners_up

        variant = self.variants[0]
        variants = self.variants[:2]
        track_runners_up = True

        with raises(AssertionError) as aerr:
            decision_tracker.get_sample(
                variant=variant, variants=variants, track_runners_up=track_runners_up)

    def test_get_sample_raises_for_non_bool_track_runners_up_01(self):
        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        variant = self.variants[0]
        track_runners_up = None

        with raises(AssertionError) as ae:
            decision_tracker.get_sample(
                variant=variant, variants=self.variants,
                track_runners_up=track_runners_up)

    def test_get_sample_raises_for_non_bool_track_runners_up_02(self):
        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        variant = self.variants[0]
        ranked_variants = self.variants[:10]
        track_runners_up = None

        with raises(AssertionError) as ae:
            decision_tracker.get_sample(
                variant=variant, variants=ranked_variants,
                track_runners_up=track_runners_up)

    def test_get_sample_raises_on_track_runners_up_and_not_ranked_variants(self):
        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        ranked_variants = ['z', 'x', 'a', 'a', 'b', 'c']
        variant = ranked_variants[2]
        track_runners_up = True

        with raises(AssertionError) as aerr:
            decision_tracker.get_sample(
                variant=variant, variants=ranked_variants,
                track_runners_up=track_runners_up)

    def test_get_sample_raises_for_wrong_variants_type(self):
        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        ranked_variants = {'variants': [None, None, None]}
        variant = ranked_variants['variants'][0]
        track_runners_up = False

        with raises(TypeError) as terr:
            decision_tracker.get_sample(
                variant=variant, variants=ranked_variants,
                track_runners_up=track_runners_up)

    def test_get_sample_not_tracks(self):

        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        variant = self.variants[0]
        track_runners_up = False

        expected_sample = \
            int(os.getenv('DECISION_TRACKER_SAMPLE_NOT_TRACKS'))

        np.random.seed(self.sample_seed)
        sample = decision_tracker.get_sample(
            variant=variant, variants=self.variants,
            track_runners_up=track_runners_up)

        assert sample == expected_sample

    def test_get_sample_no_variants_not_tracks_3_variants(self):
        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        ranked_variants = self.variants[:3]
        variant = ranked_variants[0]
        track_runners_up = False

        sample = decision_tracker.get_sample(
            variant=variant, variants=ranked_variants,
            track_runners_up=track_runners_up)

        assert sample == ranked_variants[1]
        assert sample != variant

    def test_get_sample_no_variants_not_tracks_3_identical_variants(self):
        # theoretically for 2 variants we always have runners up
        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        ranked_variants = ['a', 'a', 'a']
        variant = ranked_variants[0]
        track_runners_up = False

        sample = decision_tracker.get_sample(
            variant=variant, variants=ranked_variants,
            track_runners_up=track_runners_up)

        assert sample is ranked_variants[1]
        assert sample == variant

    def test_get_sample_no_variants_not_tracks_identical_variants_01(self):
        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        ranked_variants = ['z', 'x', 'a', 'a', 'b', 'c']
        variant = ranked_variants[2]
        track_runners_up = False

        np.random.seed(self.sample_seed)
        sample = decision_tracker.get_sample(
            variant=variant, variants=ranked_variants,
            track_runners_up=track_runners_up)

        assert sample == ranked_variants[0]
        assert sample is not None

    def test_get_sample_no_variants_not_tracks_identical_variants_02(self):
        # make sure to sample identical index with selected variant's index
        # first
        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        ranked_variants = ['z', 'x', 'a', 'a', 'b', 'c']
        variant = ranked_variants[2]
        track_runners_up = False

        np.random.seed(self.first_sample_identical_with_variant_seed)
        sample = decision_tracker.get_sample(
            variant=variant, variants=ranked_variants,
            track_runners_up=track_runners_up)

        assert sample == ranked_variants[0]
        assert sample is not None

    def test_get_sample_3_none_variants_not_tracks(self):
        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        ranked_variants = [None, None, None]
        variant = ranked_variants[0]
        track_runners_up = False

        np.random.seed(self.sample_seed)
        sample = decision_tracker.get_sample(
            variant=variant, variants=ranked_variants,
            track_runners_up=track_runners_up)

        assert sample is None

    def test_get_sample_tracks(self):
        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        variant = self.variants[0]
        track_runners_up = True

        expected_sample = \
            int(os.getenv('DECISION_TRACKER_SAMPLE_TRACKS'))

        np.random.seed(self.sample_seed)
        sample = decision_tracker.get_sample(
            variant=variant, variants=self.variants,
            track_runners_up=track_runners_up)

        assert sample == expected_sample

    def test_get_sample_tracks_no_sample_01(self):

        # make sure there are no variants to sample from
        max_runners_up = len(self.variants) - 1

        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = max_runners_up

        variant = self.variants[0]
        track_runners_up = True

        with raises(AssertionError) as ae:
            decision_tracker.get_sample(
                variant=variant, variants=self.variants,
                track_runners_up=track_runners_up)

    def test_get_sample_tracks_no_sample_02(self):
        # make sure there are no variants to sample from
        max_runners_up = len(self.variants) - 1

        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = max_runners_up

        variant = self.variants[0]
        ranked_variants = self.variants[:40]
        track_runners_up = True

        with raises(AssertionError) as ae:
            decision_tracker.get_sample(
                variant=variant, variants=ranked_variants,
                track_runners_up=track_runners_up)

    def test_get_sample_tracks_sample_is_last_variant(self):
        # make sure there are no variants to sample from
        max_runners_up = len(self.variants) - 1

        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = max_runners_up

        variant = self.variants[0]
        ranked_variants = self.variants + [100]
        track_runners_up = True

        sample = decision_tracker.get_sample(
            variant=variant, variants=ranked_variants,
            track_runners_up=track_runners_up)

        assert sample == ranked_variants[-1]

    def test_track_single_none_variant_none_given_no_runners_up(self):

        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        expected_track_body = {
            decision_tracker.TYPE_KEY: decision_tracker.DECISION_TYPE,
            decision_tracker.MODEL_KEY: self.dummy_model_name,
            decision_tracker.VARIANT_KEY: None,
            decision_tracker.VARIANTS_COUNT_KEY: 1,
        }

        expected_request_json = json.dumps(expected_track_body, sort_keys=False)

        def custom_matcher(request):

            request_dict = deepcopy(request.json())
            del request_dict[decision_tracker.MESSAGE_ID_KEY]

            if json.dumps(request_dict, sort_keys=False) != \
                    expected_request_json:

                print('request body:')
                print(request.text)
                print('expected body:')
                print(expected_request_json)
                return None
            return True

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success',
                   additional_matcher=custom_matcher)

            with catch_warnings(record=True) as w:
                simplefilter("always")
                decision_id = decision_tracker.track(
                    variant=None,
                    variants=[None],
                    givens=None, model_name=self.dummy_model_name,
                    variants_ranked_and_track_runners_up=False)
                assert len(w) == 0

            assert is_valid_ksuid(decision_id)

    def test_track_none_given_no_runners_up(self):

        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        variants = [1, 2, 3]
        variant = variants[0]
        expected_sample = 2

        expected_track_body = {
            decision_tracker.TYPE_KEY: decision_tracker.DECISION_TYPE,
            decision_tracker.MODEL_KEY: self.dummy_model_name,
            decision_tracker.VARIANT_KEY: variant,
            decision_tracker.VARIANTS_COUNT_KEY: len(variants),
            decision_tracker.SAMPLE_KEY: expected_sample}

        expected_request_json = json.dumps(expected_track_body, sort_keys=False)

        def custom_matcher(request):
            request_dict = deepcopy(request.json())
            del request_dict[decision_tracker.MESSAGE_ID_KEY]

            if json.dumps(request_dict, sort_keys=False) != \
                    expected_request_json:

                print('request body:')
                print(request.text)
                print('expected body:')
                print(expected_request_json)
                return None
            return True

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success',
                   additional_matcher=custom_matcher)

            with catch_warnings(record=True) as w:
                simplefilter("always")
                np.random.seed(self.sample_seed)
                decision_id = decision_tracker.track(
                    variant=variant,
                    variants=variants,
                    givens=None, model_name=self.dummy_model_name,
                    variants_ranked_and_track_runners_up=False)
                assert len(w) == 0

            assert is_valid_ksuid(decision_id)

    def test_track_none_given(self):

        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        variants = [el for el in range(1, 20, 1)]
        top_runners_up = variants[1:self.max_runners_up + 1]
        variant = variants[0]
        expected_sample = 12

        expected_track_body = {
            decision_tracker.TYPE_KEY: decision_tracker.DECISION_TYPE,
            decision_tracker.MODEL_KEY: self.dummy_model_name,
            decision_tracker.VARIANT_KEY: variants[0],
            decision_tracker.VARIANTS_COUNT_KEY: len(variants),
            decision_tracker.RUNNERS_UP_KEY: top_runners_up,
            decision_tracker.SAMPLE_KEY: expected_sample}

        expected_request_json = json.dumps(expected_track_body, sort_keys=False)

        def custom_matcher(request):
            request_dict = deepcopy(request.json())
            del request_dict[decision_tracker.MESSAGE_ID_KEY]

            if json.dumps(request_dict, sort_keys=False) != \
                    expected_request_json:

                print('request body:')
                print(request.text)
                print('expected body:')
                print(expected_request_json)
                return None
            return True

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success',
                   additional_matcher=custom_matcher)

            with catch_warnings(record=True) as w:
                simplefilter("always")
                np.random.seed(self.sample_seed)
                decision_id = decision_tracker.track(
                    variant=variant,
                    variants=variants,
                    givens=None, model_name=self.dummy_model_name,
                    variants_ranked_and_track_runners_up=True)
                assert len(w) == 0

            assert is_valid_ksuid(decision_id)

    def test_track_2_variants_01(self):

        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        variants = [0, 1]
        top_runners_up = [variants[1]]
        variant = variants[0]

        expected_track_body = {
            decision_tracker.TYPE_KEY: decision_tracker.DECISION_TYPE,
            decision_tracker.MODEL_KEY: self.dummy_model_name,
            decision_tracker.VARIANT_KEY: variant,
            decision_tracker.VARIANTS_COUNT_KEY: len(variants),
            decision_tracker.RUNNERS_UP_KEY: top_runners_up}

        expected_request_json = json.dumps(expected_track_body, sort_keys=False)

        def custom_matcher(request):
            request_dict = deepcopy(request.json())
            del request_dict[decision_tracker.MESSAGE_ID_KEY]

            if json.dumps(request_dict, sort_keys=False) != \
                    expected_request_json:

                print('request body:')
                print(request.text)
                print(request.json())
                print('expected body (minus message_id):')
                print(expected_track_body)
                return None
            return True

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success',
                   additional_matcher=custom_matcher)

            with catch_warnings(record=True) as w:
                simplefilter("always")
                np.random.seed(self.sample_seed)
                decision_id = decision_tracker.track(
                    variant=variant,
                    variants=variants,
                    givens=None, model_name=self.dummy_model_name,
                    variants_ranked_and_track_runners_up=True)

                assert len(w) == 0

            assert is_valid_ksuid(decision_id)

    def test_track_2_variants_02(self):

        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        variants = [0, 1]
        top_runners_up = [variants[0]]
        variant = variants[1]

        expected_track_body = {
            decision_tracker.TYPE_KEY: decision_tracker.DECISION_TYPE,
            decision_tracker.MODEL_KEY: self.dummy_model_name,
            decision_tracker.VARIANT_KEY: variant,
            decision_tracker.VARIANTS_COUNT_KEY: len(variants),
            decision_tracker.RUNNERS_UP_KEY: top_runners_up}

        expected_request_json = json.dumps(expected_track_body, sort_keys=False)

        def custom_matcher(request):
            request_dict = deepcopy(request.json())
            del request_dict[decision_tracker.MESSAGE_ID_KEY]

            if json.dumps(request_dict, sort_keys=False) != \
                    expected_request_json:

                print('request body:')
                print(request.text)
                print(request.json())
                print('expected body (minus message_id):')
                print(expected_track_body)
                return None
            return True

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success',
                   additional_matcher=custom_matcher)

            with catch_warnings(record=True) as w:
                simplefilter("always")
                np.random.seed(self.sample_seed)
                decision_id = decision_tracker.track(
                    variant=variant,
                    variants=variants,
                    givens=None, model_name=self.dummy_model_name,
                    variants_ranked_and_track_runners_up=True)
                assert len(w) == 0

            assert is_valid_ksuid(decision_id)

    def test_track_2_variants_03(self):

        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        variants = [0, 1]
        top_runners_up = [variants[1]]
        variant = variants[0]
        variants_ranked_and_track_runners_up = True

        expected_track_body = {
            decision_tracker.TYPE_KEY: decision_tracker.DECISION_TYPE,
            decision_tracker.MODEL_KEY: self.dummy_model_name,
            decision_tracker.VARIANT_KEY: variant,
            decision_tracker.VARIANTS_COUNT_KEY: len(variants),
            decision_tracker.RUNNERS_UP_KEY: top_runners_up}

        expected_request_json = json.dumps(expected_track_body, sort_keys=False)

        def custom_matcher(request):
            request_dict = deepcopy(request.json())
            del request_dict[decision_tracker.MESSAGE_ID_KEY]

            if json.dumps(request_dict, sort_keys=False) != \
                    expected_request_json:

                print('request body:')
                print(request.text)
                print(request.json())
                print('expected body (minus message_id):')
                print(expected_track_body)
                return None
            return True

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success',
                   additional_matcher=custom_matcher)

            with catch_warnings(record=True) as w:
                simplefilter("always")
                np.random.seed(self.sample_seed)
                decision_id = decision_tracker.track(
                    variant=variant,
                    variants=variants,
                    givens=None, model_name=self.dummy_model_name,
                    variants_ranked_and_track_runners_up=variants_ranked_and_track_runners_up)
                assert len(w) == 0

            assert is_valid_ksuid(decision_id)

    def test_track_2_variants_no_runners_up_sample(self):

        max_runners_up = 0

        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = max_runners_up

        variants = [0, 1]
        variant = variants[1]
        sample = variants[0]
        variants_ranked_and_track_runners_up = True

        expected_track_body = {
            decision_tracker.TYPE_KEY: decision_tracker.DECISION_TYPE,
            decision_tracker.MODEL_KEY: self.dummy_model_name,
            decision_tracker.VARIANT_KEY: variant,
            decision_tracker.VARIANTS_COUNT_KEY: len(variants),
            decision_tracker.SAMPLE_KEY: sample}

        expected_request_json = json.dumps(expected_track_body, sort_keys=False)

        def custom_matcher(request):
            request_dict = deepcopy(request.json())
            del request_dict[decision_tracker.MESSAGE_ID_KEY]

            if json.dumps(request_dict, sort_keys=False) != \
                    expected_request_json:

                print('request body:')
                print(request.text)
                print(request.json())
                print('expected body (minus message_id):')
                print(expected_track_body)
                return None
            return True

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success',
                   additional_matcher=custom_matcher)

            with catch_warnings(record=True) as w:
                simplefilter("always")
                np.random.seed(self.sample_seed)
                decision_id = decision_tracker.track(
                    variant=variant,
                    variants=variants,
                    givens=None, model_name=self.dummy_model_name,
                    variants_ranked_and_track_runners_up=variants_ranked_and_track_runners_up)
                assert len(w) == 0

            assert is_valid_ksuid(decision_id)

    def test_track_2_variants_raises_for_variants_ranked_and_track_runners_up_false(self):
        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        variants = [0, 1]
        variant = variants[0]
        variants_ranked_and_track_runners_up = False

        with raises(AssertionError) as ae:
            decision_tracker.track(
                variant=variant,
                variants=variants,
                givens=None, model_name=self.dummy_model_name,
                variants_ranked_and_track_runners_up=variants_ranked_and_track_runners_up)

    def test_track_2_variants_raises_for_positive_max_runners_up(self):

        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        variants = [0, 1]
        top_runners_up = [variants[0]]
        variant = variants[1]
        variants_ranked_and_track_runners_up = True

        expected_track_body = {
            decision_tracker.TYPE_KEY: decision_tracker.DECISION_TYPE,
            decision_tracker.MODEL_KEY: self.dummy_model_name,
            decision_tracker.VARIANT_KEY: variant,
            decision_tracker.VARIANTS_COUNT_KEY: len(variants),
            decision_tracker.RUNNERS_UP_KEY: top_runners_up}

        expected_request_json = json.dumps(expected_track_body, sort_keys=False)

        def custom_matcher(request):
            request_dict = deepcopy(request.json())
            del request_dict[decision_tracker.MESSAGE_ID_KEY]

            if json.dumps(request_dict, sort_keys=False) != \
                    expected_request_json:

                print('request body:')
                print(request.text)
                print(request.json())
                print('expected body (minus message_id):')
                print(expected_track_body)
                return None
            return True

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success',
                   additional_matcher=custom_matcher)

            with catch_warnings(record=True) as w:
                simplefilter("always")
                np.random.seed(self.sample_seed)
                decision_tracker.track(
                    variant=variant,
                    variants=variants,
                    givens=None, model_name=self.dummy_model_name,
                    variants_ranked_and_track_runners_up=variants_ranked_and_track_runners_up)
                assert len(w) == 0

    def test_track_2_variants_zero_max_runners_up(self):

        max_runners_up = 0

        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = max_runners_up

        variants = [0, 1]
        variant = variants[1]
        variants_ranked_and_track_runners_up = True
        expected_sample = variants[0]

        expected_track_body = {
            decision_tracker.TYPE_KEY: decision_tracker.DECISION_TYPE,
            decision_tracker.MODEL_KEY: self.dummy_model_name,
            decision_tracker.VARIANT_KEY: variant,
            decision_tracker.VARIANTS_COUNT_KEY: len(variants),
            # decision_tracker.RUNNERS_UP_KEY: top_runners_up
            decision_tracker.SAMPLE_KEY: expected_sample}

        expected_request_json = json.dumps(expected_track_body, sort_keys=False)

        def custom_matcher(request):
            request_dict = deepcopy(request.json())
            del request_dict[decision_tracker.MESSAGE_ID_KEY]

            if json.dumps(request_dict, sort_keys=False) != \
                    expected_request_json:

                print('request body:')
                print(request.text)
                print(request.json())
                print('expected body (minus message_id):')
                print(expected_track_body)
                return None
            return True

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success',
                   additional_matcher=custom_matcher)

            with catch_warnings(record=True) as w:
                simplefilter("always")
                np.random.seed(self.sample_seed)
                decision_tracker.track(
                    variant=variant,
                    variants=variants,
                    givens=None, model_name=self.dummy_model_name,
                    variants_ranked_and_track_runners_up=variants_ranked_and_track_runners_up)
                assert len(w) == 0

    def test_track(self):

        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        variants = [el for el in range(1, 20, 1)]
        top_runners_up = variants[1:decision_tracker.max_runners_up + 1]
        variant = variants[0]
        givens = {'dummy': 'givens'}
        expected_sample = 12

        expected_track_body = {
            decision_tracker.TYPE_KEY: decision_tracker.DECISION_TYPE,
            decision_tracker.MODEL_KEY: self.dummy_model_name,
            decision_tracker.VARIANT_KEY: variants[0],
            decision_tracker.VARIANTS_COUNT_KEY: len(variants),
            decision_tracker.RUNNERS_UP_KEY: top_runners_up,
            decision_tracker.SAMPLE_KEY: expected_sample,
            decision_tracker.GIVENS_KEY: givens}

        expected_request_json = json.dumps(expected_track_body, sort_keys=False)

        def custom_matcher(request):
            request_dict = deepcopy(request.json())
            del request_dict[decision_tracker.MESSAGE_ID_KEY]

            if json.dumps(request_dict, sort_keys=False) != \
                    expected_request_json:

                print('request body:')
                print(request.text)
                print(request.json())
                print('expected body (minus message_id):')
                print(expected_track_body)
                return None
            return True

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success',
                   additional_matcher=custom_matcher)

            with catch_warnings(record=True) as w:
                simplefilter("always")
                np.random.seed(self.sample_seed)
                decision_id = decision_tracker.track(
                    variant=variant,
                    variants=variants,
                    givens=givens, model_name=self.dummy_model_name,
                    variants_ranked_and_track_runners_up=True)
                assert len(w) == 0

            assert is_valid_ksuid(decision_id)

    def test_track_best_not_first_ranked_and_tracks(self):

        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        variants = [el for el in range(1, 20, 1)]
        variant = variants[self.max_runners_up]
        givens = {'dummy': 'givens'}

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success')

            with raises(AssertionError) as ae:
                decision_tracker.track(
                    variant=variant,
                    variants=variants,
                    givens=givens, model_name=self.dummy_model_name,
                    variants_ranked_and_track_runners_up=True)

    def test_track_ndarray_variants(self):

        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        variants = [el for el in range(1, 20, 1)]
        top_runners_up = variants[1:self.max_runners_up + 1]
        variant = variants[0]
        givens = {'dummy': 'givens'}
        expected_sample = 12

        expected_track_body = {
            decision_tracker.TYPE_KEY: decision_tracker.DECISION_TYPE,
            decision_tracker.MODEL_KEY: self.dummy_model_name,
            decision_tracker.VARIANT_KEY: variants[0],
            decision_tracker.VARIANTS_COUNT_KEY: len(variants),
            decision_tracker.RUNNERS_UP_KEY: top_runners_up,
            decision_tracker.SAMPLE_KEY: expected_sample,
            decision_tracker.GIVENS_KEY: givens}

        expected_request_json = json.dumps(expected_track_body, sort_keys=False)

        def custom_matcher(request):
            request_dict = deepcopy(request.json())
            del request_dict[decision_tracker.MESSAGE_ID_KEY]

            if json.dumps(request_dict, sort_keys=False) != \
                    expected_request_json:

                print('request body:')
                print(request.text)
                print('expected body:')
                print(expected_request_json)
                return None
            return True

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success',
                   additional_matcher=custom_matcher)

            with catch_warnings(record=True) as w:
                simplefilter("always")
                np.random.seed(self.sample_seed)
                decision_id = decision_tracker.track(
                    variant=variant,
                    variants=np.array(variants),
                    givens=givens, model_name=self.dummy_model_name,
                    variants_ranked_and_track_runners_up=True)
                assert len(w) == 0

            assert is_valid_ksuid(decision_id)

    def test_track_2_variants_and_sample(self):

        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        variants = [el for el in range(1, 3, 1)]
        top_runners_up = variants[1:]
        variant = variants[0]
        expected_sample = None

        expected_track_body = {
            decision_tracker.TYPE_KEY: decision_tracker.DECISION_TYPE,
            decision_tracker.MODEL_KEY: self.dummy_model_name,
            decision_tracker.VARIANT_KEY: variants[0],
            decision_tracker.VARIANTS_COUNT_KEY: len(variants),
            decision_tracker.RUNNERS_UP_KEY: top_runners_up,
            # decision_tracker.SAMPLE_KEY: expected_sample
        }

        expected_request_json = json.dumps(expected_track_body, sort_keys=False)

        def custom_matcher(request):
            request_dict = deepcopy(request.json())
            del request_dict[decision_tracker.MESSAGE_ID_KEY]

            if json.dumps(request_dict, sort_keys=False) != \
                    expected_request_json:

                print('request body:')
                print(request.text)
                print('expected body:')
                print(expected_request_json)
                return None
            return True

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success',
                   additional_matcher=custom_matcher)

            with catch_warnings(record=True) as w:
                simplefilter("always")
                np.random.seed(self.sample_seed)
                decision_id = decision_tracker.track(
                    variant=variant,
                    variants=variants,
                    givens=None, model_name=self.dummy_model_name,
                    variants_ranked_and_track_runners_up=True)
                assert len(w) == 0

            assert is_valid_ksuid(decision_id)

    def test_track_2_variants_max_runners_up_0(self):

        max_runners_up = 0

        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = max_runners_up

        variants = [el for el in range(1, 3, 1)]
        variant = variants[0]
        expected_sample = variants[1]

        expected_track_body = {
            decision_tracker.TYPE_KEY: decision_tracker.DECISION_TYPE,
            decision_tracker.MODEL_KEY: self.dummy_model_name,
            decision_tracker.VARIANT_KEY: variants[0],
            decision_tracker.VARIANTS_COUNT_KEY: len(variants),
            decision_tracker.SAMPLE_KEY: expected_sample
        }

        expected_request_json = json.dumps(expected_track_body, sort_keys=False)

        def custom_matcher(request):
            request_dict = deepcopy(request.json())
            del request_dict[decision_tracker.MESSAGE_ID_KEY]

            if json.dumps(request_dict, sort_keys=False) != \
                    expected_request_json:

                print('request body:')
                print(request.text)
                print('expected body:')
                print(expected_request_json)
                return None
            return True

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success',
                   additional_matcher=custom_matcher)

            with catch_warnings(record=True) as w:
                simplefilter("always")
                np.random.seed(self.sample_seed)
                decision_id = decision_tracker.track(
                    variant=variant,
                    variants=variants,
                    givens=None, model_name=self.dummy_model_name,
                    variants_ranked_and_track_runners_up=True)
                assert len(w) == 0

            assert is_valid_ksuid(decision_id)

    def test_should_track_runners_up_2_variants_1(self):

        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        assert all(
            [decision_tracker._should_track_runners_up(variants_count=2)
             for _ in range(10)])

    def test_top_runners_up_2_variants(self):

        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        variants = [el for el in range(2)]

        top_runners_up = decision_tracker._top_runners_up(ranked_variants=variants)

        assert top_runners_up == [1]

    def test_track_2_variants_no_sample_max_runners_up_50(self):

        max_runners_up = 50

        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = max_runners_up

        variants = [el for el in range(0, 1)]
        variant = variants[0]

        expected_track_body = {
            decision_tracker.TYPE_KEY: decision_tracker.DECISION_TYPE,
            decision_tracker.MODEL_KEY: self.dummy_model_name,
            decision_tracker.VARIANT_KEY: variants[0],
            decision_tracker.VARIANTS_COUNT_KEY: len(variants),
        }

        expected_request_json = json.dumps(expected_track_body, sort_keys=False)

        def custom_matcher(request):
            request_dict = deepcopy(request.json())
            del request_dict[decision_tracker.MESSAGE_ID_KEY]

            if json.dumps(request_dict, sort_keys=False) != \
                    expected_request_json:

                print('request body:')
                print(request.text)
                print('expected body:')
                print(expected_request_json)
                return None
            return True

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success',
                   additional_matcher=custom_matcher)

            with catch_warnings(record=True) as w:
                simplefilter("always")
                np.random.seed(self.sample_seed)
                decision_id = decision_tracker.track(
                    variant=variant,
                    variants=variants,
                    givens=None, model_name=self.dummy_model_name,
                    variants_ranked_and_track_runners_up=
                    decision_tracker._should_track_runners_up(len(variants)))
                assert len(w) == 0

            assert is_valid_ksuid(decision_id)

    def test_track_invalid_model_name(self):

        with catch_warnings(record=True) as w:
            simplefilter("always")
            model_name = ''
            decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
            result = decision_tracker.track(
                variant=1, variants=[1, 2, 3], givens=None, model_name=model_name,
                variants_ranked_and_track_runners_up=False)
            assert len(w) > 0
            assert result is None

        with catch_warnings(record=True) as w:
            simplefilter("always")
            model_name = '!@#$%^&*()'
            decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
            result = decision_tracker.track(
                variant=1, variants=[1, 2, 3], givens=None, model_name=model_name,
                variants_ranked_and_track_runners_up=False)
            assert len(w) > 0
            assert result is None

        with catch_warnings(record=True) as w:
            simplefilter("always")
            model_name = ''.join(['a' for _ in range(65)])
            decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
            result = decision_tracker.track(
                variant=1, variants=[1, 2, 3], givens=None, model_name=model_name,
                variants_ranked_and_track_runners_up=False)
            assert len(w) > 0
            assert result is None

    def test_add_float_reward(self):

        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        decision_id_container = {'decision_id': None}

        def custom_matcher_caching_decision_id(request):
            request_dict = deepcopy(request.json())
            decision_id_container['decision_id'] = \
                request_dict[decision_tracker.MESSAGE_ID_KEY]
            return True

        variants = [el for el in range(0, 1)]
        variant = variants[0]

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success',
                   additional_matcher=custom_matcher_caching_decision_id)

            with catch_warnings(record=True) as w:
                simplefilter("always")
                np.random.seed(self.sample_seed)
                decision_id = decision_tracker.track(
                    variant=variant,
                    variants=variants,
                    givens=None, model_name=self.dummy_model_name,
                    variants_ranked_and_track_runners_up=
                    decision_tracker._should_track_runners_up(len(variants)))
                assert len(w) == 0

            assert is_valid_ksuid(decision_id)

        decision_id = decision_id_container['decision_id']
        reward = 1.0

        expected_add_reward_body = {
            decision_tracker.TYPE_KEY: decision_tracker.REWARD_TYPE,
            decision_tracker.MODEL_KEY: self.dummy_model_name,
            decision_tracker.REWARD_KEY: reward,
            decision_tracker.DECISION_ID_KEY: decision_id,
        }

        expected_request_json = json.dumps(expected_add_reward_body, sort_keys=False)

        def custom_matcher(request):
            request_dict = deepcopy(request.json())
            del request_dict[decision_tracker.MESSAGE_ID_KEY]

            if json.dumps(request_dict, sort_keys=False) != \
                    expected_request_json:

                print('raw request body:')
                print(request.text)
                print('compared request string')
                print(json.dumps(request_dict, sort_keys=False))
                print('expected body:')
                print(expected_request_json)
                return None
            return True

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success',
                   additional_matcher=custom_matcher)

            with catch_warnings(record=True) as w:
                simplefilter("always")
                decision_tracker.add_reward(
                    reward=reward, model_name=self.dummy_model_name, decision_id=decision_id)
                assert len(w) == 0

    def test_add_int_reward(self):

        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        decision_id_container = {'decision_id': None}

        def custom_matcher_caching_decision_id(request):
            request_dict = deepcopy(request.json())
            decision_id_container['decision_id'] = \
                request_dict[decision_tracker.MESSAGE_ID_KEY]
            return True

        variants = [el for el in range(0, 1)]
        variant = variants[0]

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success',
                   additional_matcher=custom_matcher_caching_decision_id)

            with catch_warnings(record=True) as w:
                simplefilter("always")
                np.random.seed(self.sample_seed)
                decision_id = decision_tracker.track(
                    variant=variant,
                    variants=variants,
                    givens=None, model_name=self.dummy_model_name,
                    variants_ranked_and_track_runners_up=
                    decision_tracker._should_track_runners_up(len(variants)))
                assert len(w) == 0

            assert is_valid_ksuid(decision_id)

        decision_id = decision_id_container['decision_id']
        reward = 1

        expected_add_reward_body = {
            decision_tracker.TYPE_KEY: decision_tracker.REWARD_TYPE,
            decision_tracker.MODEL_KEY: self.dummy_model_name,
            decision_tracker.REWARD_KEY: reward,
            decision_tracker.DECISION_ID_KEY: decision_id,
        }

        expected_request_json = json.dumps(expected_add_reward_body, sort_keys=False)

        def custom_matcher(request):
            request_dict = deepcopy(request.json())
            del request_dict[decision_tracker.MESSAGE_ID_KEY]

            if json.dumps(request_dict, sort_keys=False) != \
                    expected_request_json:

                print('raw request body:')
                print(request.text)
                print('compared request string')
                print(json.dumps(request_dict, sort_keys=False))
                print('expected body:')
                print(expected_request_json)
                return None
            return True

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success',
                   additional_matcher=custom_matcher)

            with catch_warnings(record=True) as w:
                simplefilter("always")
                decision_tracker.add_reward(
                    reward=reward, model_name=self.dummy_model_name, decision_id=decision_id)
                assert len(w) == 0

    def test_add_reward_bad_reward_type(self):
        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        decision_id_container = {'decision_id': None}

        def custom_matcher_caching_decision_id(request):
            request_dict = deepcopy(request.json())
            decision_id_container['decision_id'] = \
                request_dict[decision_tracker.MESSAGE_ID_KEY]
            return True

        variants = [el for el in range(0, 1)]
        variant = variants[0]

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success',
                   additional_matcher=custom_matcher_caching_decision_id)

            with catch_warnings(record=True) as w:
                simplefilter("always")
                np.random.seed(self.sample_seed)
                decision_id = decision_tracker.track(
                    variant=variant,
                    variants=variants,
                    givens=None, model_name=self.dummy_model_name,
                    variants_ranked_and_track_runners_up=
                    decision_tracker._should_track_runners_up(len(variants)))
                assert len(w) == 0

            assert is_valid_ksuid(decision_id)

        decision_id = decision_id_container['decision_id']
        reward = 'bad_reward'

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success')

            with raises(AssertionError) as aerr:
                decision_tracker.add_reward(
                    reward=reward, model_name=self.dummy_model_name, decision_id=decision_id)

    def test_add_reward_inf(self):
        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        decision_id_container = {'decision_id': None}

        def custom_matcher_caching_decision_id(request):
            request_dict = deepcopy(request.json())
            decision_id_container['decision_id'] = \
                request_dict[decision_tracker.MESSAGE_ID_KEY]
            return True

        variants = [el for el in range(0, 1)]
        variant = variants[0]

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success',
                   additional_matcher=custom_matcher_caching_decision_id)

            with catch_warnings(record=True) as w:
                simplefilter("always")
                np.random.seed(self.sample_seed)
                decision_id = decision_tracker.track(
                    variant=variant,
                    variants=variants,
                    givens=None, model_name=self.dummy_model_name,
                    variants_ranked_and_track_runners_up=
                    decision_tracker._should_track_runners_up(len(variants)))
                assert len(w) == 0

            assert is_valid_ksuid(decision_id)

        decision_id = decision_id_container['decision_id']

        reward = math.inf

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success')

            with raises(AssertionError) as aerr:
                decision_tracker.add_reward(
                    reward=reward, model_name=self.dummy_model_name, decision_id=decision_id)

        reward = -math.inf

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success')

            with raises(AssertionError) as aerr:
                decision_tracker.add_reward(
                    reward=reward, model_name=self.dummy_model_name, decision_id=decision_id)

    def test_add_reward_none(self):
        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up

        decision_id_container = {'decision_id': None}

        def custom_matcher_caching_decision_id(request):
            request_dict = deepcopy(request.json())
            decision_id_container['decision_id'] = \
                request_dict[decision_tracker.MESSAGE_ID_KEY]
            return True

        variants = [el for el in range(0, 1)]
        variant = variants[0]

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success',
                   additional_matcher=custom_matcher_caching_decision_id)

            with catch_warnings(record=True) as w:
                simplefilter("always")
                np.random.seed(self.sample_seed)
                decision_id = decision_tracker.track(
                    variant=variant,
                    variants=variants,
                    givens=None, model_name=self.dummy_model_name,
                    variants_ranked_and_track_runners_up=
                    decision_tracker._should_track_runners_up(len(variants)))
                assert len(w) == 0

            assert is_valid_ksuid(decision_id)

        decision_id = decision_id_container['decision_id']

        reward = None

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success')

            with raises(AssertionError) as aerr:
                decision_tracker.add_reward(
                    reward=reward, model_name=self.dummy_model_name, decision_id=decision_id)

        reward = np.nan

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success')

            with raises(AssertionError) as aerr:
                decision_tracker.add_reward(
                    reward=reward, model_name=self.dummy_model_name, decision_id=decision_id)
                assert aerr

    def test_add_reward_raises_for_none_model_name(self):
        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up
        with raises(AssertionError) as aerr:
            decision_tracker.add_reward(reward=1.0, model_name=None, decision_id=str(Ksuid()))

    def test_add_reward_raises_for_invalid_model_name(self):
        decision_tracker = dtr.DecisionTracker(track_url=self.track_url)
        decision_tracker.max_runners_up = self.max_runners_up
        with raises(AssertionError) as aerr:
            model_name = ''
            decision_tracker.add_reward(reward=1.0, model_name=model_name, decision_id=str(Ksuid()))

        with raises(AssertionError) as aerr:
            model_name = '!@#$%^&*()'
            decision_tracker.add_reward(reward=1.0, model_name=model_name, decision_id=str(Ksuid()))

        with raises(AssertionError) as aerr:
            model_name = ''.join(['a' for _ in range(65)])
            decision_tracker.add_reward(reward=1.0, model_name=model_name, decision_id=str(Ksuid()))

    def test_tracker_with_api_headers(self):
        # self, track_url: str, max_runners_up: int = 50, track_api_key: str = None
        dummy_api_key = 'dummy-api-key'
        decision_tracker = dtr.DecisionTracker(track_url=self.track_url, track_api_key=dummy_api_key)
        decision_tracker.max_runners_up = self.max_runners_up

        headers_cache = {'headers': None}

        def cache_headers(request):
            headers_cache['headers'] = request._request.headers
            return True

        mockup_body = {"k1": 1, "k2": 2}
        expected_headers = {
            'Content-Type': 'application/json',
            decision_tracker.API_KEY_HEADER: dummy_api_key}

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success', additional_matcher=cache_headers)

            decision_id = decision_tracker.post_improve_request(
                body_values=mockup_body,
                block=
                lambda result, error: (
                    warn("Improve.track error: {}".format(error))
                    if error else 0, 0))

            print('### decision_id ###')
            print(decision_id)
            assert is_valid_ksuid(decision_id)

            for k, v in expected_headers.items():
                assert k in headers_cache['headers']
                assert v == headers_cache['headers'][k]
