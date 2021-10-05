from copy import deepcopy
import json
import numpy as np
import requests_mock as rqm
import os
from pytest import fixture, raises
import sys

sys.path.append(
    os.sep.join(str(os.path.abspath(__file__)).split(os.sep)[:-3]))

import improveai.decision_tracker as dtr


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
    def ranked_variants(self):
        return self._ranked_variants

    @ranked_variants.setter
    def ranked_variants(self, value):
        self._ranked_variants = value
        
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
            int(os.getenv('V6_DECISION_TRACKER_TRACKS_SEED'))

        self.not_tracks_seed = \
            int(os.getenv('V6_DECISION_TRACKER_NOT_TRACKS_SEED'))

        self.track_url = os.getenv('V6_DECISION_TRACKER_TEST_URL')
        self.api_key = os.getenv('V6_DECISION_TRACKER_TEST_API_KEY')

        self.variants_count = \
            int(os.getenv('V6_DECISION_TRACKER_SHOULD_TRACK_RUNNERS_UP_VARIANTS_COUNT'))

        self.ranked_variants = list(range(100))
        self.max_runners_up = \
            int(os.getenv('V6_DECISION_TRACKER_MAX_RUNNERS_UP'))

        self.sample_seed = int(os.getenv('V6_DECISION_TRACKER_SAMPLE_SEED'))
        self.first_sample_identical_with_variant_seed = \
            int(os.getenv('V6_DECISION_TRACKER_FIRST_SAMPLE_IDENTICAL_WITH_VARIANT_SEED'))

        self.dummy_variant = {'dummy0': 'variant'}
        self.dummy_ranked_variants = \
            np.array(
                [self.dummy_variant] +
                [{'dummy{}'.format(el): 'variant'}
                 for el in range(self.max_runners_up + 10)])

        self.dummy_model_name = 'dummy-model'
        self.dummy_message_id = 'dummy_message'
        self.dummy_history_id = 'dummy-history'
        self.dummy_timestamp = '2021-05-11T02:32:27.007Z'

    def test_should_track_runners_up_single_variant(self):

        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, history_id=self.dummy_history_id)

        assert decision_tracker.should_track_runners_up(variants_count=1) is False

    def test_should_track_runners_up_0_max_runners_up(self):

        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, max_runners_up=0,
                history_id=self.dummy_history_id)

        np.random.seed(self.tracks_seed)
        assert decision_tracker.should_track_runners_up(variants_count=100) is False

    def test_should_track_runners_up_true(self):
        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, history_id=self.dummy_history_id)

        np.random.seed(self.tracks_seed)
        assert decision_tracker.should_track_runners_up(variants_count=self.variants_count) is True

    def test_should_track_runners_up_false(self):
        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, history_id=self.dummy_history_id)

        np.random.seed(self.not_tracks_seed)
        assert decision_tracker.should_track_runners_up(variants_count=self.variants_count) is False

    def test_should_track_runners_up_2_variants(self):
        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, history_id=self.dummy_history_id)

        np.random.seed(self.not_tracks_seed)
        assert decision_tracker.should_track_runners_up(variants_count=2) is True

    def test_top_runners_up(self):

        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        top_runners_up = \
            decision_tracker.top_runners_up(
                ranked_variants=self.ranked_variants)

        assert len(top_runners_up) == self.max_runners_up
        np.testing.assert_array_equal(
            top_runners_up, self.ranked_variants[1:self.max_runners_up + 1])

    def test_top_runners_up_empty_variants(self):

        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        top_runners_up = \
            decision_tracker.top_runners_up(ranked_variants=[])

        assert top_runners_up is None

    def test_top_runners_up_single_variant(self):

        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        ranked_variants = self.ranked_variants[:1]

        top_runners_up = \
            decision_tracker.top_runners_up(ranked_variants=ranked_variants)

        assert top_runners_up is None

    def test_top_runners_up_none_variant(self):

        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        ranked_variants = [None]

        top_runners_up = \
            decision_tracker.top_runners_up(ranked_variants=ranked_variants)

        assert top_runners_up is None

    def test_top_runners_up_less_variants_than_runners_up(self):

        max_runners_up = len(self.ranked_variants) + 10

        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, max_runners_up=max_runners_up,
                history_id=self.dummy_history_id)

        top_runners_up = \
            decision_tracker.top_runners_up(
                ranked_variants=self.ranked_variants)

        assert len(top_runners_up) == len(self.ranked_variants) - 1
        np.testing.assert_array_equal(
            top_runners_up, self.ranked_variants[1:])

    def test_get_sample_single_variant_not_tracks(self):

        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        variant = self.ranked_variants[0]
        ranked_variants = self.ranked_variants[:1]
        track_runners_up = False

        sample, has_sample = decision_tracker.get_sample(
            variant=variant, ranked_variants=ranked_variants,
            track_runners_up=track_runners_up)

        assert sample is None
        assert has_sample is False

    def test_get_sample_single_variant_tracks(self):

        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        variant = self.ranked_variants[0]
        ranked_variants = self.ranked_variants[:1]
        track_runners_up = True

        sample, has_sample = decision_tracker.get_sample(
            variant=variant, ranked_variants=ranked_variants,
            track_runners_up=track_runners_up)

        assert sample is None
        assert has_sample is False

    def test_get_sample_single_variant_raises(self):
        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        variant = self.ranked_variants[1]
        ranked_variants = self.ranked_variants[:1]
        track_runners_up = True

        with raises(NotImplementedError) as nee:
            decision_tracker.get_sample(
                variant=variant, ranked_variants=ranked_variants,
                track_runners_up=track_runners_up)

    def test_get_sample_raises_for_non_bool_track_runners_up_01(self):
        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        variant = self.ranked_variants[0]
        track_runners_up = None

        with raises(AssertionError) as ae:
            decision_tracker.get_sample(
                variant=variant, ranked_variants=self.ranked_variants,
                track_runners_up=track_runners_up)

    def test_get_sample_raises_for_non_bool_track_runners_up_02(self):
        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        variant = self.ranked_variants[0]
        ranked_variants = self.ranked_variants[:10]
        track_runners_up = None

        with raises(AssertionError) as ae:
            decision_tracker.get_sample(
                variant=variant, ranked_variants=ranked_variants,
                track_runners_up=track_runners_up)

    def test_get_sample_not_tracks(self):

        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        variant = self.ranked_variants[0]
        track_runners_up = False

        expected_sample = \
            int(os.getenv('V6_DECISION_TRACKER_SAMPLE_NOT_TRACKS'))

        np.random.seed(self.sample_seed)
        sample, has_sample = decision_tracker.get_sample(
            variant=variant, ranked_variants=self.ranked_variants,
            track_runners_up=track_runners_up)

        assert sample == expected_sample
        assert has_sample is True

    def test_get_sample_no_variants_not_tracks_3_variants(self):
        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        ranked_variants = self.ranked_variants[:3]
        variant = ranked_variants[0]
        track_runners_up = False

        sample, has_sample = decision_tracker.get_sample(
            variant=variant, ranked_variants=ranked_variants,
            track_runners_up=track_runners_up)

        assert sample == ranked_variants[1]
        assert sample != variant
        assert has_sample is True

    def test_get_sample_no_variants_not_tracks_3_identical_variants(self):
        # theoretically for 2 variants we always have runners up
        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        ranked_variants = ['a', 'a', 'a']
        variant = ranked_variants[0]
        track_runners_up = False

        sample, has_sample = decision_tracker.get_sample(
            variant=variant, ranked_variants=ranked_variants,
            track_runners_up=track_runners_up)

        assert sample is ranked_variants[1]
        assert sample == variant
        assert has_sample is True

    def test_get_sample_no_variants_not_tracks_identical_variants_01(self):
        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        ranked_variants = ['z', 'x', 'a', 'a', 'b', 'c']
        variant = ranked_variants[2]
        track_runners_up = False

        np.random.seed(self.sample_seed)
        sample, has_sample = decision_tracker.get_sample(
            variant=variant, ranked_variants=ranked_variants,
            track_runners_up=track_runners_up)

        assert sample == ranked_variants[0]
        assert sample is not None
        assert has_sample is True

    def test_get_sample_raises_on_track_runners_up_and_not_ranked_variants(self):
        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        ranked_variants = ['z', 'x', 'a', 'a', 'b', 'c']
        variant = ranked_variants[2]
        track_runners_up = True

        with raises(AssertionError) as aerr:
            decision_tracker.get_sample(
                variant=variant, ranked_variants=ranked_variants,
                track_runners_up=track_runners_up)

    def test_get_sample_no_variants_not_tracks_identical_variants_02(self):
        # make sure to sample identical index with selected variant's index
        # first
        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        ranked_variants = ['z', 'x', 'a', 'a', 'b', 'c']
        variant = ranked_variants[2]
        track_runners_up = False

        np.random.seed(self.first_sample_identical_with_variant_seed)
        sample, has_sample = decision_tracker.get_sample(
            variant=variant, ranked_variants=ranked_variants,
            track_runners_up=track_runners_up)

        assert sample == ranked_variants[0]
        assert sample is not None
        assert has_sample is True

    def test_get_sample_3_none_variants_not_tracks(self):
        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        ranked_variants = [None, None, None]
        variant = ranked_variants[0]
        track_runners_up = False

        np.random.seed(self.sample_seed)
        sample, has_sample = decision_tracker.get_sample(
            variant=variant, ranked_variants=ranked_variants,
            track_runners_up=track_runners_up)

        assert sample is None
        assert has_sample is True

    def test_get_sample_raises_for_wrong_variants_type(self):
        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        ranked_variants = {'variants': [None, None, None]}
        variant = ranked_variants['variants'][0]
        track_runners_up = False

        with raises(TypeError) as terr:
            decision_tracker.get_sample(
                variant=variant, ranked_variants=ranked_variants,
                track_runners_up=track_runners_up)

    def test_get_sample_raises_for_zero_len_variants(self):
        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        ranked_variants = []
        variant = self.ranked_variants[0]
        track_runners_up = False

        with raises(ValueError) as verr:
            decision_tracker.get_sample(
                variant=variant, ranked_variants=ranked_variants,
                track_runners_up=track_runners_up)

    def test_get_sample_no_variants_not_tracks(self):
        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        ranked_variants = [None]
        variant = ranked_variants[0]
        track_runners_up = False

        np.random.seed(self.sample_seed)
        sample, has_sample = decision_tracker.get_sample(
            variant=variant, ranked_variants=ranked_variants,
            track_runners_up=track_runners_up)

        assert sample is None
        assert has_sample is False

    def test_get_sample_no_variants_tracks(self):
        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        ranked_variants = [None]
        variant = ranked_variants[0]
        track_runners_up = True

        np.random.seed(self.sample_seed)
        sample, has_sample = decision_tracker.get_sample(
            variant=variant, ranked_variants=ranked_variants,
            track_runners_up=track_runners_up)

        assert sample is None
        assert has_sample is False

    def test_get_sample_tracks(self):
        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        variant = self.ranked_variants[0]
        track_runners_up = True

        expected_sample = \
            int(os.getenv('V6_DECISION_TRACKER_SAMPLE_TRACKS'))

        np.random.seed(self.sample_seed)
        sample, has_sample = decision_tracker.get_sample(
            variant=variant, ranked_variants=self.ranked_variants,
            track_runners_up=track_runners_up)

        assert sample == expected_sample
        assert has_sample is True

    def test_get_sample_no_variants_track(self):
        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        ranked_variants = [None]
        variant = ranked_variants[0]
        track_runners_up = True

        np.random.seed(self.sample_seed)
        sample, has_sample = decision_tracker.get_sample(
            variant=variant, ranked_variants=ranked_variants,
            track_runners_up=track_runners_up)

        assert sample is None
        assert has_sample is False

    def test_get_sample_tracks_no_sample_01(self):

        # make sure there are no variants to sample from
        max_runners_up = len(self.ranked_variants) - 1

        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, max_runners_up=max_runners_up,
                history_id=self.dummy_history_id)

        variant = self.ranked_variants[0]
        track_runners_up = True

        sample, has_sample = decision_tracker.get_sample(
            variant=variant, ranked_variants=self.ranked_variants,
            track_runners_up=track_runners_up)

        assert sample is None
        assert has_sample is False

    def test_get_sample_tracks_no_sample_02(self):

        # make sure there are no variants to sample from
        max_runners_up = len(self.ranked_variants) - 1

        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, max_runners_up=max_runners_up,
                history_id=self.dummy_history_id)

        variant = self.ranked_variants[0]
        ranked_variants = self.ranked_variants[:40]
        track_runners_up = True

        sample, has_sample = decision_tracker.get_sample(
            variant=variant, ranked_variants=ranked_variants,
            track_runners_up=track_runners_up)

        assert sample is None
        assert has_sample is False

    def test_get_sample_tracks_sample_is_last_variant(self):

        # make sure there are no variants to sample from
        max_runners_up = len(self.ranked_variants) - 1

        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, max_runners_up=max_runners_up,
                history_id=self.dummy_history_id)

        variant = self.ranked_variants[0]
        ranked_variants = self.ranked_variants + [100]
        track_runners_up = True

        sample, has_sample = decision_tracker.get_sample(
            variant=variant, ranked_variants=ranked_variants,
            track_runners_up=track_runners_up)

        assert sample == ranked_variants[-1]
        assert has_sample is True

    def test_track_single_none_variant_none_given_no_runners_up(self):

        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url,  max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        expected_track_body = {
            decision_tracker.TIMESTAMP_KEY: self.dummy_timestamp,
            decision_tracker.HISTORY_ID_KEY: self.dummy_history_id,
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

            resp = decision_tracker.track(
                variant=None,
                variants=[None],
                givens=None, model_name=self.dummy_model_name,
                variants_ranked_and_track_runners_up=False,
                timestamp=self.dummy_timestamp)

            if resp is None:
                print('The input request body and expected request body mismatch')
            assert resp is not None
            assert resp.status_code == 200
            assert resp.text == 'success'

    def test_track_none_given_no_runners_up(self):

        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url,  max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        variants = [1, 2, 3]
        variant = variants[0]
        expected_sample = 2

        expected_track_body = {
            decision_tracker.TIMESTAMP_KEY: self.dummy_timestamp,
            decision_tracker.HISTORY_ID_KEY: self.dummy_history_id,
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

            np.random.seed(self.sample_seed)
            resp = decision_tracker.track(
                variant=variant,
                variants=variants,
                givens=None, model_name=self.dummy_model_name,
                variants_ranked_and_track_runners_up=False,
                timestamp=self.dummy_timestamp)

            if resp is None:
                print('The input request body and expected request body mismatch')
            assert resp is not None
            assert resp.status_code == 200
            assert resp.text == 'success'

    def test_track_none_given(self):

        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url,  max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        variants = [el for el in range(1, 20, 1)]
        top_runners_up = variants[1:self.max_runners_up + 1]
        variant = variants[0]
        expected_sample = 12

        expected_track_body = {
            decision_tracker.TIMESTAMP_KEY: self.dummy_timestamp,
            decision_tracker.HISTORY_ID_KEY: self.dummy_history_id,
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

            np.random.seed(self.sample_seed)
            resp = decision_tracker.track(
                variant=variant,
                variants=variants,
                givens=None, model_name=self.dummy_model_name,
                variants_ranked_and_track_runners_up=True,
                timestamp=self.dummy_timestamp)

            if resp is None:
                print('The input request body and expected request body mismatch')
            assert resp is not None
            assert resp.status_code == 200
            assert resp.text == 'success'

    def test_track_2_variants_01(self):

        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url,  max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        variants = [0, 1]
        top_runners_up = [variants[1]]
        variant = variants[0]

        expected_track_body = {
            decision_tracker.TIMESTAMP_KEY: self.dummy_timestamp,
            decision_tracker.HISTORY_ID_KEY: self.dummy_history_id,
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

            np.random.seed(self.sample_seed)
            resp = decision_tracker.track(
                variant=variant,
                variants=variants,
                givens=None, model_name=self.dummy_model_name,
                variants_ranked_and_track_runners_up=True,
                timestamp=self.dummy_timestamp)

            if resp is None:
                print('The input request body and expected request body mismatch')
            assert resp is not None
            assert resp.status_code == 200
            assert resp.text == 'success'

    def test_track_2_variants_02(self):

        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url,  max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        variants = [0, 1]
        top_runners_up = [variants[0]]
        variant = variants[1]

        expected_track_body = {
            decision_tracker.TIMESTAMP_KEY: self.dummy_timestamp,
            decision_tracker.HISTORY_ID_KEY: self.dummy_history_id,
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

            np.random.seed(self.sample_seed)
            resp = decision_tracker.track(
                variant=variant,
                variants=variants,
                givens=None, model_name=self.dummy_model_name,
                variants_ranked_and_track_runners_up=True,
                timestamp=self.dummy_timestamp)

            if resp is None:
                print('The input request body and expected request body mismatch')
            assert resp is not None
            assert resp.status_code == 200
            assert resp.text == 'success'

    def test_track_2_variants_03(self):

        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url,  max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        variants = [0, 1]
        top_runners_up = [variants[1]]
        variant = variants[0]

        expected_track_body = {
            decision_tracker.TIMESTAMP_KEY: self.dummy_timestamp,
            decision_tracker.HISTORY_ID_KEY: self.dummy_history_id,
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

            np.random.seed(self.sample_seed)
            resp = decision_tracker.track(
                variant=variant,
                variants=variants,
                givens=None, model_name=self.dummy_model_name,
                variants_ranked_and_track_runners_up=False,
                timestamp=self.dummy_timestamp)

            if resp is None:
                print('The input request body and expected request body mismatch')
            assert resp is not None
            assert resp.status_code == 200
            assert resp.text == 'success'

    def test_track_2_variants_04(self):

        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url,  max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        variants = [0, 1]
        top_runners_up = [variants[0]]
        variant = variants[1]

        expected_track_body = {
            decision_tracker.TIMESTAMP_KEY: self.dummy_timestamp,
            decision_tracker.HISTORY_ID_KEY: self.dummy_history_id,
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

            np.random.seed(self.sample_seed)
            resp = decision_tracker.track(
                variant=variant,
                variants=variants,
                givens=None, model_name=self.dummy_model_name,
                variants_ranked_and_track_runners_up=False,
                timestamp=self.dummy_timestamp)

            if resp is None:
                print('The input request body and expected request body mismatch')
            assert resp is not None
            assert resp.status_code == 200
            assert resp.text == 'success'

    def test_track(self):

        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url,  max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        variants = [el for el in range(1, 20, 1)]
        top_runners_up = variants[1:self.max_runners_up + 1]
        variant = variants[0]
        givens = {'dummy': 'givens'}
        expected_sample = 12

        expected_track_body = {
            decision_tracker.TIMESTAMP_KEY: self.dummy_timestamp,
            decision_tracker.HISTORY_ID_KEY: self.dummy_history_id,
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

            np.random.seed(self.sample_seed)
            resp = decision_tracker.track(
                variant=variant,
                variants=variants,
                givens=givens, model_name=self.dummy_model_name,
                variants_ranked_and_track_runners_up=True,
                timestamp=self.dummy_timestamp)

            if resp is None:
                print('The input request body and expected request body mismatch')
            assert resp is not None
            assert resp.status_code == 200
            assert resp.text == 'success'

    def test_track_ndarray_variants(self):

        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url,  max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        variants = [el for el in range(1, 20, 1)]
        top_runners_up = variants[1:self.max_runners_up + 1]
        variant = variants[0]
        givens = {'dummy': 'givens'}
        expected_sample = 12

        expected_track_body = {
            decision_tracker.TIMESTAMP_KEY: self.dummy_timestamp,
            decision_tracker.HISTORY_ID_KEY: self.dummy_history_id,
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

            np.random.seed(self.sample_seed)
            resp = decision_tracker.track(
                variant=variant,
                variants=np.array(variants),
                givens=givens, model_name=self.dummy_model_name,
                variants_ranked_and_track_runners_up=True,
                timestamp=self.dummy_timestamp)

            if resp is None:
                print('The input request body and expected request body mismatch')
            assert resp is not None
            assert resp.status_code == 200
            assert resp.text == 'success'

    def test_track_event_no_properties(self):
        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        event_name = 'dummy-event'

        expected_track_body = {
            decision_tracker.TIMESTAMP_KEY: self.dummy_timestamp,
            decision_tracker.HISTORY_ID_KEY: self.dummy_history_id,
            decision_tracker.TYPE_KEY: decision_tracker.EVENT_TYPE,
            decision_tracker.EVENT_KEY: event_name}

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

            np.random.seed(self.sample_seed)
            resp = decision_tracker.track_event(
                event_name=event_name,
                timestamp=self.dummy_timestamp)

            if resp is None:
                print(
                    'The input request body and expected request body mismatch')
            assert resp is not None
            assert resp.status_code == 200
            assert resp.text == 'success'

    def test_track_event(self):
        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        event_name = 'dummy-event'
        dummy_properties = {'dummy': {'nested': ['properties', 'test']}}

        expected_track_body = {
            decision_tracker.TIMESTAMP_KEY: self.dummy_timestamp,
            decision_tracker.HISTORY_ID_KEY: self.dummy_history_id,
            decision_tracker.TYPE_KEY: decision_tracker.EVENT_TYPE,
            decision_tracker.EVENT_KEY: event_name,
            decision_tracker.PROPERTIES_KEY: dummy_properties}

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

            np.random.seed(self.sample_seed)
            resp = decision_tracker.track_event(
                event_name=event_name, properties=dummy_properties,
                timestamp=self.dummy_timestamp)

            if resp is None:
                print(
                    'The input request body and expected request body mismatch')
            assert resp is not None
            assert resp.status_code == 200
            assert resp.text == 'success'

    def test_track_event_raises_type_error(self):
        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        event_name = 'dummy-event'
        dummy_properties = 'this is not a dict'

        with raises(TypeError) as terr:

            decision_tracker.track_event(
                event_name=event_name, properties=dummy_properties,
                timestamp=self.dummy_timestamp)

            assert str(terr.value)

    def test_track_2_variants_and_sample(self):

        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url,  max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        variants = [el for el in range(1, 3, 1)]
        top_runners_up = variants[1:]
        variant = variants[0]
        expected_sample = None

        expected_track_body = {
            decision_tracker.TIMESTAMP_KEY: self.dummy_timestamp,
            decision_tracker.HISTORY_ID_KEY: self.dummy_history_id,
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

            np.random.seed(self.sample_seed)
            resp = decision_tracker.track(
                variant=variant,
                variants=variants,
                givens=None, model_name=self.dummy_model_name,
                variants_ranked_and_track_runners_up=True,
                timestamp=self.dummy_timestamp)

            if resp is None:
                print('The input request body and expected request body mismatch')
            assert resp is not None
            assert resp.status_code == 200
            assert resp.text == 'success'

    def test_track_2_variants__max_runners_up_0(self):

        max_runners_up = 0

        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url,  max_runners_up=max_runners_up,
                history_id=self.dummy_history_id)

        variants = [el for el in range(1, 3, 1)]
        variant = variants[0]
        expected_sample = variants[1]

        expected_track_body = {
            decision_tracker.TIMESTAMP_KEY: self.dummy_timestamp,
            decision_tracker.HISTORY_ID_KEY: self.dummy_history_id,
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

            np.random.seed(self.sample_seed)
            resp = decision_tracker.track(
                variant=variant,
                variants=variants,
                givens=None, model_name=self.dummy_model_name,
                variants_ranked_and_track_runners_up=True,
                timestamp=self.dummy_timestamp)

            if resp is None:
                print('The input request body and expected request body mismatch')
            assert resp is not None
            assert resp.status_code == 200
            assert resp.text == 'success'

    # TODO determine if:
    #   - those tests are 100% correct
    #   - max_runners_up = 0 be overridden by count = 2 and runners up should be tracked
    def test_should_track_runners_up_2_variants(self):

        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, history_id=self.dummy_history_id)

        assert all(
            [decision_tracker.should_track_runners_up(variants_count=2)
             for _ in range(10)])

    def test_top_runners_up_2_variants(self):

        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, history_id=self.dummy_history_id)

        variants = [el for el in range(2)]

        top_runners_up = decision_tracker.top_runners_up(ranked_variants=variants)

        assert top_runners_up == [1]

    def test_track_2_variants_no_sample_max_runners_up_50(self):

        max_runners_up = 50

        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url,  max_runners_up=max_runners_up,
                history_id=self.dummy_history_id)

        variants = [el for el in range(0, 1)]
        variant = variants[0]

        expected_track_body = {
            decision_tracker.TIMESTAMP_KEY: self.dummy_timestamp,
            decision_tracker.HISTORY_ID_KEY: self.dummy_history_id,
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

            np.random.seed(self.sample_seed)
            resp = decision_tracker.track(
                variant=variant,
                variants=variants,
                givens=None, model_name=self.dummy_model_name,
                variants_ranked_and_track_runners_up=
                decision_tracker.should_track_runners_up(len(variants)),
                timestamp=self.dummy_timestamp)

            if resp is None:
                print('The input request body and expected request body mismatch')
            assert resp is not None
            assert resp.status_code == 200
            assert resp.text == 'success'

    # TODO test for max_runners_up = 0
