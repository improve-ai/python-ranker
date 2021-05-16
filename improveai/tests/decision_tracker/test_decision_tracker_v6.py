from copy import deepcopy
import json
import numpy as np
import requests_mock as rqm
import os
from pytest import fixture, raises
import sys

sys.path.append(
    os.sep.join(str(os.path.abspath(__file__)).split(os.sep)[:-3]))

import improveai.tracker as dtr


class TestDecisionTracker:

    @property
    def tracks_seed(self) -> str:
        return self._tracks_seed

    @tracks_seed.setter
    def tracks_seed(self, value: str):
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

        self.ranked_variants = range(100)
        self.max_runners_up = \
            int(os.getenv('V6_DECISION_TRACKER_MAX_RUNNERS_UP'))

        self.sample_seed = int(os.getenv('V6_DECISION_TRACKER_SAMPLE_SEED'))

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

        decision_tracker.should_track_runners_up(variants_count=1)

        assert decision_tracker.track_runners_up is False

    def test_should_track_runners_up_0_max_runners_up(self):

        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, max_runners_up=0,
                history_id=self.dummy_history_id)

        np.random.seed(self.tracks_seed)
        decision_tracker.should_track_runners_up(variants_count=100)

        assert decision_tracker.track_runners_up is False

    def test_should_track_runners_up_true(self):
        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, history_id=self.dummy_history_id)

        np.random.seed(self.tracks_seed)
        decision_tracker.should_track_runners_up(
            variants_count=self.variants_count)

        assert decision_tracker.track_runners_up is True

    def test_should_track_runners_up_false(self):
        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, history_id=self.dummy_history_id)

        np.random.seed(self.not_tracks_seed)
        decision_tracker.should_track_runners_up(
            variants_count=self.variants_count)

        assert decision_tracker.track_runners_up is False

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

        assert len(top_runners_up) == 0
        np.testing.assert_array_equal(top_runners_up, [])

    def test_top_runners_up_single_variant(self):

        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        ranked_variants = self.ranked_variants[:1]

        top_runners_up = \
            decision_tracker.top_runners_up(ranked_variants=ranked_variants)

        assert len(top_runners_up) == 0
        np.testing.assert_array_equal(top_runners_up, [])

    def test_top_runners_up_none_variant(self):

        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        ranked_variants = [None]

        top_runners_up = \
            decision_tracker.top_runners_up(ranked_variants=ranked_variants)

        assert len(top_runners_up) == 0
        np.testing.assert_array_equal(top_runners_up, [])

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
            top_runners_up, self.ranked_variants[1:len(self.ranked_variants)])

    def test_get_sample_single_variant(self):

        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        variant = self.ranked_variants[0]
        ranked_variants = self.ranked_variants[:1]
        track_runners_up = True

        sample = decision_tracker.get_sample(
            variant=variant, ranked_variants=ranked_variants,
            track_runners_up=track_runners_up)

        assert sample is None

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
        sample = decision_tracker.get_sample(
            variant=variant, ranked_variants=self.ranked_variants,
            track_runners_up=track_runners_up)

        assert sample == expected_sample

    def test_get_sample_no_variants_not_tracks(self):
        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        ranked_variants = [None]
        variant = ranked_variants[0]
        track_runners_up = False

        np.random.seed(self.sample_seed)
        sample = decision_tracker.get_sample(
            variant=variant, ranked_variants=ranked_variants,
            track_runners_up=track_runners_up)

        assert sample is None

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
        sample = decision_tracker.get_sample(
            variant=variant, ranked_variants=self.ranked_variants,
            track_runners_up=track_runners_up)

        assert sample == expected_sample

    def test_get_sample_no_variants_track(self):
        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        ranked_variants = [None]
        variant = ranked_variants[0]
        track_runners_up = True

        np.random.seed(self.sample_seed)
        sample = decision_tracker.get_sample(
            variant=variant, ranked_variants=ranked_variants,
            track_runners_up=track_runners_up)

        assert sample is None

    def test_get_sample_tracks_no_sample(self):

        # make sure there are no variants to sample from
        max_runners_up = len(self.ranked_variants) - 1

        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url, max_runners_up=max_runners_up,
                history_id=self.dummy_history_id)

        variant = self.ranked_variants[0]
        track_runners_up = True

        sample = decision_tracker.get_sample(
            variant=variant, ranked_variants=self.ranked_variants,
            track_runners_up=track_runners_up)

        assert sample is None

    def test_track_single_none_variant_none_given_no_runners_up(self):

        decision_tracker = \
            dtr.DecisionTracker(
                track_url=self.track_url,  max_runners_up=self.max_runners_up,
                history_id=self.dummy_history_id)

        expected_track_body = {
            decision_tracker.TIMESTAMP_KEY: self.dummy_timestamp,
            decision_tracker.HISTORY_ID_KEY: self.dummy_history_id,
            # decision_tracker.MESSAGE_ID_KEY: self.dummy_message_id,
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
                message_id=self.dummy_message_id,
                history_id=self.dummy_history_id,
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
            # decision_tracker.MESSAGE_ID_KEY: self.dummy_message_id,
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
                message_id=self.dummy_message_id,
                history_id=self.dummy_history_id,
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
            # decision_tracker.MESSAGE_ID_KEY: self.dummy_message_id,
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
                message_id=self.dummy_message_id,
                history_id=self.dummy_history_id,
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
            # decision_tracker.MESSAGE_ID_KEY: None,
            decision_tracker.TYPE_KEY: decision_tracker.DECISION_TYPE,
            decision_tracker.MODEL_KEY: self.dummy_model_name,
            decision_tracker.VARIANT_KEY: variants[0],
            decision_tracker.VARIANTS_COUNT_KEY: len(variants),
            decision_tracker.RUNNERS_UP_KEY: top_runners_up,
            decision_tracker.SAMPLE_KEY: expected_sample,
            decision_tracker.GIVEN_KEY: givens}

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
                message_id=self.dummy_message_id,
                history_id=self.dummy_history_id,
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
            # decision_tracker.MESSAGE_ID_KEY: self.dummy_message_id,
            decision_tracker.TYPE_KEY: decision_tracker.DECISION_TYPE,
            decision_tracker.MODEL_KEY: self.dummy_model_name,
            decision_tracker.VARIANT_KEY: variants[0],
            decision_tracker.VARIANTS_COUNT_KEY: len(variants),
            decision_tracker.RUNNERS_UP_KEY: top_runners_up,
            decision_tracker.SAMPLE_KEY: expected_sample,
            decision_tracker.GIVEN_KEY: givens}

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
                message_id=self.dummy_message_id,
                history_id=self.dummy_history_id,
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
            # decision_tracker.MESSAGE_ID_KEY: self.dummy_message_id,
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
                message_id=self.dummy_message_id,
                history_id=self.dummy_history_id,
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
            # decision_tracker.MESSAGE_ID_KEY: self.dummy_message_id,
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
                event_name=event_name,
                properties=dummy_properties,
                message_id=self.dummy_message_id,
                history_id=self.dummy_history_id,
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
                event_name=event_name,
                properties=dummy_properties,
                message_id=self.dummy_message_id,
                history_id=self.dummy_history_id,
                timestamp=self.dummy_timestamp)

            assert str(terr.value)

