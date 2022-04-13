from copy import deepcopy
import json
import math
import requests_mock as rqm
import numpy as np
import os
from pytest import fixture, raises, warns
import sys
from unittest import TestCase
import warnings

sys.path.append(
    os.sep.join(str(os.path.abspath(__file__)).split(os.sep)[:-3]))

import improveai.decision as d
import improveai.decision_model as dm
from improveai.tests.test_utils import get_test_data


class TestDecision(TestCase):

    @property
    def decision_model_without_tracker(self) -> dm.DecisionModel:
        return self._decision_model

    @decision_model_without_tracker.setter
    def decision_model_without_tracker(self, value: dm.DecisionModel):
        self._decision_model = value

    @property
    def mockup_variants(self):
        return self._mockup_variants

    @mockup_variants.setter
    def mockup_variants(self, value):
        self._mockup_variants = value

    @property
    def mockup_givens(self):
        return self._mockup_given

    @mockup_givens.setter
    def mockup_givens(self, value):
        self._mockup_given = value
        
    @property
    def test_jsons_data_directory(self):
        return self._test_jsons_data_directory
    
    @test_jsons_data_directory.setter
    def test_jsons_data_directory(self, value):
        self._test_jsons_data_directory = value

    @property
    def decision_model_with_tracker(self) -> dm.DecisionModel:
        return self._decision_model_with_tracker

    @decision_model_with_tracker.setter
    def decision_model_with_tracker(self, value: dm.DecisionModel):
        self._decision_model_with_tracker = value

    @property
    def tracker(self):
        return self._tracker

    @tracker.setter
    def tracker(self, value):
        self._tracker = value

    @property
    def track_url(self):
        return self._track_url

    @track_url.setter
    def track_url(self, value):
        self._track_url = value

    @fixture(autouse=True)
    def prep_env(self):
        # create an instance of DecisionModel
        decision_tests_model_url = os.getenv('DUMMY_MODEL_PATH')

        self.test_jsons_data_directory = \
            os.getenv('DECISION_TEST_SUITE_JSONS_DIR')

        self.track_url = os.getenv('DECISION_TRACKER_TEST_URL')

        self.decision_model_without_tracker = \
            dm.DecisionModel(model_name=None)\
            .load(model_url=decision_tests_model_url)

        self.decision_model_with_tracker = \
            dm.DecisionModel(model_name=None, track_url=self.track_url)\
            .load(model_url=decision_tests_model_url)

        self.mockup_variants = [
            {'$value': 123},
            {"$value": {'nested': 'dict'}, }]

        self.mockup_givens = {
            'key1': [1, 2, 3],
            'key2': {'nested': 'dict'},
            'key3': 1,
            'key4': 1.0}

        self.dummy_variants = ['dummy_variant']
        self.dummy_givens = {'dummy': 'givens'}

        self.tracks_seed = int(os.getenv('DECISION_TRACKER_TRACKS_SEED'))
        self.not_tracks_seed = \
            int(os.getenv('DECISION_TRACKER_NOT_TRACKS_SEED'))

        self.dummy_history_id = 'dummy-history-id'
        self.dummy_timestamp = '2021-05-11T02:32:27.007Z'

    def test_raises_value_error_when_model_none(self):
        with raises(AssertionError) as aerr:
            d.Decision(decision_model=None)

    def test_raises_value_error_when_model_str(self):
        with raises(AssertionError) as aerr:
            d.Decision(decision_model='abc')

    def test_raises_value_error_when_model_number(self):
        with raises(AssertionError) as aerr:
            d.Decision(decision_model=123.13)

    def test_raises_value_error_when_model_bool(self):
        with raises(AssertionError) as aerr:
            d.Decision(decision_model=True)

    def _generic_outer_setter_raises_test(
            self, decision: d.Decision, set_attr_name: str,
            set_attr_value: object):

        print('### set_attr_value ###')
        print(set_attr_name)
        print(set_attr_value)

        with raises(AttributeError) as aerr:
            setattr(decision, set_attr_name, set_attr_value)
            assert str(aerr.value) == "AttributeError: can't set attribute"

    def test_model_setter(self):
        decision = d.Decision(decision_model=self.decision_model_without_tracker)
        assert decision.decision_model is not None

        assert decision.variants == [None]
        assert decision.givens is None
        assert decision.chosen is False
        assert decision.best is None

        self._generic_outer_setter_raises_test(
            decision=decision, set_attr_name='decision_model',
            set_attr_value='dummy value')

        assert decision.variants == [None]
        assert decision.givens is None
        assert decision.chosen is False
        assert decision.best is None

    def test_model_getter(self):
        decision = d.Decision(decision_model=self.decision_model_without_tracker)
        assert decision.decision_model is not None
        assert decision.decision_model == self.decision_model_without_tracker

    def test_choose_from_variants_setter(self):
        decision = d.Decision(decision_model=self.decision_model_without_tracker)
        decision.variants = self.mockup_variants

        assert decision.variants is not None
        np.testing.assert_array_equal(decision.variants, self.mockup_variants)

        assert decision.givens is None
        assert decision.chosen is False
        assert decision.best is None

        with raises(AssertionError) as aerr:
            setattr(decision, 'variants', 'dummy_value')
            assert str(aerr.value)

    def test_choose_from_variants_setter_raises_type_error_for_string(self):

        with raises(AssertionError) as aerr:
            decision_0 = d.Decision(decision_model=self.decision_model_without_tracker)
            decision_0.variants = 'dummy string'
            assert str(aerr.value)

    def test_choose_from_variants_setter_raises_type_error_for_non_iterable(
            self):
        with raises(AssertionError) as aerr:
            decision_1 = d.Decision(decision_model=self.decision_model_without_tracker)
            decision_1.variants = {'dummy': 'string'}

            assert str(aerr.value)

        with raises(AssertionError) as aerr:
            decision_2 = d.Decision(decision_model=self.decision_model_without_tracker)
            decision_2.variants = 1234
            assert str(aerr.value)

        with raises(AssertionError) as aerr:
            decision_3 = d.Decision(decision_model=self.decision_model_without_tracker)
            decision_3.variants = 1234.1234
            assert str(aerr.value)

    def test_givens_setter(self):
        decision = d.Decision(decision_model=self.decision_model_without_tracker)
        decision.givens = self.mockup_givens

        assert decision.givens is not None
        assert decision.givens == self.mockup_givens
        assert isinstance(decision.givens, dict)

        assert decision.variants == [None]
        assert decision.chosen is False
        assert decision.best is None

        with raises(AssertionError) as aerr:
            setattr(decision, 'givens', 'dummy_value')
            assert str(aerr.value) == "AttributeError: can't set attribute"

    def test_givens_setter_raises_type_error_for_non_dict(self):

        with raises(AssertionError) as terr:
            decision = d.Decision(decision_model=self.decision_model_without_tracker)
            decision.givens = 'dummy string'
            assert str(terr.value)

        with raises(AssertionError) as terr:
            decision = d.Decision(decision_model=self.decision_model_without_tracker)
            decision.givens = ['dummy', 'string']
            assert str(terr.value)

        with raises(AssertionError) as terr:
            decision = d.Decision(decision_model=self.decision_model_without_tracker)
            decision.givens = 1234
            assert str(terr.value)

        with raises(AssertionError) as terr:
            decision = d.Decision(decision_model=self.decision_model_without_tracker)
            decision.givens = 1234.1234
            assert str(terr.value)

    def test_get_01(self):

        test_case_filename = os.getenv('DECISION_TEST_GET_01_JSON')
        path_to_test_case_file = \
            os.sep.join([self.test_jsons_data_directory, test_case_filename])

        test_data = get_test_data(path_to_test_json=path_to_test_case_file, method='read')

        test_case = test_data.get("test_case", None)
        assert test_case is not None

        test_variants = test_case.get('variants', None)
        assert test_variants is not None

        test_given = test_case.get('givens', None)
        assert test_given is not None

        decision = d.Decision(decision_model=self.decision_model_with_tracker)

        assert decision.chosen is False

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success')
            decision.variants = test_variants
            decision.givens = test_given
            best_variant = decision.get()

        assert decision.chosen is True

        expected_output = test_data.get('test_output', None)
        assert expected_output is not None

        expected_best = expected_output.get('best', None)
        assert expected_best is not None

        assert best_variant == expected_best

    def test_get_02(self):

        test_case_filename = os.getenv('DECISION_TEST_GET_02_JSON')
        path_to_test_case_file = \
            os.sep.join([self.test_jsons_data_directory, test_case_filename])

        test_data = get_test_data(path_to_test_json=path_to_test_case_file, method='read')

        test_case = test_data.get("test_case", None)
        assert test_case is not None

        test_variants = test_case.get('variants', None)
        assert test_variants is not None

        test_givens = test_case.get('givens', None)
        assert test_givens is not None

        decision = d.Decision(decision_model=self.decision_model_with_tracker)

        assert decision.chosen is False

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success')
            decision.variants = test_variants
            decision.givens = test_givens
            best_variant = decision.get()

        assert decision.chosen is True

        expected_output = test_data.get('test_output', None)
        assert expected_output is not None

        expected_best = expected_output.get('best', None)
        assert expected_best is not None

        assert best_variant == expected_best

    def test_get_03(self):

        decision = d.Decision(decision_model=self.decision_model_with_tracker)

        assert decision.chosen is False

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success')
            decision.variants = [None]
            decision.givens = {}
            best_variant = decision.get()

        assert decision.chosen is True
        assert best_variant is None

    def test_get_04(self):

        decision = d.Decision(decision_model=self.decision_model_with_tracker)

        assert decision.chosen is False

        with raises(AssertionError) as aerr:
            with rqm.Mocker() as m:
                m.post(self.track_url, text='success')
                decision.variants = None
                decision.givens = {}
                decision.get()

    def test_get_05(self):

        decision = d.Decision(decision_model=self.decision_model_without_tracker)

        assert decision.chosen is False

        with raises(ValueError) as verr:
            decision.variants = [None]
            decision.givens = {}
            decision.get()

    def test_get_06(self):
        # this is a test case which covers tracking runners up from within
        # get() call

        variants = [el for el in range(100)]
        variants[9] = {
            "text": "lovely corgi",
            "chars": 12,
            "words": 2}

        runners_up_tracked = []

        def custom_matcher(request):

            request_dict = deepcopy(request.json())
            del request_dict[self.decision_model_with_tracker._tracker.MESSAGE_ID_KEY]

            if 'runners_up' in request_dict:
                runners_up_tracked.append(True)
            else:
                runners_up_tracked.append(False)
            return True

        for _ in variants:
            with rqm.Mocker() as m:
                m.post(self.track_url, text='success', additional_matcher=custom_matcher)

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")

                    # np.random.seed(self.tracks_seed)
                    decision = d.Decision(decision_model=self.decision_model_with_tracker)
                    assert decision.chosen is False
                    decision.variants = variants
                    decision.givens = {}
                    best_variant = decision.get()
                    print('Got following warnings count')
                    print(len(w))

        assert any(runners_up_tracked)
        assert decision.chosen is True
        assert best_variant == variants[9]

    def test_get_07(self):
        # this is a test case which covers NOT tracking runners up from within
        # get() call

        decision = d.Decision(decision_model=self.decision_model_with_tracker)

        assert decision.chosen is False

        variants = [el for el in range(20)]
        variants[9] = {
            "text": "lovely corgi",
            "chars": 12,
            "words": 2}

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success')

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                decision.variants = variants
                decision.givens = {}
                best_variant = decision.get()
                assert len(w) == 0

        assert decision.chosen is True
        assert best_variant == variants[9]

    def test_get_08(self):
        # this is a test case which covers tracking runners up from within
        # get() call

        decision_tracker = self.decision_model_with_tracker._tracker

        expected_track_body = {
            decision_tracker.TIMESTAMP_KEY: self.dummy_timestamp,
            decision_tracker.TYPE_KEY: decision_tracker.DECISION_TYPE,
            decision_tracker.MODEL_KEY: self.decision_model_with_tracker.model_name,
            decision_tracker.VARIANT_KEY: None,
            decision_tracker.VARIANTS_COUNT_KEY: 1,
        }

        expected_request_json = json.dumps(expected_track_body, sort_keys=False)

        request_validity = {
            'request_body_ok': True}

        def custom_matcher(request):

            request_dict = deepcopy(request.json())
            del request_dict[self.decision_model_with_tracker._tracker.MESSAGE_ID_KEY]

            if json.dumps(request_dict, sort_keys=False) != \
                    expected_request_json:
                request_validity['request_body_ok'] = True

            return True

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success', additional_matcher=custom_matcher)

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                decision = d.Decision(decision_model=self.decision_model_with_tracker)
                assert decision.chosen is False

                memoized_variant = decision.get()
                assert decision.chosen is True
                assert memoized_variant is None
                assert len(w) == 0
                assert request_validity['request_body_ok']

    def _get_complete_decision(self):
        test_case_filename = os.getenv('DECISION_TEST_GET_02_JSON')
        path_to_test_case_file = \
            os.sep.join([self.test_jsons_data_directory, test_case_filename])

        test_data = get_test_data(path_to_test_json=path_to_test_case_file, method='read')

        test_case = test_data.get("test_case", None)
        if test_case is None:
            raise ValueError('`test_case` can`t be empty')

        test_variants = test_case.get('variants', None)
        if test_variants is None:
            raise ValueError('`variants` can`t be empty')

        test_givens = test_case.get('givens', None)
        if test_givens is None:
            raise ValueError('`givens` can`t be empty')

        decision = d.Decision(decision_model=self.decision_model_with_tracker)

        assert decision.chosen is False

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success')
            decision.givens = test_givens
            decision.variants = test_variants
            best_variant = decision.get()

        assert decision.chosen is True
        return decision, best_variant

    def test_choose_from_already_chosen(self):
        decision, best_variant = self._get_complete_decision()

        assert decision.chosen is True

        set_variants = decision.variants

        decision.variants = self.dummy_variants

        assert decision.variants == set_variants
        assert best_variant == decision.best

    def test_choose_from_variants_already_set(self):
        decision = d.Decision(decision_model=self.decision_model_without_tracker)

        assert decision.variants == [None]
        assert decision.givens is None
        assert decision.chosen is False
        assert decision.best is None

        decision.variants = self.dummy_variants

        assert decision.variants == self.dummy_variants
        assert decision.givens is None
        assert decision.chosen is False
        assert decision.best is None

        decision.variants = self.dummy_variants + self.dummy_variants

        assert decision.variants == self.dummy_variants
        assert decision.givens is None
        assert decision.chosen is False
        assert decision.best is None

    def test_given_givens_already_set(self):
        decision = d.Decision(decision_model=self.decision_model_without_tracker)

        assert decision.variants == [None]
        assert decision.givens is None
        assert decision.chosen is False
        assert decision.best is None

        decision.givens = self.dummy_givens

        assert decision.variants == [None]
        assert decision.givens is self.dummy_givens
        assert decision.chosen is False
        assert decision.best is None

        appended_dummy_givens = deepcopy(self.dummy_givens)
        appended_dummy_givens['dummy1'] = 'givens'

        decision.givens = appended_dummy_givens

        assert decision.variants == [None]
        assert decision.givens is self.dummy_givens
        assert decision.chosen is False
        assert decision.best is None

    def test_given_already_chosen(self):
        decision, best_variant = self._get_complete_decision()

        assert decision.chosen is True

        existing_givens = decision.givens

        assert {'dummy_given': 'abc'} != existing_givens
        decision.givens = {'dummy_given': 'abc'}

        assert decision.givens == existing_givens
        assert best_variant == decision.best

    def test_get_already_chosen(self):
        decision, memoized_variant = self._get_complete_decision()

        assert decision.chosen is True

        memoized_variant_from_decision = decision.get()

        assert memoized_variant == decision.best == \
               memoized_variant_from_decision

    def test_get_with_zero_len_variants(self):

        test_case_filename = \
            os.getenv('DECISION_TEST_GET_ZERO_LENGTH_VARIANTS_JSON')
        path_to_test_case_file = \
            os.sep.join([self.test_jsons_data_directory, test_case_filename])

        test_data = get_test_data(path_to_test_json=path_to_test_case_file, method='read')

        test_case = test_data.get("test_case", None)
        if test_case is None:
            raise ValueError('`test_case` can`t be empty')

        test_variants = test_case.get('variants', None)
        if test_variants is None:
            raise ValueError('`variants` can`t be empty')

        test_given = test_case.get('givens', None)
        if test_given is None:
            raise ValueError('`givens` can`t be empty')

        decision = d.Decision(decision_model=self.decision_model_with_tracker)

        assert decision.chosen is False

        with raises(ValueError) as verr:
            with rqm.Mocker() as m:
                m.post(self.track_url, text='success')
                decision.variants = test_variants
                decision.givens = test_given
                decision.get()

    def test_get_with_no_tracker(self):
        variants = [el for el in range(10)]

        with raises(ValueError) as verr:
            self.decision_model_without_tracker.choose_from(variants=variants).get()
            assert str(verr.value)

    def test_consistent_encoding(self):
        # since it is impossible to access encoded features from get() call
        #  I'll simply test encoding method used by chooser
        chooser = self.decision_model_with_tracker.chooser

        variants = [{'a': [], 'b': '{} as str'.format(el)} for el in range(5)]
        givens = {'g1': 1, 'g2': '2 as str', 'g3': [0, 1, 2]}
        # consecutive calls with identical seed should return identical encodings
        np.random.seed(0)
        encoded_variants_same_seed_0 = \
            chooser._encode_variants_single_givens(variants=variants, givens=givens)
        np.random.seed(0)
        encoded_variants_same_seed_1 = \
            chooser._encode_variants_single_givens(variants=variants, givens=givens)

        np.testing.assert_array_equal(
            encoded_variants_same_seed_0, encoded_variants_same_seed_1)

        # consecutive calls with no / different seed should return different encodings
        encoded_variants_no_seed_0 = \
            chooser._encode_variants_single_givens(variants=variants, givens=givens)
        encoded_variants_no_seed_1 = \
            chooser._encode_variants_single_givens(variants=variants, givens=givens)

        with np.testing.assert_raises(AssertionError):
            np.testing.assert_array_equal(
                encoded_variants_no_seed_0, encoded_variants_no_seed_1)

    def test_get_consistent_scores(self):
        variants = [{'num': el, 'string': str(el)} for el in range(10)]
        consistency_seed = int(os.getenv('DECISION_TEST_CONSISTENCY_SEED'))

        if consistency_seed is None:
            raise ValueError('consistency seed must not be None')

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success')

            decision_1 = d.Decision(decision_model=self.decision_model_with_tracker)
            decision_1.variants = variants
            np.random.seed(consistency_seed)
            decision_1.get()

            decision_2 = d.Decision(decision_model=self.decision_model_with_tracker)
            decision_2.variants = variants
            np.random.seed(consistency_seed)
            decision_2.get()

            np.testing.assert_array_equal(decision_1.scores, decision_2.scores)

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success')

            decision_3 = d.Decision(decision_model=self.decision_model_with_tracker)
            decision_3.variants = variants
            decision_3.get()

            decision_4 = d.Decision(decision_model=self.decision_model_with_tracker)
            decision_4.variants = variants
            decision_4.get()

            with np.testing.assert_raises(AssertionError):
                np.testing.assert_array_equal(decision_3.scores, decision_4.scores)

    def test_add_reward(self):
        decision = d.Decision(decision_model=self.decision_model_with_tracker)

        reward = 1

        expected_add_reward_body = {
            decision.decision_model._tracker.TYPE_KEY: decision.decision_model._tracker.REWARD_TYPE,
            decision.decision_model._tracker.MODEL_KEY: self.decision_model_with_tracker.model_name,
            decision.decision_model._tracker.REWARD_KEY: reward,
            decision.decision_model._tracker.DECISION_ID_KEY: None,
        }

        def decision_id_matcher(request):
            request_dict = deepcopy(request.json())
            expected_add_reward_body[decision.decision_model._tracker.DECISION_ID_KEY] = \
                request_dict[decision.decision_model._tracker.MESSAGE_ID_KEY]
            return True

        variants = [el for el in range(10)]

        with rqm.Mocker() as m0:
            m0.post(self.track_url, text='success', additional_matcher=decision_id_matcher)
            decision.variants = variants
            decision.get()

        expected_request_json = json.dumps(expected_add_reward_body, sort_keys=False)

        def custom_matcher(request):
            request_dict = deepcopy(request.json())
            del request_dict[decision.decision_model._tracker.MESSAGE_ID_KEY]
            del request_dict[decision.decision_model._tracker.TIMESTAMP_KEY]

            if json.dumps(request_dict, sort_keys=False) != expected_request_json:

                print('raw request body:')
                print(request.text)
                print('compared request string')
                print(json.dumps(request_dict, sort_keys=False))
                print('expected body:')
                print(expected_request_json)
                return None
            return True

        with rqm.Mocker() as m1:
            m1.post(self.track_url, text='success', additional_matcher=custom_matcher)
            decision.add_reward(reward=reward)

    def test_add_reward_string_reward(self):
        decision = d.Decision(decision_model=self.decision_model_with_tracker)

        variants = list(range(20))

        with rqm.Mocker() as m0:
            m0.post(self.track_url, text='success')
            decision.variants = variants
            decision.get()
            # decision.choose_from(variants=variants).get()

        reward = 'string'

        with rqm.Mocker() as m1:
            m1.post(self.track_url, text='success')
            with raises(AssertionError) as aerr:
                decision.add_reward(reward=reward)

    def test_add_reward_inf_reward(self):
        decision = d.Decision(decision_model=self.decision_model_with_tracker)

        variants = list(range(20))

        with rqm.Mocker() as m0:
            m0.post(self.track_url, text='success')
            decision.variants = variants
            decision.get()

        reward = math.inf

        with rqm.Mocker() as m1:
            m1.post(self.track_url, text='success')
            with raises(AssertionError) as aerr:
                decision.add_reward(reward=reward)

        reward = -math.inf

        with rqm.Mocker() as m1:
            m1.post(self.track_url, text='success')
            with raises(AssertionError) as aerr:
                decision.add_reward(reward=reward)

    def test_add_reward_none_reward(self):
        decision = d.Decision(decision_model=self.decision_model_with_tracker)

        variants = list(range(20))

        with rqm.Mocker() as m0:
            m0.post(self.track_url, text='success')
            decision.variants = variants
            decision.get()

        reward = None

        with rqm.Mocker() as m1:
            m1.post(self.track_url, text='success')
            with raises(AssertionError) as aerr:
                decision.add_reward(reward=reward)

        reward = np.nan

        with rqm.Mocker() as m1:
            m1.post(self.track_url, text='success')
            with raises(AssertionError) as aerr:
                decision.add_reward(reward=reward)

    def test__set_message_id_once(self):
        decision = d.Decision(decision_model=self.decision_model_with_tracker)
        decision._set_message_id()

        with warns(UserWarning) as uw:
            decision._set_message_id()
            assert len(uw) != 0
            assert uw.list[0].message

    def test__set_message_id_twice(self):
        decision = d.Decision(decision_model=self.decision_model_with_tracker)
        decision._set_message_id()
        assert decision.id_ is not None

    def _generic_test_peek(self, test_case_filename: str, no_chooser: bool = False):
        path_to_test_case_file = \
            os.sep.join([self.test_jsons_data_directory, test_case_filename])

        test_data = get_test_data(path_to_test_json=path_to_test_case_file, method='read')

        test_case = test_data.get("test_case", None)
        assert test_case is not None

        test_variants = test_case.get('variants', None)
        assert test_variants is not None

        test_givens = test_case.get('givens', None)

        if no_chooser:
            self.decision_model_with_tracker.chooser = None

        decision = d.Decision(decision_model=self.decision_model_with_tracker)

        assert decision.chosen is False

        decision.variants = test_variants
        decision.givens = test_givens

        expected_output = test_data.get('test_output', None)
        assert expected_output is not None

        expected_best = expected_output.get('best', None)
        assert expected_best is not None

        expected_scores = expected_output.get('scores', None)
        assert expected_scores is not None

        decision.variants = test_variants
        decision.givens = test_givens
        calculated_best = decision.peek()

        assert expected_best == calculated_best
        assert decision.id_ is not None

    def test_peek_valid_variants_no_givens(self):
        test_case_filename = \
            os.getenv('DECISION_TEST_PEEK_VALID_VARIANTS_NO_GIVENS_JSON', None)
        assert test_case_filename is not None
        self._generic_test_peek(test_case_filename=test_case_filename)

    def test_peek_valid_variants_no_givens_no_model(self):
        test_case_filename = \
            os.getenv('DECISION_TEST_PEEK_VALID_VARIANTS_NO_GIVENS_NO_MODEL_JSON', None)
        assert test_case_filename is not None
        self._generic_test_peek(test_case_filename=test_case_filename)

    def test_peek_valid_variants_valid_givens(self):
        test_case_filename = \
            os.getenv('DECISION_TEST_PEEK_VALID_VARIANTS_VALID_GIVENS_JSON', None)
        assert test_case_filename is not None
        self._generic_test_peek(test_case_filename=test_case_filename)

    def test_peek_valid_variants_valid_givens_no_model(self):
        test_case_filename = \
            os.getenv('DECISION_TEST_PEEK_VALID_VARIANTS_VALID_GIVENS_NO_MODEL_JSON', None)
        assert test_case_filename is not None
        self._generic_test_peek(test_case_filename=test_case_filename)

    def test_peek_already_chosen(self):
        test_case_filename = \
            os.getenv('DECISION_TEST_PEEK_ALREADY_CHOSEN_JSON', None)
        assert test_case_filename is not None
        self._generic_test_peek(test_case_filename=test_case_filename)

    def test_peek_raises_for_no_variants(self):
        decision = d.Decision(decision_model=self.decision_model_with_tracker)
        best = decision.peek()
        assert best is None

    def test_variants_setter_valid_int_variants(self):
        decision = d.Decision(self.decision_model_with_tracker)
        expected_variants = [1, 2, 3, 4]
        decision.variants = expected_variants

        assert decision.variants == expected_variants

        with warns(UserWarning) as uw:
            decision.variants = expected_variants
            assert len(uw) != 0
            assert uw.list[0].message

    def test_variants_setter_valid_str_variants(self):
        decision = d.Decision(self.decision_model_with_tracker)
        expected_variants = ['1', '2', '3', '4']
        decision.variants = expected_variants

        assert decision.variants == expected_variants

        with warns(UserWarning) as uw:
            decision.variants = expected_variants
            assert len(uw) != 0
            assert uw.list[0].message

    def test_variants_setter_valid_bool_variants(self):
        decision = d.Decision(self.decision_model_with_tracker)
        expected_variants = [True, False]
        decision.variants = expected_variants

        assert decision.variants == expected_variants

        with warns(UserWarning) as uw:
            decision.variants = expected_variants
            assert len(uw) != 0
            assert uw.list[0].message

    def test_variants_setter_valid_object_variants(self):
        decision = d.Decision(self.decision_model_with_tracker)
        expected_variants = [1, {'2': 3, '4': [5, 6], '6': {7: 8}}, [9, 10, 11]]
        decision.variants = expected_variants

        assert decision.variants == expected_variants

        with warns(UserWarning) as uw:
            decision.variants = expected_variants
            assert len(uw) != 0
            assert uw.list[0].message

    def test_variants_setter_raises_for_zero_length_variants(self):
        decision = d.Decision(self.decision_model_with_tracker)
        with raises(ValueError) as verr:
            decision.variants = []

    def test_variants_setter_raises_for_none_variants(self):
        decision = d.Decision(self.decision_model_with_tracker)
        with raises(AssertionError) as aerr:
            decision.variants = None

    def test_variants_setter_raises_for_bad_type_variants(self):
        decision = d.Decision(self.decision_model_with_tracker)
        with raises(AssertionError) as aerr:
            decision.variants = 123

        with raises(AssertionError) as aerr:
            decision.variants = 123.123

        with raises(AssertionError) as aerr:
            decision.variants = '123'

        with raises(AssertionError) as aerr:
            decision.variants = True

        with raises(AssertionError) as aerr:
            decision.variants = {}

        with raises(AssertionError) as aerr:
            decision.variants = {'1': 2, '3': '4'}

    def test_givens_setter_valid_givens(self):
        decision = d.Decision(self.decision_model_with_tracker)
        expected_givens = {'g0': 1, 'g1': 2, 'g3': {1: 2, '3': 4}}
        decision.givens = expected_givens

        assert decision.givens == expected_givens

        with warns(UserWarning) as uw:
            decision.givens = expected_givens
            assert len(uw) != 0
            assert uw.list[0].message

    def test_givens_setter_valid_empty_givens(self):
        decision = d.Decision(self.decision_model_with_tracker)
        expected_givens = {}
        decision.givens = expected_givens

        assert decision.givens == expected_givens

        with warns(UserWarning) as uw:
            decision.givens = expected_givens
            assert len(uw) != 0
            assert uw.list[0].message

    def test_givens_setter_raises_for_bad_type_givens(self):
        decision = d.Decision(self.decision_model_with_tracker)
        with raises(AssertionError) as aerr:
            decision.givens = 123

        with raises(AssertionError) as aerr:
            decision.givens = 123.123

        with raises(AssertionError) as aerr:
            decision.givens = '123'

        with raises(AssertionError) as aerr:
            decision.givens = True

        with raises(AssertionError) as aerr:
            decision.givens = []

        with raises(AssertionError) as aerr:
            decision.givens = np.array([])

        with raises(AssertionError) as aerr:
            decision.givens = tuple([1, 2, 3])

################################################################################

    def test_scores_setter_valid_scores(self):
        decision = d.Decision(self.decision_model_with_tracker)
        expected_variants = [1, 2, 3, 4]
        expected_scores = [0.1, 0.2, 0.3, 0.4]

        decision.variants = expected_variants
        decision.scores = expected_scores

        assert decision.variants == expected_variants
        assert decision.scores == expected_scores

        with warns(UserWarning) as uw:
            decision.variants = expected_variants
            assert len(uw) != 0
            assert uw.list[0].message

    def test_scores_setter_raises_for_scores_shorter_than_variants(self):
        decision = d.Decision(self.decision_model_with_tracker)
        expected_variants = [1, 2, 3]
        decision.variants = expected_variants

        expected_scores = [0.1, 0.2, 0.3, 0.4]

        with raises(AssertionError) as aerr:
            decision.scores = expected_scores

    def test_scores_setter_raises_for_empty_scores(self):
        decision = d.Decision(self.decision_model_with_tracker)
        expected_variants = [1, 2, 3]
        decision.variants = expected_variants
        expected_scores = []

        with raises(AssertionError) as aerr:
            decision.scores = expected_scores

    def test_scores_setter_raises_for_none_scores(self):
        decision = d.Decision(self.decision_model_with_tracker)
        expected_variants = [1, 2, 3]
        decision.variants = expected_variants
        expected_scores = None

        with raises(AssertionError) as aerr:
            decision.scores = expected_scores

    def test_scores_setter_for_bad_types(self):
        decision = d.Decision(self.decision_model_with_tracker)
        expected_variants = [1, 2, 3]
        decision.variants = expected_variants

        with raises(AssertionError) as aerr:
            decision.scores = 123

        with raises(AssertionError) as aerr:
            decision.scores = 123.123

        with raises(AssertionError) as aerr:
            decision.scores = '123'

        with raises(AssertionError) as aerr:
            decision.scores = {'1': 123, '2': '123'}

        with raises(AssertionError) as aerr:
            decision.scores = {'1': 123, '2': '123'}
