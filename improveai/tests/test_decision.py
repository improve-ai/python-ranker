import time
from copy import deepcopy
import json
from ksuid import Ksuid
import math
import requests_mock as rqm
import numpy as np
import os
from pytest import fixture, raises
import sys
from unittest import TestCase
import warnings

sys.path.append(
    os.sep.join(str(os.path.abspath(__file__)).split(os.sep)[:-3]))

import improveai.decision as d
import improveai.decision_model as dm
from improveai.tests.test_utils import get_test_data, is_valid_ksuid


class TestDecision(TestCase):

    @property
    def decision_model_no_track_url(self) -> dm.DecisionModel:
        return self._decision_model

    @decision_model_no_track_url.setter
    def decision_model_no_track_url(self, value: dm.DecisionModel):
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
    def decision_model_valid_track_url(self) -> dm.DecisionModel:
        return self._decision_model_with_tracker

    @decision_model_valid_track_url.setter
    def decision_model_valid_track_url(self, value: dm.DecisionModel):
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

        self.decision_model_no_track_url = \
            dm.DecisionModel(model_name=None)\
            .load(model_url=decision_tests_model_url)

        self.decision_model_valid_track_url = \
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

    def test_raises_value_error_when_model_none(self):
        with raises(AssertionError) as aerr:
            d.Decision(decision_model=None, ranked=[1, 2, 3, 4], givens=None)

    def test_raises_value_error_when_model_str(self):
        with raises(AssertionError) as aerr:
            d.Decision(decision_model='abc', ranked=[1, 2, 3, 4], givens=None)

    def test_raises_value_error_when_model_number(self):
        with raises(AssertionError) as aerr:
            d.Decision(decision_model=123.13, ranked=[1, 2, 3, 4], givens=None)

    def test_raises_value_error_when_model_bool(self):
        with raises(AssertionError) as aerr:
            d.Decision(decision_model=True, ranked=[1, 2, 3, 4], givens=None)

    def _generic_outer_setter_raises_test(
            self, decision: d.Decision, set_attr_name: str,
            set_attr_value: object):

        with raises(AttributeError) as aerr:
            setattr(decision, set_attr_name, set_attr_value)
            assert str(aerr.value) == "AttributeError: can't set attribute"

    def test_model_setter(self):
        decision = d.Decision(
            decision_model=self.decision_model_no_track_url, ranked=[1, 2, 3],
            givens=None)

        assert decision.decision_model is not None
        self._generic_outer_setter_raises_test(
            decision=decision, set_attr_name='decision_model',
            set_attr_value='dummy value')

    def test_model_getter(self):
        decision = d.Decision(
            decision_model=self.decision_model_no_track_url, ranked=[1, 2, 3],
            givens=None)
        assert decision.decision_model is not None
        assert decision.decision_model == self.decision_model_no_track_url

    def test_decision_ranked_variants_setter(self):
        decision = d.Decision(
            decision_model=self.decision_model_no_track_url,
            ranked=self.mockup_variants, givens=None)

        decision.variants = self.mockup_variants

        assert decision.variants is not None
        np.testing.assert_array_equal(decision.variants, self.mockup_variants)

        assert decision.givens is None

        with raises(AssertionError) as aerr:
            setattr(decision, 'variants', 'dummy_value')
            assert str(aerr.value)

    def test_variants_setter_raises_assertion_error_for_string(self):

        with raises(AssertionError) as aerr:
            variants = 'dummy string'
            d.Decision(
                decision_model=self.decision_model_no_track_url, ranked=variants, givens=None)
            assert str(aerr.value)

    def test_variants_setter_raises_assertion_error_for_non_iterable(self):
        with raises(AssertionError) as aerr:
            variants = {'dummy': 'string'}
            d.Decision(
                decision_model=self.decision_model_no_track_url,
                ranked=variants, givens=None)

            assert str(aerr.value)

        with raises(AssertionError) as aerr:
            variants = 1234
            d.Decision(
                decision_model=self.decision_model_no_track_url,
                ranked=variants, givens=None)

            assert str(aerr.value)

        with raises(AssertionError) as aerr:
            variants = 1234.1234
            d.Decision(
                decision_model=self.decision_model_no_track_url,
                ranked=variants, givens=None)

            assert str(aerr.value)

    def test_givens_setter(self):
        decision = d.Decision(
            decision_model=self.decision_model_no_track_url,
            ranked=[1, 2, 3], givens=self.mockup_givens)

        assert decision.givens is not None
        assert decision.givens == self.mockup_givens
        assert isinstance(decision.givens, dict)

        assert decision.ranked == [1, 2, 3]
        assert decision.id_ is None

    def test_givens_setter_raises_attribute_error_for_givens_setting_attempt(self):

        with raises(AttributeError) as atrerr:
            decision = d.Decision(
                decision_model=self.decision_model_no_track_url,
                ranked=[1, 2, 3], givens=None)
            decision.givens = 'dummy string'
            assert str(atrerr.value)

        with raises(AttributeError) as atrerr:
            decision = d.Decision(
                decision_model=self.decision_model_no_track_url,
                ranked=[1, 2, 3], givens=None)
            decision.givens = ['dummy', 'string']
            assert str(atrerr.value)

        with raises(AttributeError) as atrerr:
            decision = d.Decision(
                decision_model=self.decision_model_no_track_url,
                ranked=[1, 2, 3], givens=None)
            decision.givens = 1234
            assert str(atrerr.value)

        with raises(AttributeError) as atrerr:
            decision = d.Decision(
                decision_model=self.decision_model_no_track_url,
                ranked=[1, 2, 3], givens=None)
            decision.givens = 1234.1234
            assert str(atrerr.value)

    def test_givens_setter_raises_assertion_error_for_bad_givens_type(self):

        with raises(AssertionError) as aerr:
            d.Decision(
                decision_model=self.decision_model_no_track_url,
                ranked=[1, 2, 3], givens='dummy string')
            assert str(aerr.value)

        with raises(AssertionError) as aerr:
            d.Decision(
                decision_model=self.decision_model_no_track_url,
                ranked=[1, 2, 3], givens=['dummy', 'string'])
            assert str(aerr.value)

        with raises(AssertionError) as aerr:
            d.Decision(
                decision_model=self.decision_model_no_track_url,
                ranked=[1, 2, 3], givens=1234)
            assert str(aerr.value)

        with raises(AssertionError) as aerr:
            d.Decision(
                decision_model=self.decision_model_no_track_url,
                ranked=[1, 2, 3], givens=1234.1234)
            assert str(aerr.value)

    def _check_decision_before_call(self, decision, test_variants, test_givens):
        assert decision.id_ is None
        assert decision.givens == test_givens
        np.testing.assert_array_equal(decision.ranked, test_variants)

    def _check_decision_after_call(
            self, decision, ranked_variants, test_variants, best_variant, expected_best,
            decision_id, check_ranked: bool = False, check_best: bool = False, check_tracked: bool = False):

        if check_ranked:
            np.testing.assert_array_equal(ranked_variants, decision.ranked)
            np.testing.assert_array_equal(ranked_variants, test_variants)
            assert test_variants[0] == decision.ranked[0]

        if check_best:
            assert best_variant == decision.ranked[0]
            assert test_variants[0] == decision.ranked[0]
            assert best_variant == expected_best

        if check_tracked:
            assert decision.id_ is not None
            # check that decision.id_ is loaded from string without any errors
            assert Ksuid.from_base62(decision.id_)
            assert decision.decision_model.last_decision_id == decision.id_
            assert decision_id == decision.id_

    def test_get_01(self):
        test_case_filename = os.getenv('DECISION_TEST_GET_01_JSON')
        path_to_test_case_file = \
            os.sep.join([self.test_jsons_data_directory, test_case_filename])

        test_data = get_test_data(path_to_test_json=path_to_test_case_file, method='read')

        test_case = test_data.get("test_case", None)
        assert test_case is not None

        test_variants = test_case.get('variants', None)
        assert test_variants is not None

        test_givens = test_case.get('givens', None)
        assert test_givens is not None

        decision = d.Decision(
            decision_model=self.decision_model_valid_track_url,
            ranked=test_variants,
            givens=test_givens)

        self._check_decision_before_call(decision, test_variants, test_givens)

        best_variant = decision.get()

        expected_output = test_data.get('test_output', None)
        assert expected_output is not None

        expected_best = expected_output.get('best', None)
        assert expected_best is not None

        # decision, ranked_variants, test_variants, best_variant, expected_best,
        #             decision_id, check_ranked: bool = False, check_best: bool = False,
        #             check_tracked: bool = False
        self._check_decision_after_call(
            decision=decision, ranked_variants=None, test_variants=test_variants,
            best_variant=best_variant, expected_best=expected_best, decision_id=None,
            check_ranked=False, check_best=True, check_tracked=False)

    def test_ranked_01(self):
        test_case_filename = os.getenv('DECISION_TEST_GET_01_JSON')
        path_to_test_case_file = \
            os.sep.join([self.test_jsons_data_directory, test_case_filename])

        test_data = get_test_data(path_to_test_json=path_to_test_case_file, method='read')

        test_case = test_data.get("test_case", None)
        assert test_case is not None

        test_variants = test_case.get('variants', None)
        assert test_variants is not None

        test_givens = test_case.get('givens', None)
        assert test_givens is not None

        decision = d.Decision(
            decision_model=self.decision_model_valid_track_url,
            ranked=test_variants,
            givens=test_givens)

        ranked_variants = decision.ranked
        np.testing.assert_array_equal(test_variants, decision.ranked)
        np.testing.assert_array_equal(test_variants, ranked_variants)

    def test_track_01(self):

        test_case_filename = os.getenv('DECISION_TEST_GET_01_JSON')
        path_to_test_case_file = \
            os.sep.join([self.test_jsons_data_directory, test_case_filename])

        test_data = get_test_data(path_to_test_json=path_to_test_case_file, method='read')

        test_case = test_data.get("test_case", None)
        assert test_case is not None

        test_variants = test_case.get('variants', None)
        assert test_variants is not None

        test_givens = test_case.get('givens', None)
        assert test_givens is not None

        decision = d.Decision(
            decision_model=self.decision_model_valid_track_url,
            ranked=test_variants,
            givens=test_givens)

        self._check_decision_before_call(decision, test_variants, test_givens)

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success')
            # freeze seed to always replicate track_runners_up value
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                np.random.seed(0)
                decision_id = decision.track()
                time.sleep(0.15)
                assert len(w) == 0

        self._check_decision_after_call(
            decision=decision, ranked_variants=None, test_variants=test_variants,
            best_variant=None, expected_best=None, decision_id=decision_id,
            check_ranked=False, check_best=False, check_tracked=True)

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

        decision = d.Decision(
            decision_model=self.decision_model_valid_track_url,
            ranked=test_variants,
            givens=test_givens)

        self._check_decision_before_call(decision, test_variants, test_givens)

        best_variant = decision.get()

        expected_output = test_data.get('test_output', None)
        assert expected_output is not None

        expected_best = expected_output.get('best', None)
        assert expected_best is not None

        self._check_decision_after_call(
            decision=decision, ranked_variants=None, test_variants=test_variants,
            best_variant=best_variant, expected_best=expected_best, decision_id=None,
            check_ranked=False, check_best=True, check_tracked=False)

    def test_ranked_02(self):
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

        decision = d.Decision(
            decision_model=self.decision_model_valid_track_url,
            ranked=test_variants,
            givens=test_givens)

        self._check_decision_before_call(decision, test_variants, test_givens)

        ranked_variants = decision.ranked

        self._check_decision_after_call(
            decision=decision, ranked_variants=ranked_variants, test_variants=test_variants,
            best_variant=None, expected_best=None, decision_id=None,
            check_ranked=True, check_best=False, check_tracked=False)

    def test_track_02(self):
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

        decision = d.Decision(
            decision_model=self.decision_model_valid_track_url,
            ranked=test_variants,
            givens=test_givens)

        self._check_decision_before_call(decision, test_variants, test_givens)

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success')
            # freeze seed to always replicate track_runners_up value
            np.random.seed(0)
            decision_id = decision.track()
            time.sleep(0.15)

        self._check_decision_after_call(
            decision=decision, ranked_variants=None, test_variants=test_variants,
            best_variant=None, expected_best=None, decision_id=decision_id,
            check_ranked=False, check_best=False, check_tracked=True)

    def test_get_03(self):

        decision = d.Decision(
            decision_model=self.decision_model_valid_track_url,
            ranked=[None], givens=None)

        self._check_decision_before_call(decision, [None], None)

        best_variant = decision.get()

        assert decision.id_ is not None
        assert self.decision_model_valid_track_url.last_decision_id == decision.id_
        np.testing.assert_array_equal(decision.ranked, [None])
        assert best_variant is None

    def test_ranked_03(self):

        decision = d.Decision(
            decision_model=self.decision_model_valid_track_url,
            ranked=[None], givens=None)

        assert decision.id_ is None
        np.testing.assert_array_equal(decision.ranked, [None])
        ranked_variants = decision.ranked

        assert decision.id_ is None
        np.testing.assert_array_equal(decision.ranked, [None])
        np.testing.assert_array_equal(decision.ranked, ranked_variants)

    def test_track_03(self):

        decision = d.Decision(
            decision_model=self.decision_model_valid_track_url,
            ranked=[None], givens=None)

        assert decision.id_ is None

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success')
            decision_id = decision.track()
            time.sleep(0.15)

        assert decision.id_ is not None
        np.testing.assert_array_equal(decision.ranked, [None])
        is_valid_ksuid(decision_id)
        assert decision.get() is None

    def test_get_05(self):

        decision = d.Decision(
            decision_model=self.decision_model_no_track_url,
            ranked=[1, 2, 3],  givens={})

        assert decision.id_ is None

        with raises(AssertionError) as aerr:
            decision.get()

    # TODO check if tracking follows desired frequency (1/min(max_runners_up, len(variants))
    def test_track_06(self):
        # this is a test case which covers tracking runners up from within
        # get() call

        ranked_variants = [el for el in range(100)]
        ranked_variants[0] = {
            "text": "lovely corgi",
            "chars": 12,
            "words": 2}

        runners_up_tracked = []

        def custom_matcher(request):

            request_dict = deepcopy(request.json())
            del request_dict[self.decision_model_valid_track_url.tracker.MESSAGE_ID_KEY]

            if 'runners_up' in request_dict:
                runners_up_tracked.append(True)
            else:
                runners_up_tracked.append(False)
            return True

        for _ in ranked_variants:
            with rqm.Mocker() as m:
                m.post(self.track_url, text='success', additional_matcher=custom_matcher)

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")

                    np.random.seed(self.tracks_seed)
                    decision = d.Decision(
                        decision_model=self.decision_model_valid_track_url,
                        ranked=ranked_variants, givens={})
                    assert decision.id_ is None
                    decision_id = decision.track()
                    time.sleep(0.15)

        assert all(runners_up_tracked)
        is_valid_ksuid(decision.id_)
        is_valid_ksuid(decision_id)
        assert decision.id_ == decision_id

    def test_track_07(self):
        # this is a test case which covers NOT tracking runners up from within
        # _track() call

        ranked_variants = [el for el in range(20)]
        ranked_variants[0] = {
            "text": "lovely corgi",
            "chars": 12,
            "words": 2}

        decision = d.Decision(
            decision_model=self.decision_model_valid_track_url,
            ranked=ranked_variants, givens={})

        assert decision.id_ is None

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success')

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                np.random.seed(self.not_tracks_seed)
                decision_id = decision.track()
                time.sleep(0.15)
                assert len(w) == 0

        is_valid_ksuid(decision.id_)
        is_valid_ksuid(decision_id)
        assert decision.id_ == decision_id

    def test_double_track_raises(self):
        ranked_variants = [el for el in range(20)]
        ranked_variants[0] = {
            "text": "lovely corgi",
            "chars": 12,
            "words": 2}

        decision = d.Decision(
            decision_model=self.decision_model_valid_track_url,
            ranked=ranked_variants, givens={})

        assert decision.id_ is None

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success')
            decision_id = decision.track()
            assert decision_id is not None
            assert decision_id == decision.id_
            is_valid_ksuid(decision_id)

            with raises(AssertionError) as aerr:
                decision.track()

    def _get_complete_tracked_decision(self):
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

        decision = d.Decision(
            decision_model=self.decision_model_valid_track_url,
            ranked=test_variants, givens=test_givens)

        assert decision.id_ is None

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success')
            np.random.seed(0)
            decision.track()
            best_variant = decision.get()
            time.sleep(0.15)

        assert decision.id_ is not None
        return decision, best_variant

    def test_setting_ranked_variants_after_decision_creation_raises(self):
        decision = d.Decision(
            decision_model=self.decision_model_no_track_url,
            ranked=list(range(10)), givens=None)

        assert decision.ranked == list(range(10))
        assert decision.givens is None

    def test_get_with_none_track_url(self):
        variants = [el for el in range(10)]

        with raises(AssertionError) as aerr:
            np.random.seed(1)
            decision = self.decision_model_no_track_url.choose_from(variants=variants, scores=None)
            decision.get()

    def test_consistent_encoding(self):
        # since it is impossible to access encoded features from get() call
        #  I'll simply test encoding method used by chooser
        chooser = self.decision_model_valid_track_url.chooser

        variants = [{'a': [], 'b': '{} as str'.format(el)} for el in range(5)]
        givens = {'g1': 1, 'g2': '2 as str', 'g3': [0, 1, 2]}
        # consecutive calls with identical seed should return identical encodings
        np.random.seed(0)
        encoded_variants_same_seed_0 = \
            chooser.encode_variants_single_givens(variants=variants, givens=givens)
        np.random.seed(0)
        encoded_variants_same_seed_1 = \
            chooser.encode_variants_single_givens(variants=variants, givens=givens)

        np.testing.assert_array_equal(
            encoded_variants_same_seed_0, encoded_variants_same_seed_1)

        # consecutive calls with no / different seed should return different encodings
        encoded_variants_no_seed_0 = \
            chooser.encode_variants_single_givens(variants=variants, givens=givens)
        encoded_variants_no_seed_1 = \
            chooser.encode_variants_single_givens(variants=variants, givens=givens)

        with np.testing.assert_raises(AssertionError):
            np.testing.assert_array_equal(
                encoded_variants_no_seed_0, encoded_variants_no_seed_1)

    def test_add_reward(self):
        variants = [el for el in range(10)]
        decision = d.Decision(
            decision_model=self.decision_model_valid_track_url,
            ranked=variants, givens=None)

        reward = 1

        expected_add_reward_body = {
            decision.decision_model.tracker.TYPE_KEY: decision.decision_model.tracker.REWARD_TYPE,
            decision.decision_model.tracker.MODEL_KEY: self.decision_model_valid_track_url.model_name,
            decision.decision_model.tracker.REWARD_KEY: reward,
            decision.decision_model.tracker.DECISION_ID_KEY: None,
        }

        def decision_id_matcher(request):
            request_dict = deepcopy(request.json())
            expected_add_reward_body[decision.decision_model.tracker.DECISION_ID_KEY] = \
                request_dict[decision.decision_model.tracker.MESSAGE_ID_KEY]
            return True

        with rqm.Mocker() as m0:
            m0.post(self.track_url, text='success', additional_matcher=decision_id_matcher)
            decision.variants = variants
            # freeze seed to assure runners up are not tracked
            np.random.seed(0)
            decision.track()
            time.sleep(0.15)

        expected_request_json = json.dumps(expected_add_reward_body, sort_keys=False)

        def custom_matcher(request):
            request_dict = deepcopy(request.json())
            del request_dict[decision.decision_model.tracker.MESSAGE_ID_KEY]

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
            time.sleep(0.15)

    def test_add_reward_none_track_url(self):
        variants = list(range(20))
        decision = d.Decision(
            decision_model=self.decision_model_no_track_url,
            ranked=variants, givens=None)

        with raises(AssertionError) as aerr:
            decision.track()

    def test_add_reward_string_reward(self):
        variants = list(range(20))
        decision = d.Decision(
            decision_model=self.decision_model_valid_track_url,
            ranked=variants, givens=None)

        with rqm.Mocker() as m0:
            m0.post(self.track_url, text='success')
            np.random.seed(0)
            decision.track()
            time.sleep(0.15)

        reward = 'string'

        with rqm.Mocker() as m1:
            m1.post(self.track_url, text='success')
            with raises(AssertionError) as aerr:
                decision.add_reward(reward=reward)

    def test_add_reward_inf_reward(self):

        variants = list(range(20))
        decision = d.Decision(
            decision_model=self.decision_model_valid_track_url,
            ranked=variants, givens=None)

        with rqm.Mocker() as m0:
            m0.post(self.track_url, text='success')
            np.random.seed(0)
            decision.track()
            time.sleep(0.15)

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
        variants = list(range(20))
        decision = d.Decision(
            decision_model=self.decision_model_valid_track_url, ranked=variants,
            givens=None)

        with rqm.Mocker() as m0:
            m0.post(self.track_url, text='success')
            np.random.seed(0)
            decision.get()
            time.sleep(0.15)

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

    def test_variants_setter_valid_int_variants(self):
        expected_ranked_variants = [1, 2, 3, 4]
        decision = d.Decision(
            decision_model=self.decision_model_valid_track_url,
            ranked=expected_ranked_variants, givens=None)

        np.testing.assert_array_equal(expected_ranked_variants, decision.ranked)

    def test_variants_setter_valid_str_variants(self):
        ranked_expected_variants = ['1', '2', '3', '4']

        decision = d.Decision(
            decision_model=self.decision_model_valid_track_url,
            ranked=ranked_expected_variants, givens=None)

        assert decision.ranked == ranked_expected_variants

    def test_attempt_to_set_ranked_raises(self):
        ranked_expected_variants = ['1', '2', '3', '4']

        decision = d.Decision(
            decision_model=self.decision_model_valid_track_url,
            ranked=ranked_expected_variants, givens=None)

        with raises(AttributeError) as atrrerr:
            decision.ranked = ['4', '3', '2', '1']

        np.testing.assert_array_equal(decision.ranked, ranked_expected_variants)

    def test_attempt_to_set_givens_raises(self):
        ranked_expected_variants = ['1', '2', '3', '4']
        givens = {'test': 'givens'}

        decision = d.Decision(
            decision_model=self.decision_model_valid_track_url,
            ranked=ranked_expected_variants, givens=givens)

        with raises(AttributeError) as atrrerr:
            decision.givens = {'test': 'modified givens'}

        assert decision.givens == givens

    def test_attempt_to_set_id__raises(self):
        ranked_expected_variants = ['1', '2', '3', '4']

        decision = d.Decision(
            decision_model=self.decision_model_valid_track_url,
            ranked=ranked_expected_variants, givens=None)

        with raises(AttributeError) as atrrerr:
            decision.id_ = 'dummy-id'

        assert decision.id_ is None

    def test_attempt_to_set_decision_model_raises(self):
        ranked_expected_variants = ['1', '2', '3', '4']

        decision = d.Decision(
            decision_model=self.decision_model_valid_track_url,
            ranked=ranked_expected_variants, givens=None)

        with raises(AttributeError) as atrrerr:
            decision.decision_model = self.decision_model_no_track_url

        assert decision.decision_model == self.decision_model_valid_track_url

    def test_variants_setter_valid_bool_variants(self):
        expected_ranked_variants = [True, False]
        decision = d.Decision(
            decision_model=self.decision_model_valid_track_url,
            ranked=expected_ranked_variants, givens=None)

        np.testing.assert_array_equal(expected_ranked_variants, decision.ranked)

    def test_variants_setter_valid_object_variants(self):
        expected_ranked_variants = [1, {'2': 3, '4': [5, 6], '6': {7: 8}}, [9, 10, 11]]
        decision = d.Decision(
            decision_model=self.decision_model_valid_track_url,
            ranked=expected_ranked_variants, givens=None)

        np.testing.assert_array_equal(expected_ranked_variants, decision.ranked)

    def test_variants_setter_raises_for_zero_length_variants(self):
        with raises(AssertionError):
            d.Decision(
                decision_model=self.decision_model_valid_track_url,
                ranked=[], givens=None)

        with raises(AssertionError):
            d.Decision(
                decision_model=self.decision_model_valid_track_url,
                ranked=(), givens=None)

        with raises(AssertionError):
            d.Decision(
                decision_model=self.decision_model_valid_track_url,
                ranked=np.array([]), givens=None)

        with raises(AssertionError):
            d.Decision(
                decision_model=self.decision_model_valid_track_url,
                ranked=None, givens=None)

    def test_variants_setter_raises_for_bad_type_variants(self):
        with raises(AssertionError) as aerr:
            d.Decision(
                decision_model=self.decision_model_valid_track_url,
                ranked=123, givens=None)

        with raises(AssertionError) as aerr:
            d.Decision(
                decision_model=self.decision_model_valid_track_url,
                ranked=123.123, givens=None)

        with raises(AssertionError) as aerr:
            d.Decision(
                decision_model=self.decision_model_valid_track_url,
                ranked='123', givens=None)

        with raises(AssertionError) as aerr:
            d.Decision(
                decision_model=self.decision_model_valid_track_url,
                ranked=True, givens=None)

        with raises(AssertionError) as aerr:
            d.Decision(
                decision_model=self.decision_model_valid_track_url,
                ranked={}, givens=None)

        with raises(AssertionError) as aerr:
            d.Decision(
                decision_model=self.decision_model_valid_track_url,
                ranked={'1': 2, '3': '4'}, givens=None)

    def test_givens_setter_valid_givens(self):
        expected_givens = {'g0': 1, 'g1': 2, 'g3': {1: 2, '3': 4}}
        decision = d.Decision(
            decision_model=self.decision_model_valid_track_url,
            ranked=[1, 2, 3, 4], givens=expected_givens)

        assert decision.givens == expected_givens

    def test_decision_raises_for_best_set_attempt(self):

        expected_ranked = [1, 2, 3, 4]
        decision = d.Decision(
            decision_model=self.decision_model_valid_track_url,
            ranked=expected_ranked, givens={})
        with raises(AttributeError) as atrerr:
            decision.best = 5
        assert decision.best == expected_ranked[0]

    def test_givens_setter_valid_empty_givens(self):
        expected_givens = {}
        decision = d.Decision(
            decision_model=self.decision_model_valid_track_url,
            ranked=[1, 2, 3], givens=expected_givens)

        assert decision.givens == expected_givens

    def test_ranked_variants(self):
        tested_ranked_variants = [1, 2, 3, 4]
        decision = d.Decision(
            decision_model=self.decision_model_valid_track_url,
            ranked=tested_ranked_variants, givens={})
        calculated_ranked_variants = decision.ranked
        np.testing.assert_array_equal(tested_ranked_variants, calculated_ranked_variants)
        np.testing.assert_array_equal(tested_ranked_variants, decision.ranked)

    def test_ranked_variants_with_nones(self):
        tested_ranked_variants = [None, 2, None, 4]
        decision = d.Decision(
            decision_model=self.decision_model_valid_track_url,
            ranked=tested_ranked_variants, givens={})
        calculated_ranked_variants = decision.ranked
        assert calculated_ranked_variants[0] is None
        np.testing.assert_array_equal(tested_ranked_variants, calculated_ranked_variants)
        np.testing.assert_array_equal(tested_ranked_variants, decision.ranked)

    def test_best_not_none(self):
        expected_best = 1
        decision = d.Decision(
            decision_model=self.decision_model_valid_track_url,
            ranked=[expected_best, 91, 92, 93], givens={})
        assert decision.best == expected_best
        assert decision.id_ is None

    def test_best_none(self):
        expected_best = None
        decision = d.Decision(
            decision_model=self.decision_model_valid_track_url,
            ranked=[expected_best, 91, 92, 93], givens={})
        assert decision.best == expected_best
        assert decision.id_ is None
