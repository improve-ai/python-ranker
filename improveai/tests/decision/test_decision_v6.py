import warnings
from copy import deepcopy
import json
import requests_mock as rqm
import numpy as np
import os
from pytest import fixture, raises, warns
import sys
from unittest import TestCase

sys.path.append(
    os.sep.join(str(os.path.abspath(__file__)).split(os.sep)[:-3]))

import improveai.decision as d
import improveai.decision_model as dm
import improveai.decision_tracker as dt
from improveai.utils.general_purpose_tools import read_jsonstring_from_file


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
    def mockup_given(self):
        return self._mockup_given

    @mockup_given.setter
    def mockup_given(self, value):
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
        decision_tests_model_url = os.getenv('V6_DUMMY_MODEL_PATH')

        self.test_jsons_data_directory = \
            os.getenv('V6_DECISION_TEST_SUITE_JSONS_DIR')

        self.decision_model_without_tracker = \
            dm.DecisionModel.load(model_url=decision_tests_model_url)

        self.decision_model_with_tracker = \
            dm.DecisionModel.load(model_url=decision_tests_model_url)

        self.track_url = os.getenv('V6_DECISION_TRACKER_TEST_URL')
        self.tracker = \
            dt.DecisionTracker(
                track_url=self.track_url, history_id='dummy-history-id')

        self.decision_model_with_tracker.track_with(tracker=self.tracker)

        self.mockup_variants = [
            {'$value': 123},
            {"$value": {'nested': 'dict'}, }]

        self.mockup_given = {
            'key1': [1, 2, 3],
            'key2': {'nested': 'dict'},
            'key3': 1,
            'key4': 1.0}

        self.dummy_variants = ['dummy_variant']
        self.dummy_givens = {'dummy': 'givens'}

        self.tracks_seed = int(os.getenv('V6_DECISION_TRACKER_TRACKS_SEED'))
        self.not_tracks_seed = \
            int(os.getenv('V6_DECISION_TRACKER_NOT_TRACKS_SEED'))

        self.dummy_history_id = 'dummy-history-id'

    def _get_test_data(
            self, path_to_test_json: str, method: str = 'readlines') -> object:

        loaded_jsonstring = read_jsonstring_from_file(
            path_to_file=path_to_test_json, method=method)

        loaded_json = json.loads(loaded_jsonstring)

        return loaded_json

    # test constructor raises ValueError
    def test_raises_value_error_when_model_none(self):
        with raises(ValueError) as verr:
            d.Decision(decision_model=None)
            # assert that value error is thrown
            assert str(verr.value)

    def _generic_outer_setter_raises_test(
            self, decision: d.Decision, set_attr_name: str,
            set_attr_value: object):
        with raises(AttributeError) as aerr:
            setattr(decision, set_attr_name, set_attr_value)
            assert str(aerr.value) == "AttributeError: can't set attribute"

    def test_model_setter(self):
        decision = d.Decision(decision_model=self.decision_model_without_tracker)
        assert decision.model is not None

        assert decision.variants == [None]
        assert decision.givens is None
        assert decision.chosen is False
        assert decision.memoized_variant is None

        self._generic_outer_setter_raises_test(
            decision=decision, set_attr_name='model',
            set_attr_value='dummy value')

        assert decision.variants == [None]
        assert decision.givens is None
        assert decision.chosen is False
        assert decision.memoized_variant is None

    def test_model_getter(self):
        decision = d.Decision(decision_model=self.decision_model_without_tracker)
        assert decision.model is not None
        assert decision.model == self.decision_model_without_tracker

    def test_choose_from_variants_setter(self):
        decision = \
            d.Decision(decision_model=self.decision_model_without_tracker)\
            .choose_from(self.mockup_variants)

        assert decision.variants is not None
        np.testing.assert_array_equal(decision.variants, self.mockup_variants)

        assert decision.givens is None
        assert decision.chosen is False
        assert decision.memoized_variant is None

        self._generic_outer_setter_raises_test(
            decision=decision, set_attr_name='variants',
            set_attr_value='dummy_value')

    def test_choose_from_variants_setter_raises_type_error_for_string(self):

        with raises(TypeError) as terr:
            d.Decision(decision_model=self.decision_model_without_tracker)\
                .choose_from(variants='dummy string')
            assert str(terr.value)

    def test_choose_from_variants_setter_raises_type_error_for_non_iterable(
            self):
        with raises(TypeError) as terr:
            d.Decision(decision_model=self.decision_model_without_tracker) \
                .choose_from(variants={'dummy': 'string'})
            assert str(terr.value)

        with raises(TypeError) as terr:
            d.Decision(decision_model=self.decision_model_without_tracker) \
                .choose_from(variants=1234)
            assert str(terr.value)

        with raises(TypeError) as terr:
            d.Decision(decision_model=self.decision_model_without_tracker) \
                .choose_from(variants=1234.1234)
            assert str(terr.value)

    def test_givens_setter(self):
        decision = \
            d.Decision(decision_model=self.decision_model_without_tracker)\
            .given(self.mockup_given)

        assert decision.givens is not None
        assert decision.givens == self.mockup_given
        assert isinstance(decision.givens, dict)

        assert decision.variants == [None]
        assert decision.chosen is False
        assert decision.memoized_variant is None

        self._generic_outer_setter_raises_test(
            decision=decision, set_attr_name='givens',
            set_attr_value='dummy_value')

    def test_givens_setter_raises_type_error_for_non_dict(self):

        with raises(TypeError) as terr:
            d.Decision(decision_model=self.decision_model_without_tracker)\
                .given(givens='dummy string')
            assert str(terr.value)

        with raises(TypeError) as terr:
            d.Decision(decision_model=self.decision_model_without_tracker) \
                .given(givens=['dummy', 'string'])
            assert str(terr.value)

        with raises(TypeError) as terr:
            d.Decision(decision_model=self.decision_model_without_tracker) \
                .given(givens=1234)
            assert str(terr.value)

        with raises(TypeError) as terr:
            d.Decision(decision_model=self.decision_model_without_tracker) \
                .given(givens=1234.1234)
            assert str(terr.value)

    # TODO test get() for 100% coverage
    def test_get_01(self):

        test_case_filename = os.getenv('V6_DECISION_TEST_GET_01_JSON')
        path_to_test_case_file = \
            os.sep.join([self.test_jsons_data_directory, test_case_filename])

        test_data = \
            self._get_test_data(
                path_to_test_json=path_to_test_case_file, method='read')

        test_case = test_data.get("test_case", None)
        if test_case is None:
            raise ValueError('`test_case` can`t be empty')

        test_variants = test_case.get('variants', None)

        if test_variants is None:
            raise ValueError('`variants` can`t be empty')

        test_given = test_case.get('given', None)
        if test_given is None:
            raise ValueError('`given` can`t be empty')

        decision = d.Decision(decision_model=self.decision_model_with_tracker)

        assert decision.chosen is False

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success')
            memoized_variant = \
                decision.choose_from(variants=test_variants)\
                .given(givens=test_given).get()

        assert decision.chosen is True

        expected_output = test_data.get('test_output', None)

        if expected_output is None:
            raise ValueError('`test_output` can`t be empty')

        assert memoized_variant == expected_output

    def test_get_02(self):

        test_case_filename = os.getenv('V6_DECISION_TEST_GET_02_JSON')
        path_to_test_case_file = \
            os.sep.join([self.test_jsons_data_directory, test_case_filename])

        test_data = \
            self._get_test_data(
                path_to_test_json=path_to_test_case_file, method='read')

        test_case = test_data.get("test_case", None)
        if test_case is None:
            raise ValueError('`test_case` can`t be empty')

        test_variants = test_case.get('variants', None)
        if test_variants is None:
            raise ValueError('`variants` can`t be empty')

        test_given = test_case.get('given', None)
        if test_given is None:
            raise ValueError('`given` can`t be empty')

        decision = d.Decision(decision_model=self.decision_model_with_tracker)

        assert decision.chosen is False

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success')
            memoized_variant = \
                decision.choose_from(variants=test_variants)\
                .given(givens=test_given).get()

        assert decision.chosen is True

        expected_output = test_data.get('test_output', None)

        if expected_output is None:
            raise ValueError('`test_output` can`t be empty')

        assert memoized_variant == expected_output

    def test_get_03(self):

        decision = d.Decision(decision_model=self.decision_model_with_tracker)

        assert decision.chosen is False

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success')
            memoized_variant = \
                decision.choose_from(variants=[None])\
                .given(givens={}).get()

        assert decision.chosen is True
        assert memoized_variant is None

    def test_get_04(self):

        decision = d.Decision(decision_model=self.decision_model_with_tracker)

        assert decision.chosen is False

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success')
            memoized_variant = \
                decision.choose_from(variants=None)\
                .given(givens={}).get()

        assert decision.chosen is True
        assert memoized_variant is None

    def test_get_05(self):

        tracker = dt.DecisionTracker(
            track_url=self.track_url, history_id=self.dummy_history_id)
        decision = d.Decision(decision_model=self.decision_model_without_tracker)
        decision.model.track_with(tracker=tracker)

        assert decision.chosen is False

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success')

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                np.random.seed(self.tracks_seed)
                memoized_variant = \
                    decision.choose_from(variants=[None])\
                    .given(givens={}).get()
                assert len(w) == 0

        assert decision.chosen is True
        assert memoized_variant is None

    def test_get_06(self):
        # this is a test case which covers tracking runners up from within
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

                np.random.seed(self.tracks_seed)
                memoized_variant = \
                    decision.choose_from(variants=variants)\
                    .given(givens={}).get()

                assert len(w) == 0

        assert decision.chosen is True
        assert memoized_variant == variants[9]

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

                np.random.seed(self.not_tracks_seed)
                memoized_variant = \
                    decision.choose_from(variants=variants)\
                    .given(givens={}).get()
                assert len(w) == 0

        assert decision.chosen is True
        assert memoized_variant == variants[9]

    def _get_complete_decision(self):
        test_case_filename = os.getenv('V6_DECISION_TEST_GET_02_JSON')
        path_to_test_case_file = \
            os.sep.join([self.test_jsons_data_directory, test_case_filename])

        test_data = \
            self._get_test_data(
                path_to_test_json=path_to_test_case_file, method='read')

        test_case = test_data.get("test_case", None)
        if test_case is None:
            raise ValueError('`test_case` can`t be empty')

        test_variants = test_case.get('variants', None)
        if test_variants is None:
            raise ValueError('`variants` can`t be empty')

        test_given = test_case.get('given', None)
        if test_given is None:
            raise ValueError('`given` can`t be empty')

        decision = d.Decision(decision_model=self.decision_model_with_tracker)

        assert decision.chosen is False

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success')
            memoized_variant = \
                decision.choose_from(variants=test_variants)\
                .given(givens=test_given).get()

        assert decision.chosen is True
        return decision, memoized_variant

    def test_choose_from_already_chosen(self):
        decision, memoized_variant = self._get_complete_decision()

        assert decision.chosen is True

        with warns(UserWarning) as uw:
            decision.choose_from(variants=self.dummy_variants)
            assert str(uw.list[0].message) == \
                   'The best variant has already been chosen'

        assert memoized_variant == decision.memoized_variant

    def test_choose_from_variants_already_set(self):
        decision = d.Decision(decision_model=self.decision_model_without_tracker)

        assert decision.variants == [None]
        assert decision.givens is None
        assert decision.chosen is False
        assert decision.memoized_variant is None

        decision.choose_from(variants=self.dummy_variants)

        assert decision.variants == self.dummy_variants
        assert decision.givens is None
        assert decision.chosen is False
        assert decision.memoized_variant is None

        with warns(UserWarning) as uw:
            decision.choose_from(
                variants=self.dummy_variants + self.dummy_variants)
            assert str(uw.list[0].message) == \
                   '`variants` have already been set - ignoring this call'

        assert decision.variants == self.dummy_variants
        assert decision.givens is None
        assert decision.chosen is False
        assert decision.memoized_variant is None

    def test_given_givens_already_set(self):
        decision = d.Decision(decision_model=self.decision_model_without_tracker)

        assert decision.variants == [None]
        assert decision.givens is None
        assert decision.chosen is False
        assert decision.memoized_variant is None

        decision.given(givens=self.dummy_givens)

        assert decision.variants == [None]
        assert decision.givens is self.dummy_givens
        assert decision.chosen is False
        assert decision.memoized_variant is None

        appended_dummy_givens = deepcopy(self.dummy_givens)
        appended_dummy_givens['dummy1'] = 'givens'

        with warns(UserWarning) as uw:
            decision.given(givens=appended_dummy_givens)
            assert str(uw.list[0].message) == \
                   '`givens` have already been set - ignoring this call'

        assert decision.variants == [None]
        assert decision.givens is self.dummy_givens
        assert decision.chosen is False
        assert decision.memoized_variant is None

    def test_given_already_chosen(self):
        decision, memoized_variant = self._get_complete_decision()

        assert decision.chosen is True

        with warns(UserWarning) as uw:
            decision.given(givens={'dummy_given': 'abc'})
            assert str(uw.list[0].message) == \
                   'The best variant has already been chosen'

        assert memoized_variant == decision.memoized_variant

    def test_get_already_chosen(self):
        decision, memoized_variant = self._get_complete_decision()

        assert decision.chosen is True

        with warns(UserWarning) as uw:
            memoized_variant_from_decision = decision.get()
            assert str(uw.list[0].message) == \
                   'The best variant has already been chosen'

        assert memoized_variant == decision.memoized_variant == \
               memoized_variant_from_decision

    def test_memoized_variant_set_from_outside(self):
        decision = d.Decision(decision_model=self.decision_model_without_tracker)

        with raises(AttributeError) as aerr:
            decision.memoized_variant = 'dummy_variant'
            assert str(aerr.value)

    def test_get_with_zero_len_variants(self):

        test_case_filename = \
            os.getenv('V6_DECISION_TEST_GET_ZERO_LENGTH_VARIANTS_JSON')
        path_to_test_case_file = \
            os.sep.join([self.test_jsons_data_directory, test_case_filename])

        test_data = \
            self._get_test_data(
                path_to_test_json=path_to_test_case_file, method='read')

        test_case = test_data.get("test_case", None)
        if test_case is None:
            raise ValueError('`test_case` can`t be empty')

        test_variants = test_case.get('variants', None)
        if test_variants is None:
            raise ValueError('`variants` can`t be empty')

        test_given = test_case.get('given', None)
        if test_given is None:
            raise ValueError('`given` can`t be empty')

        decision = d.Decision(decision_model=self.decision_model_with_tracker)

        assert decision.chosen is False

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success')
            memoized_variant = \
                decision.choose_from(variants=test_variants)\
                .given(givens=test_given).get()

        assert decision.chosen is True

        expected_output = test_data.get('test_output', None)

        assert memoized_variant == expected_output

    def test_get_with_no_tracker(self):
        variants = [el for el in range(10)]

        with raises(ValueError) as verr:
            d.Decision(decision_model=self.decision_model_without_tracker)\
                .choose_from(variants=variants).get()
            assert str(verr.value)
