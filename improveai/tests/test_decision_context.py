from copy import deepcopy
import json
import numpy as np
import os
from pytest import fixture, raises
import requests_mock as rqm
from unittest import TestCase

import improveai.decision_model as dm
import improveai.decision_context as dc
from improveai.tests.test_utils import assert_valid_decision, get_test_data, \
    convert_values_to_float32, is_valid_ksuid


class TestDecisionContext(TestCase):

    @property
    def test_decision_model(self):
        return self._test_decision_model

    @test_decision_model.setter
    def test_decision_model(self, value):
        self._test_decision_model = value

    @property
    def valid_test_givens(self):
        return self._valid_test_givens

    @valid_test_givens.setter
    def valid_test_givens(self, value):
        self._valid_test_givens = value

    @property
    def decision_context_test_cases_dir(self):
        return self._decision_context_test_cases_dir

    @decision_context_test_cases_dir.setter
    def decision_context_test_cases_dir(self, value):
        self._decision_context_test_cases_dir = value

    @fixture(autouse=True)
    def prepare_test_artifacts(self):
        self.decision_context_test_cases_dir = os.getenv('DECISION_CONTEXT_TEST_CASES_DIR', None)
        assert self.decision_context_test_cases_dir is not None
        self.test_models_dir = os.getenv('DECISION_MODEL_PREDICTORS_DIR', None)
        assert self.test_models_dir is not None
        self.test_track_url = 'http://mockup.url'
        self.test_decision_model = dm.DecisionModel('dummy-model', track_url=self.test_track_url)

    # test choose_from
    # - valid variants, valid givens
    def test_choose_from_valid_list_variants_valid_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_VALID_GIVENS_NO_MODEL_JSON'),
            tested_method_name='choose_from', variants_converter=list)

    def test_choose_from_valid_tuple_variants_valid_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_VALID_GIVENS_NO_MODEL_JSON'),
            tested_method_name='choose_from', variants_converter=tuple)

    def test_choose_from_valid_nparray_variants_valid_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_VALID_GIVENS_NO_MODEL_JSON'),
            tested_method_name='choose_from', variants_converter=np.array)

    def test_choose_from_valid_list_variants_valid_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_VALID_GIVENS_JSON'),
            tested_method_name='choose_from', variants_converter=list)

    def test_choose_from_valid_tuple_variants_valid_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_VALID_GIVENS_JSON'),
            tested_method_name='choose_from', variants_converter=tuple)

    def test_choose_from_valid_nparray_variants_valid_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_VALID_GIVENS_JSON'),
            tested_method_name='choose_from', variants_converter=np.array)

    # - valid variants, {} givens
    def test_choose_from_valid_list_variants_empty_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_EMPTY_GIVENS_NO_MODEL_JSON'),
            tested_method_name='choose_from', variants_converter=list)

    def test_choose_from_valid_tuple_variants_empty_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_EMPTY_GIVENS_NO_MODEL_JSON'),
            tested_method_name='choose_from', variants_converter=tuple)

    def test_choose_from_valid_nparray_variants_empty_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_EMPTY_GIVENS_NO_MODEL_JSON'),
            tested_method_name='choose_from', variants_converter=np.array)

    def test_choose_from_valid_list_variants_empty_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_EMPTY_GIVENS_JSON'),
            tested_method_name='choose_from', variants_converter=list)

    def test_choose_from_valid_tuple_variants_empty_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_EMPTY_GIVENS_JSON'),
            tested_method_name='choose_from', variants_converter=tuple)

    def test_choose_from_valid_nparray_variants_empty_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_EMPTY_GIVENS_JSON'),
            tested_method_name='choose_from', variants_converter=np.array)

    # - valid variants, None givens
    def test_choose_from_valid_list_variants_none_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_NONE_GIVENS_NO_MODEL_JSON'),
            tested_method_name='choose_from', variants_converter=list)

    def test_choose_from_valid_tuple_variants_none_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_NONE_GIVENS_NO_MODEL_JSON'),
            tested_method_name='choose_from', variants_converter=tuple)

    def test_choose_from_valid_nparray_variants_none_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_NONE_GIVENS_NO_MODEL_JSON'),
            tested_method_name='choose_from', variants_converter=np.array)

    def test_choose_from_valid_list_variants_none_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_NONE_GIVENS_JSON'),
            tested_method_name='choose_from', variants_converter=list)

    def test_choose_from_valid_tuple_variants_none_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_NONE_GIVENS_JSON'),
            tested_method_name='choose_from', variants_converter=tuple)

    def test_choose_from_valid_nparray_variants_none_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_NONE_GIVENS_JSON'),
            tested_method_name='choose_from', variants_converter=np.array)

    # - valid variants, invalid givens
    def test_decision_context_invalid_givens(self):
        invalid_test_givens = ['abc', ['a', 'b', 'c'], ('a', 'b', 'c'), 1234, 1234.1234]
        for ig in invalid_test_givens:
            with raises(AssertionError) as aerr:
                dc.DecisionContext(decision_model=self.test_decision_model, givens=ig)

    # - invalid variants, valid givens
    def test_choose_from_invalid_variants_valid_givens(self):
        invalid_test_variants = ['abc', {'a': 1, 'b': [1, 2, 3], 'c': 'd'}, 1234, 1234.1234]
        valid_givens = {'a': 1, 'b': [1, 2, 3]}
        for iv in invalid_test_variants:
            with raises(AssertionError) as aerr:
                dc.DecisionContext(decision_model=self.test_decision_model, givens=valid_givens)\
                    .choose_from(variants=iv)

    # test score

    def test_score_valid_list_variants_valid_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_VALID_GIVENS_NO_MODEL_JSON'),
            tested_method_name='score', variants_converter=list)

    def test_score_valid_tuple_variants_valid_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_VALID_GIVENS_NO_MODEL_JSON'),
            tested_method_name='score', variants_converter=tuple)

    def test_score_valid_nparray_variants_valid_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_VALID_GIVENS_NO_MODEL_JSON'),
            tested_method_name='score', variants_converter=np.array)

    def test_score_valid_list_variants_valid_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_VALID_GIVENS_JSON'),
            tested_method_name='score', variants_converter=list)

    def test_score_valid_tuple_variants_valid_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_VALID_GIVENS_JSON'),
            tested_method_name='score', variants_converter=tuple)

    def test_score_valid_nparray_variants_valid_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_VALID_GIVENS_JSON'),
            tested_method_name='score', variants_converter=np.array)

    # - valid variants, {} givens
    def test_score_valid_list_variants_empty_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_EMPTY_GIVENS_JSON'),
            tested_method_name='score', variants_converter=list)

    def test_score_valid_tuple_variants_empty_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_EMPTY_GIVENS_JSON'),
            tested_method_name='score', variants_converter=tuple)

    def test_score_valid_nparray_variants_empty_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_EMPTY_GIVENS_JSON'),
            tested_method_name='score', variants_converter=np.array)

    # - valid variants, None givens
    def test_score_valid_list_variants_none_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_NONE_GIVENS_JSON'),
            tested_method_name='score', variants_converter=list)

    def test_score_valid_tuple_variants_none_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_NONE_GIVENS_JSON'),
            tested_method_name='score', variants_converter=tuple)

    def test_score_valid_nparray_variants_none_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_NONE_GIVENS_JSON'),
            tested_method_name='score', variants_converter=np.array)

    # - valid variants, {} givens
    def test_score_valid_list_variants_empty_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_EMPTY_GIVENS_NO_MODEL_JSON'),
            tested_method_name='score', variants_converter=list)

    def test_score_valid_tuple_variants_empty_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_EMPTY_GIVENS_NO_MODEL_JSON'),
            tested_method_name='score', variants_converter=tuple)

    def test_score_valid_nparray_variants_empty_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_EMPTY_GIVENS_NO_MODEL_JSON'),
            tested_method_name='score', variants_converter=np.array)

    # - valid variants, None givens
    def test_score_valid_list_variants_none_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_NONE_GIVENS_NO_MODEL_JSON'),
            tested_method_name='score', variants_converter=list)

    def test_score_valid_tuple_variants_none_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_NONE_GIVENS_NO_MODEL_JSON'),
            tested_method_name='score', variants_converter=tuple)

    def test_score_valid_nparray_variants_none_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_NONE_GIVENS_NO_MODEL_JSON'),
            tested_method_name='score', variants_converter=np.array)

    # - invalid variants, valid givens
    def test_score_invalid_variants_valid_givens(self):
        invalid_test_variants = ['abc', {'a': 1, 'b': [1, 2, 3], 'c': 'd'}, 1234, 1234.1234]
        valid_givens = {'a': 1, 'b': [1, 2, 3]}
        for iv in invalid_test_variants:
            with raises(AssertionError) as aerr:
                dc.DecisionContext(decision_model=self.test_decision_model, givens=valid_givens).score(variants=iv)

    def _generic_test_selected_method(
            self, test_case_json_name, tested_method_name, variants_converter=None):
        path_to_test_json = \
            os.sep.join([self.decision_context_test_cases_dir, test_case_json_name])

        test_case_json = get_test_data(path_to_test_json)

        test_case = test_case_json.get('test_case', None)
        assert test_case is not None

        predictor_filename = test_case.get('model_filename', None)
        if predictor_filename is not None:
            # load model
            model_url = ('{}' + os.sep + '{}').format(self.test_models_dir, predictor_filename)
            self.test_decision_model.load(model_url=model_url)

        # get test variants
        variants = test_case.get('variants', None)
        assert variants is not None
        if variants_converter is not None:
            variants = variants_converter(variants)
        givens = test_case.get('givens', None)
        # assert givens is not None
        scores_seed = test_case_json.get('scores_seed', None)
        assert scores_seed is not None

        expected_output = test_case_json.get('test_output', None)
        assert expected_output is not None
        expected_best = expected_output.get('best', None)
        assert expected_best is not None
        decision_context = dc.DecisionContext(decision_model=self.test_decision_model, givens=givens)

        if tested_method_name == 'score':
            np.random.seed(scores_seed)
            scores = decision_context.score(variants=variants)

            test_output = test_case_json.get('test_output', None)
            assert test_output is not None
            expected_scores = test_output.get('scores', None)
            assert expected_scores is not None
            np.testing.assert_array_equal(
                convert_values_to_float32(scores), convert_values_to_float32(expected_scores))

        elif tested_method_name == 'choose_from':
            np.random.seed(scores_seed)
            decision = decision_context.choose_from(variants=variants, scores=None)

            expected_variants = variants
            expected_givens = givens
            test_output = test_case_json.get('test_output', None)
            assert test_output is not None
            expected_scores = test_output.get('scores', None)
            assert expected_scores is not None
            expected_best = test_output.get('best', None)
            assert expected_best is not None

            assert_valid_decision(
                decision=decision, expected_ranked_variants=expected_variants,
                expected_givens=expected_givens)

        elif tested_method_name == 'which':
            with rqm.Mocker() as m:
                m.post(self.test_track_url, text='success')

                np.random.seed(scores_seed)
                best, decision_id = decision_context.which(variants)
                # print('### BEST ###')
                # print(best)
                # assert False
                assert best == expected_best
                assert is_valid_ksuid(decision_id)

                np.random.seed(scores_seed)
                best, decision_id = decision_context.which(*variants)
                assert best == expected_best
                assert is_valid_ksuid(decision_id)

        elif tested_method_name == 'choose_first':
            np.random.seed(scores_seed)
            decision = decision_context.choose_first(variants=variants)

            assert decision.chosen is True
            assert decision.tracked is False
            expected_scores = expected_output.get('scores', None)
            assert expected_scores is not None
            assert_valid_decision(
                decision=decision, expected_ranked_variants=variants, expected_givens=givens)

        elif tested_method_name == 'first':
            with rqm.Mocker() as m:
                m.post(self.test_track_url, text='success')

                print('### type(variants) ###')
                print(type(variants))

                np.random.seed(scores_seed)
                best, decision_id = decision_context.first(*variants)
                assert best == expected_best
                assert is_valid_ksuid(decision_id)

                np.random.seed(scores_seed)
                best, decision_id = decision_context.first(variants)
                assert best == expected_best
                assert is_valid_ksuid(decision_id)

        elif tested_method_name == 'choose_random':
            np.random.seed(scores_seed)
            decision = decision_context.choose_random(variants=variants)

            assert decision.chosen is True
            assert decision.tracked is False
            expected_scores = expected_output.get('scores', None)
            assert expected_scores is not None
            assert_valid_decision(
                decision=decision, expected_ranked_variants=variants, expected_givens=givens)

        elif tested_method_name == 'random':
            with rqm.Mocker() as m:
                m.post(self.test_track_url, text='success')

                np.random.seed(scores_seed)
                best, decision_id = decision_context.random(*variants)
                assert best == expected_best
                assert is_valid_ksuid(decision_id)

                np.random.seed(scores_seed)
                best, decision_id = decision_context.random(variants)
                assert best == expected_best
                assert is_valid_ksuid(decision_id)
        else:
            raise ValueError(f'tested_method_name: {tested_method_name} not suported')

    def test_which_valid_list_variants_valid_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_VALID_GIVENS_JSON'),
            tested_method_name='which', variants_converter=list)

    def test_which_valid_tuple_variants_valid_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_VALID_GIVENS_JSON'),
            tested_method_name='which', variants_converter=tuple)

    def test_which_valid_ndarray_variants_valid_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_VALID_GIVENS_JSON'),
            tested_method_name='which', variants_converter=np.array)

    def test_which_valid_list_variants_valid_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_VALID_GIVENS_NO_MODEL_JSON'),
            tested_method_name='which', variants_converter=list)

    def test_which_valid_tuple_variants_valid_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_VALID_GIVENS_NO_MODEL_JSON'),
            tested_method_name='which', variants_converter=tuple)

    def test_which_valid_ndarray_variants_valid_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_VALID_GIVENS_NO_MODEL_JSON'),
            tested_method_name='which', variants_converter=np.array)

    def test_which_valid_list_variants_empty_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_EMPTY_GIVENS_JSON'),
            tested_method_name='which', variants_converter=list)

    def test_which_valid_tuple_variants_empty_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_EMPTY_GIVENS_JSON'),
            tested_method_name='which', variants_converter=tuple)

    def test_which_valid_ndarray_variants_empty_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_EMPTY_GIVENS_JSON'),
            tested_method_name='which', variants_converter=np.array)

    def test_which_valid_list_variants_empty_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_EMPTY_GIVENS_NO_MODEL_JSON'),
            tested_method_name='which', variants_converter=list)

    def test_which_valid_tuple_variants_empty_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_EMPTY_GIVENS_NO_MODEL_JSON'),
            tested_method_name='which', variants_converter=tuple)

    def test_which_valid_ndarray_variants_empty_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_EMPTY_GIVENS_NO_MODEL_JSON'),
            tested_method_name='which', variants_converter=np.array)

    def test_which_valid_list_variants_none_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_NONE_GIVENS_JSON'),
            tested_method_name='which', variants_converter=list)

    def test_which_valid_tuple_variants_none_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_NONE_GIVENS_JSON'),
            tested_method_name='which', variants_converter=tuple)

    def test_which_valid_ndarray_variants_none_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_NONE_GIVENS_JSON'),
            tested_method_name='which', variants_converter=np.array)

    def test_which_valid_list_variants_none_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_NONE_GIVENS_NO_MODEL_JSON'),
            tested_method_name='which', variants_converter=list)

    def test_which_valid_tuple_variants_none_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_NONE_GIVENS_NO_MODEL_JSON'),
            tested_method_name='which', variants_converter=tuple)

    def test_which_valid_ndarray_variants_none_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_VALID_VARIANTS_NONE_GIVENS_NO_MODEL_JSON'),
            tested_method_name='which', variants_converter=np.array)

    # - invalid variants, valid givens
    def test_which_invalid_variants_valid_givens_raises(self):
        invalid_test_variants = ['abc', {'a': 1, 'b': [1, 2, 3], 'c': 'd'}, 1234, 1234.1234]
        valid_givens = {'a': 1, 'b': [1, 2, 3]}
        for iv in invalid_test_variants:
            with rqm.Mocker() as m:
                m.post(self.test_track_url, text='success')
                with raises(AssertionError) as aerr:
                    dc.DecisionContext(decision_model=self.test_decision_model, givens=valid_givens).which(*[iv])

    # - invalid variants, valid givens
    def test_which_empty_variants_valid_givens_raises(self):
        invalid_test_variants = [[], tuple([]), np.array([])]
        valid_givens = {'a': 1, 'b': [1, 2, 3]}
        for iv in invalid_test_variants:
            with rqm.Mocker() as m:
                m.post(self.test_track_url, text='success')
                with raises(ValueError) as verr:
                    dc.DecisionContext(decision_model=self.test_decision_model, givens=valid_givens).which(*iv)

    def test_choose_first_valid_list_variants_valid_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_FIRST_VALID_VARIANTS_VALID_GIVENS_JSON'),
            tested_method_name='choose_first', variants_converter=list)

    def test_choose_first_valid_tuple_variants_valid_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_FIRST_VALID_VARIANTS_VALID_GIVENS_JSON'),
            tested_method_name='choose_first', variants_converter=tuple)

    def test_choose_first_valid_ndarray_variants_valid_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_FIRST_VALID_VARIANTS_VALID_GIVENS_JSON'),
            tested_method_name='choose_first', variants_converter=np.array)

    def test_choose_first_valid_list_variants_valid_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_FIRST_VALID_VARIANTS_VALID_GIVENS_NO_MODEL_JSON'),
            tested_method_name='choose_first', variants_converter=list)

    def test_choose_first_valid_tuple_variants_valid_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_FIRST_VALID_VARIANTS_VALID_GIVENS_NO_MODEL_JSON'),
            tested_method_name='choose_first', variants_converter=tuple)

    def test_choose_first_valid_ndarray_variants_valid_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_FIRST_VALID_VARIANTS_VALID_GIVENS_NO_MODEL_JSON'),
            tested_method_name='choose_first', variants_converter=np.array)

    def test_choose_first_valid_list_variants_empty_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_FIRST_VALID_VARIANTS_EMPTY_GIVENS_JSON'),
            tested_method_name='choose_first', variants_converter=list)

    def test_choose_first_valid_tuple_variants_empty_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_FIRST_VALID_VARIANTS_EMPTY_GIVENS_JSON'),
            tested_method_name='choose_first', variants_converter=tuple)

    def test_choose_first_valid_ndarray_variants_empty_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_FIRST_VALID_VARIANTS_EMPTY_GIVENS_JSON'),
            tested_method_name='choose_first', variants_converter=np.array)

    def test_choose_first_valid_list_variants_empty_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_FIRST_VALID_VARIANTS_EMPTY_GIVENS_NO_MODEL_JSON'),
            tested_method_name='choose_first', variants_converter=list)

    def test_choose_first_valid_tuple_variants_empty_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_FIRST_VALID_VARIANTS_EMPTY_GIVENS_NO_MODEL_JSON'),
            tested_method_name='choose_first', variants_converter=tuple)

    def test_choose_first_valid_ndarray_variants_empty_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_FIRST_VALID_VARIANTS_EMPTY_GIVENS_NO_MODEL_JSON'),
            tested_method_name='choose_first', variants_converter=np.array)

    def test_choose_first_valid_list_variants_none_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_FIRST_VALID_VARIANTS_NONE_GIVENS_JSON'),
            tested_method_name='choose_first', variants_converter=list)

    def test_choose_first_valid_tuple_variants_none_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_FIRST_VALID_VARIANTS_NONE_GIVENS_JSON'),
            tested_method_name='choose_first', variants_converter=tuple)

    def test_choose_first_valid_ndarray_variants_none_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_FIRST_VALID_VARIANTS_NONE_GIVENS_JSON'),
            tested_method_name='choose_first', variants_converter=np.array)

    def test_choose_first_valid_list_variants_none_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_FIRST_VALID_VARIANTS_NONE_GIVENS_NO_MODEL_JSON'),
            tested_method_name='choose_first', variants_converter=list)

    def test_choose_first_valid_tuple_variants_none_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_FIRST_VALID_VARIANTS_NONE_GIVENS_NO_MODEL_JSON'),
            tested_method_name='choose_first', variants_converter=tuple)

    def test_choose_first_valid_ndarray_variants_none_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_FIRST_VALID_VARIANTS_NONE_GIVENS_NO_MODEL_JSON'),
            tested_method_name='choose_first', variants_converter=np.array)

    # - invalid variants, valid givens
    def test_choose_first_invalid_variants_valid_givens_raises(self):
        invalid_test_variants = ['abc', {'a': 1, 'b': [1, 2, 3], 'c': 'd'}, 1234, 1234.1234]
        valid_givens = {'a': 1, 'b': [1, 2, 3]}
        for iv in invalid_test_variants:
            with rqm.Mocker() as m:
                m.post(self.test_track_url, text='success')
                with raises(AssertionError) as aerr:
                    dc.DecisionContext(decision_model=self.test_decision_model, givens=valid_givens)\
                        .choose_first(variants=iv)

    # - invalid variants, valid givens
    def test_choose_first_empty_variants_valid_givens_raises(self):
        invalid_test_variants = [[], tuple([]), np.array([])]
        valid_givens = {'a': 1, 'b': [1, 2, 3]}
        for iv in invalid_test_variants:
            with rqm.Mocker() as m:
                m.post(self.test_track_url, text='success')
                with raises(ValueError) as verr:
                    dc.DecisionContext(decision_model=self.test_decision_model, givens=valid_givens)\
                        .choose_first(variants=iv)

    def test_first_valid_list_variants_valid_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_FIRST_VALID_VARIANTS_VALID_GIVENS_JSON'),
            tested_method_name='first', variants_converter=list)

    def test_first_valid_tuple_variants_valid_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_FIRST_VALID_VARIANTS_VALID_GIVENS_JSON'),
            tested_method_name='first', variants_converter=tuple)

    def test_first_valid_ndarray_variants_valid_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_FIRST_VALID_VARIANTS_VALID_GIVENS_JSON'),
            tested_method_name='first', variants_converter=np.array)

    def test_first_valid_list_variants_valid_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_FIRST_VALID_VARIANTS_VALID_GIVENS_NO_MODEL_JSON'),
            tested_method_name='first', variants_converter=list)

    def test_first_valid_tuple_variants_valid_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_FIRST_VALID_VARIANTS_VALID_GIVENS_NO_MODEL_JSON'),
            tested_method_name='first', variants_converter=tuple)

    def test_first_valid_ndarray_variants_valid_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_FIRST_VALID_VARIANTS_VALID_GIVENS_NO_MODEL_JSON'),
            tested_method_name='first', variants_converter=np.array)

    def test_first_valid_list_variants_empty_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_FIRST_VALID_VARIANTS_EMPTY_GIVENS_JSON'),
            tested_method_name='first', variants_converter=list)

    def test_first_valid_tuple_variants_empty_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_FIRST_VALID_VARIANTS_EMPTY_GIVENS_JSON'),
            tested_method_name='first', variants_converter=tuple)

    def test_first_valid_ndarray_variants_empty_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_FIRST_VALID_VARIANTS_EMPTY_GIVENS_JSON'),
            tested_method_name='first', variants_converter=np.array)

    def test_first_valid_list_variants_empty_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_FIRST_VALID_VARIANTS_EMPTY_GIVENS_NO_MODEL_JSON'),
            tested_method_name='first', variants_converter=list)

    def test_first_valid_tuple_variants_empty_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_FIRST_VALID_VARIANTS_EMPTY_GIVENS_NO_MODEL_JSON'),
            tested_method_name='first', variants_converter=tuple)

    def test_first_valid_ndarray_variants_empty_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_FIRST_VALID_VARIANTS_EMPTY_GIVENS_NO_MODEL_JSON'),
            tested_method_name='first', variants_converter=np.array)

    def test_first_valid_list_variants_none_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_FIRST_VALID_VARIANTS_NONE_GIVENS_JSON'),
            tested_method_name='first', variants_converter=list)

    def test_first_valid_tuple_variants_none_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_FIRST_VALID_VARIANTS_NONE_GIVENS_JSON'),
            tested_method_name='first', variants_converter=tuple)

    def test_first_valid_ndarray_variants_none_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_FIRST_VALID_VARIANTS_NONE_GIVENS_JSON'),
            tested_method_name='first', variants_converter=np.array)

    def test_first_valid_list_variants_none_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_FIRST_VALID_VARIANTS_NONE_GIVENS_NO_MODEL_JSON'),
            tested_method_name='first', variants_converter=list)

    def test_first_valid_tuple_variants_none_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_FIRST_VALID_VARIANTS_NONE_GIVENS_NO_MODEL_JSON'),
            tested_method_name='first', variants_converter=tuple)

    def test_first_valid_ndarray_variants_none_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_FIRST_VALID_VARIANTS_NONE_GIVENS_NO_MODEL_JSON'),
            tested_method_name='first', variants_converter=np.array)

    # TODO test *[[1, 2, 3 ,],] being interpreted as a list of variants
    # - invalid variants, valid givens
    def test_first_invalid_variants_valid_givens_raises(self):
        invalid_test_variants = ['abc', {'a': 1, 'b': [1, 2, 3], 'c': 'd'}, 1234, 1234.1234]
        valid_givens = {'a': 1, 'b': [1, 2, 3]}
        for iv in invalid_test_variants:
            with rqm.Mocker() as m:
                m.post(self.test_track_url, text='success')
                with raises(AssertionError) as aerr:
                    dc.DecisionContext(decision_model=self.test_decision_model, givens=valid_givens)\
                        .first(*[iv])

    # - invalid variants, valid givens
    def test_first_empty_variants_valid_givens_raises(self):
        invalid_test_variants = [[], tuple([]), np.array([])]
        valid_givens = {'a': 1, 'b': [1, 2, 3]}
        for iv in invalid_test_variants:
            with rqm.Mocker() as m:
                m.post(self.test_track_url, text='success')
                with raises(ValueError) as verr:
                    dc.DecisionContext(decision_model=self.test_decision_model, givens=valid_givens)\
                        .first(*[iv])

    def test_choose_random_valid_list_variants_valid_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_RANDOM_VALID_VARIANTS_VALID_GIVENS_JSON'),
            tested_method_name='choose_random', variants_converter=list)

    def test_choose_random_valid_tuple_variants_valid_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_RANDOM_VALID_VARIANTS_VALID_GIVENS_JSON'),
            tested_method_name='choose_random', variants_converter=tuple)

    def test_choose_random_valid_ndarray_variants_valid_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_RANDOM_VALID_VARIANTS_VALID_GIVENS_JSON'),
            tested_method_name='choose_random', variants_converter=np.array)

    def test_choose_random_valid_list_variants_valid_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_RANDOM_VALID_VARIANTS_VALID_GIVENS_NO_MODEL_JSON'),
            tested_method_name='choose_random', variants_converter=list)

    def test_choose_random_valid_tuple_variants_valid_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_RANDOM_VALID_VARIANTS_VALID_GIVENS_NO_MODEL_JSON'),
            tested_method_name='choose_random', variants_converter=tuple)

    def test_choose_random_valid_ndarray_variants_valid_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_RANDOM_VALID_VARIANTS_VALID_GIVENS_NO_MODEL_JSON'),
            tested_method_name='choose_random', variants_converter=np.array)

    def test_choose_random_valid_list_variants_empty_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_RANDOM_VALID_VARIANTS_EMPTY_GIVENS_JSON'),
            tested_method_name='choose_random', variants_converter=list)

    def test_choose_random_valid_tuple_variants_empty_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_RANDOM_VALID_VARIANTS_EMPTY_GIVENS_JSON'),
            tested_method_name='choose_random', variants_converter=tuple)

    def test_choose_random_valid_ndarray_variants_empty_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_RANDOM_VALID_VARIANTS_EMPTY_GIVENS_JSON'),
            tested_method_name='choose_random', variants_converter=np.array)

    def test_choose_random_valid_list_variants_empty_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_RANDOM_VALID_VARIANTS_EMPTY_GIVENS_NO_MODEL_JSON'),
            tested_method_name='choose_random', variants_converter=list)

    def test_choose_random_valid_tuple_variants_empty_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_RANDOM_VALID_VARIANTS_EMPTY_GIVENS_NO_MODEL_JSON'),
            tested_method_name='choose_random', variants_converter=tuple)

    def test_choose_random_valid_ndarray_variants_empty_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_RANDOM_VALID_VARIANTS_EMPTY_GIVENS_NO_MODEL_JSON'),
            tested_method_name='choose_random', variants_converter=np.array)

    def test_choose_random_valid_list_variants_none_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_RANDOM_VALID_VARIANTS_NONE_GIVENS_JSON'),
            tested_method_name='choose_random', variants_converter=list)

    def test_choose_random_valid_tuple_variants_none_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_RANDOM_VALID_VARIANTS_NONE_GIVENS_JSON'),
            tested_method_name='choose_random', variants_converter=tuple)

    def test_choose_random_valid_ndarray_variants_none_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_RANDOM_VALID_VARIANTS_NONE_GIVENS_JSON'),
            tested_method_name='choose_random', variants_converter=np.array)

    def test_choose_random_valid_list_variants_none_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_RANDOM_VALID_VARIANTS_NONE_GIVENS_NO_MODEL_JSON'),
            tested_method_name='choose_random', variants_converter=list)

    def test_choose_random_valid_tuple_variants_none_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_RANDOM_VALID_VARIANTS_NONE_GIVENS_NO_MODEL_JSON'),
            tested_method_name='choose_random', variants_converter=tuple)

    def test_choose_random_valid_ndarray_variants_none_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_RANDOM_VALID_VARIANTS_NONE_GIVENS_NO_MODEL_JSON'),
            tested_method_name='choose_random', variants_converter=np.array)

    # TODO test *[[1, 2, 3 ,],] being interpreted as a list of variants
    # - invalid variants, valid givens
    def test_choose_random_invalid_variants_valid_givens_raises(self):
        invalid_test_variants = ['abc', {'a': 1, 'b': [1, 2, 3], 'c': 'd'}, 1234, 1234.1234]
        valid_givens = {'a': 1, 'b': [1, 2, 3]}
        for iv in invalid_test_variants:
            with rqm.Mocker() as m:
                m.post(self.test_track_url, text='success')
                with raises(AssertionError) as aerr:
                    dc.DecisionContext(decision_model=self.test_decision_model, givens=valid_givens)\
                        .choose_random(variants=iv)

    # - invalid variants, valid givens
    def test_choose_random_empty_variants_valid_givens_raises(self):
        invalid_test_variants = [[], tuple([]), np.array([])]
        valid_givens = {'a': 1, 'b': [1, 2, 3]}
        for iv in invalid_test_variants:
            with rqm.Mocker() as m:
                m.post(self.test_track_url, text='success')
                with raises(ValueError) as verr:
                    dc.DecisionContext(decision_model=self.test_decision_model, givens=valid_givens)\
                        .choose_random(variants=iv)

    def test_random_valid_list_variants_valid_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_RANDOM_VALID_VARIANTS_VALID_GIVENS_JSON'),
            tested_method_name='random', variants_converter=list)

    def test_random_valid_tuple_variants_valid_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_RANDOM_VALID_VARIANTS_VALID_GIVENS_JSON'),
            tested_method_name='random', variants_converter=tuple)

    def test_random_valid_ndarray_variants_valid_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_RANDOM_VALID_VARIANTS_VALID_GIVENS_JSON'),
            tested_method_name='random', variants_converter=np.array)

    def test_random_valid_list_variants_valid_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_RANDOM_VALID_VARIANTS_VALID_GIVENS_NO_MODEL_JSON'),
            tested_method_name='random', variants_converter=list)

    def test_random_valid_tuple_variants_valid_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_RANDOM_VALID_VARIANTS_VALID_GIVENS_NO_MODEL_JSON'),
            tested_method_name='random', variants_converter=tuple)

    def test_random_valid_ndarray_variants_valid_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_RANDOM_VALID_VARIANTS_VALID_GIVENS_NO_MODEL_JSON'),
            tested_method_name='random', variants_converter=np.array)

    def test_random_valid_list_variants_empty_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_RANDOM_VALID_VARIANTS_EMPTY_GIVENS_JSON'),
            tested_method_name='random', variants_converter=list)

    def test_random_valid_tuple_variants_empty_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_RANDOM_VALID_VARIANTS_EMPTY_GIVENS_JSON'),
            tested_method_name='random', variants_converter=tuple)

    def test_random_valid_ndarray_variants_empty_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_RANDOM_VALID_VARIANTS_EMPTY_GIVENS_JSON'),
            tested_method_name='random', variants_converter=np.array)

    def test_random_valid_list_variants_empty_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_RANDOM_VALID_VARIANTS_EMPTY_GIVENS_NO_MODEL_JSON'),
            tested_method_name='random', variants_converter=list)

    def test_random_valid_tuple_variants_empty_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_RANDOM_VALID_VARIANTS_EMPTY_GIVENS_NO_MODEL_JSON'),
            tested_method_name='random', variants_converter=tuple)

    def test_random_valid_ndarray_variants_empty_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_RANDOM_VALID_VARIANTS_EMPTY_GIVENS_NO_MODEL_JSON'),
            tested_method_name='random', variants_converter=np.array)

    def test_random_valid_list_variants_none_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_RANDOM_VALID_VARIANTS_NONE_GIVENS_JSON'),
            tested_method_name='random', variants_converter=list)

    def test_random_valid_tuple_variants_none_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_RANDOM_VALID_VARIANTS_NONE_GIVENS_JSON'),
            tested_method_name='random', variants_converter=tuple)

    def test_random_valid_ndarray_variants_none_givens(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_RANDOM_VALID_VARIANTS_NONE_GIVENS_JSON'),
            tested_method_name='random', variants_converter=np.array)

    def test_random_valid_list_variants_none_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_RANDOM_VALID_VARIANTS_NONE_GIVENS_NO_MODEL_JSON'),
            tested_method_name='random', variants_converter=list)

    def test_random_valid_tuple_variants_none_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_RANDOM_VALID_VARIANTS_NONE_GIVENS_NO_MODEL_JSON'),
            tested_method_name='random', variants_converter=tuple)

    def test_random_valid_ndarray_variants_none_givens_no_model(self):
        self._generic_test_selected_method(
            os.getenv('DECISION_CONTEXT_TEST_RANDOM_VALID_VARIANTS_NONE_GIVENS_NO_MODEL_JSON'),
            tested_method_name='random', variants_converter=np.array)

    # TODO test *[[1, 2, 3 ,],] being interpreted as a list of variants
    # - invalid variants, valid givens
    def test_random_invalid_variants_valid_givens_raises(self):
        invalid_test_variants = ['abc', {'a': 1, 'b': [1, 2, 3], 'c': 'd'}, 1234, 1234.1234]
        valid_givens = {'a': 1, 'b': [1, 2, 3]}
        for iv in invalid_test_variants:
            with rqm.Mocker() as m:
                m.post(self.test_track_url, text='success')
                with raises(AssertionError) as aerr:
                    dc.DecisionContext(decision_model=self.test_decision_model, givens=valid_givens).random(*[iv])

    # - invalid variants, valid givens
    def test_random_empty_variants_valid_givens_raises(self):
        invalid_test_variants = [[], tuple([]), np.array([])]
        valid_givens = {'a': 1, 'b': [1, 2, 3]}
        for iv in invalid_test_variants:
            with rqm.Mocker() as m:
                m.post(self.test_track_url, text='success')
                with raises(ValueError) as verr:
                    dc.DecisionContext(decision_model=self.test_decision_model, givens=valid_givens).random(*[iv])

    def test_choose_random_orders_runners_up_randomly(self):
        variants = [1, 2, 3, 4, 5]
        decision_model = dm.DecisionModel('dummy-model', track_url=self.test_track_url)
        decision_context = dc.DecisionContext(decision_model=decision_model, givens=None)

        np.random.seed(1)
        decision = decision_context.choose_random(variants=variants)
        expected_random_scores = \
            [1.6243453636632417, -0.6117564136500754, -0.5281717522634557, -1.0729686221561705, 0.8654076293246785]

        np.testing.assert_array_equal(
            convert_values_to_float32(expected_random_scores),
            convert_values_to_float32(decision.scores))

        request_validity = {'request_body_ok': False}

        decision_tracker = decision_model.tracker

        expected_track_body = {
            decision_tracker.TYPE_KEY: decision_tracker.DECISION_TYPE,
            decision_tracker.MODEL_KEY: decision_model.model_name,
            decision_tracker.VARIANT_KEY: 1,
            decision_tracker.VARIANTS_COUNT_KEY: 5,
            # runners up are shuffled
            decision_tracker.RUNNERS_UP_KEY: [5, 3, 2, 4]
        }

        expected_request_json = json.dumps(expected_track_body, sort_keys=False)

        def custom_matcher(request):
            request_dict = deepcopy(request.json())
            del request_dict[decision_tracker.MESSAGE_ID_KEY]

            if json.dumps(request_dict, sort_keys=False) == expected_request_json:
                request_validity['request_body_ok'] = True

            return True

        tracks_runners_up_seed = os.getenv('DECISION_TRACKER_TRACKS_SEED', None)
        assert tracks_runners_up_seed is not None

        with rqm.Mocker() as m:
            m.post(self.test_track_url, text='success', additional_matcher=custom_matcher)

            np.random.seed(int(tracks_runners_up_seed))
            best = decision.get()
            assert decision.chosen is True
            assert decision.tracked is True
            assert best == 1
