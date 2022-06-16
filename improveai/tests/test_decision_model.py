from copy import deepcopy
import json
from ksuid import Ksuid
import numpy as np
import math
import os
from pytest import fixture, raises
import requests_mock as rqm
import string
import sys
from unittest import TestCase
import warnings
import xgboost as xgb

sys.path.append(
    os.sep.join(str(os.path.abspath(__file__)).split(os.sep)[:-3]))

import improveai.decision as d
import improveai.decision_context as dc
import improveai.decision_model as dm
import improveai.decision_tracker as dt
from improveai.chooser import XGBChooser
from improveai.tests.test_utils import convert_values_to_float32, get_test_data, \
    assert_valid_decision, is_valid_ksuid


class TestDecisionModel(TestCase):

    BAD_SPECIAL_CHARACTERS = [el for el in '`~!@#$%^&*()=+[]{};:"<>,/?' + "'"]
    ALNUM_CHARS = [el for el in string.digits + string.ascii_letters]

    @property
    def score_seed(self):
        return self._score_seed

    @score_seed.setter
    def score_seed(self, value):
        self._score_seed = value

    @property
    def test_cases_directory(self) -> str:
        return self._test_cases_directory

    @test_cases_directory.setter
    def test_cases_directory(self, value: str):
        self._test_cases_directory = value

    @property
    def predictors_fs_directory(self) -> str:
        return self._predictors_fs_directory

    @predictors_fs_directory.setter
    def predictors_fs_directory(self, value: str):
        self._predictors_fs_directory = value

    @fixture(autouse=True)
    def prepare_env(self):
        self.test_cases_directory = \
            os.getenv('DECISION_MODEL_TEST_CASES_DIRECTORY')

        self.predictors_fs_directory = \
            os.getenv('DECISION_MODEL_PREDICTORS_DIR')

        self.track_url = os.getenv('DECISION_TRACKER_TEST_URL', None)
        assert self.track_url is not None

    def _assert_metadata_entries_equal(
            self, tested_metadata: dict, expected_metadata: dict,
            asserted_key: str):

        tested_value = \
            tested_metadata.get(asserted_key, None)

        if tested_value is None:
            raise ValueError('Tested `{}` can`t be None'.format(asserted_key))

        expected_value = \
            expected_metadata.get(asserted_key, None)

        if expected_value is None:
            raise ValueError('Expected `{}` can`t be None'.format(asserted_key))

        assert tested_value == expected_value

    def _generic_test_loaded_model(
            self, test_data_filename: str, expected_predictor_type: object,
            test_case_key: str = 'test_case',
            test_output_key: str = 'test_output',
            test_output_feature_names_key: str = 'feature_names',
            test_output_model_name_key: str = 'model_name',
            test_output_model_seed_key: str = 'model_seed',
            test_output_version_key: str = 'version',
            model_filename_key: str = 'model_filename', load_mode: str = 'sync'):

        path_to_test_json = \
            ('{}' + os.sep + '{}').format(
                self.test_cases_directory, test_data_filename)

        test_data = get_test_data(path_to_test_json=path_to_test_json, method='read')

        test_case = test_data.get(test_case_key, None)

        if test_case is None:
            raise ValueError('Test case can`t be None')

        predictor_filename = test_case.get(model_filename_key, None)

        if predictor_filename is None:
            raise ValueError('Model filename can`t be None')

        model_url = \
            ('{}' + os.sep + '{}').format(
                self.predictors_fs_directory, predictor_filename)
        # loading model
        if load_mode == 'sync':
            decision_model = dm.DecisionModel(model_name=None).load(model_url=model_url)
        else:
            raise ValueError('Allowed values for `load_mode` are sync and async')

        # is returned object a decision model
        assert isinstance(decision_model, dm.DecisionModel)

        # is predictor of a desired type
        print('decision_model.chooser')
        print(decision_model.chooser)
        print(expected_predictor_type)
        print(model_url)
        predictor = decision_model.chooser.model
        assert isinstance(predictor, expected_predictor_type)

        # has predictor got all desired metadata
        metadata = \
            json.loads(
                predictor.attr('user_defined_metadata')).get('json', None) \
            if isinstance(predictor, xgb.Booster) \
            else json.loads(
                getattr(predictor, 'user_defined_metadata').get('json', None))

        if metadata is None:
            raise ValueError('Model metadata can`t be None')

        test_output = test_data.get(test_output_key, None)

        if test_output is None:
            raise ValueError('Test output can`t be None')

        print(test_output)

        self._assert_metadata_entries_equal(
            tested_metadata=metadata, expected_metadata=test_output,
            asserted_key=test_output_feature_names_key)

        self._assert_metadata_entries_equal(
            tested_metadata=metadata, expected_metadata=test_output,
            asserted_key=test_output_model_name_key)

        self._assert_metadata_entries_equal(
            tested_metadata=metadata, expected_metadata=test_output,
            asserted_key=test_output_model_seed_key)

        self._assert_metadata_entries_equal(
            tested_metadata=metadata, expected_metadata=test_output,
            asserted_key=test_output_version_key)

    def _generic_test_loaded_fs_none_model(
            self, test_data_filename: str, test_case_key: str = 'test_case',
            model_filename_key: str = 'model_filename', load_mode: str = 'sync'):
        path_to_test_json = \
            ('{}' + os.sep + '{}').format(
                self.test_cases_directory, test_data_filename)

        test_data = \
            get_test_data(
                path_to_test_json=path_to_test_json, method='read')

        test_case = test_data.get(test_case_key, None)

        if test_case is None:
            raise ValueError('Test case can`t be None')

        predictor_filename = test_case.get(model_filename_key, None)

        if predictor_filename is None:
            raise ValueError('Model filename can`t be None')

        model_url = \
            ('{}' + os.sep + '{}').format(
                self.predictors_fs_directory, predictor_filename)

        # loading model
        with raises(ValueError) as verr:
            if load_mode == 'sync':
                decision_model = dm.DecisionModel(model_name=None).load(model_url=model_url)
            else:
                raise RuntimeError(
                    'Allowed values for `load_mode` are sync and async')

            assert str(verr.value)

    def _generic_test_loaded_url_none_model(
            self, test_data_filename: str, test_case_key: str = 'test_case',
            model_url_key: str = 'model_url', load_mode: str = 'sync'):

        path_to_test_json = \
            ('{}' + os.sep + '{}').format(
                self.test_cases_directory, test_data_filename)

        test_data = get_test_data(path_to_test_json=path_to_test_json, method='read')

        test_case = test_data.get(test_case_key, None)

        if test_case is None:
            raise ValueError('Test case can`t be None')

        model_url = test_case.get(model_url_key, None)

        if model_url is None:
            raise ValueError('Model filename can`t be None')

        # loading model
        with raises(ValueError) as verr:
            if load_mode == 'sync':
                decision_model = dm.DecisionModel(model_name=None).load(model_url=model_url)
            else:
                raise RuntimeError(
                    'Allowed values for `load_mode` are sync and async')

            assert str(verr.value)

    def test_constructor_valid_track_url(self):

        test_model_name = 'dummy-model'
        decision_model = \
            dm.DecisionModel(model_name=test_model_name, track_url=self.track_url)
        assert decision_model.model_name == test_model_name
        assert decision_model.track_url == self.track_url
        assert decision_model.tracker is not None and isinstance(decision_model.tracker, dt.DecisionTracker)
        assert decision_model.tracker.track_url == self.track_url

    def test_constructor_none_track_url(self):
        decision_model = dm.DecisionModel(model_name='dummy-model')
        assert decision_model.model_name == 'dummy-model'
        assert decision_model.track_url is None
        assert decision_model.tracker is not None and isinstance(decision_model.tracker, dt.DecisionTracker)
        assert decision_model.tracker.track_url is None

    # test model loading
    def test_load_model_sync_native_fs(self):

        self._generic_test_loaded_model(
            test_data_filename=os.getenv(
                'DECISION_MODEL_TEST_LOAD_MODEL_FS_NATIVE_JSON'),
            expected_predictor_type=xgb.Booster)

    def test_load_model_sync_native_fs_no_model(self):

        self._generic_test_loaded_fs_none_model(
            test_data_filename=os.getenv(
                'DECISION_MODEL_TEST_LOAD_MODEL_FS_NATIVE_NO_MODEL_JSON'))

    def test_load_model_sync_native_url_no_model(self):
        self._generic_test_loaded_url_none_model(
            test_data_filename=os.getenv(
                'DECISION_MODEL_TEST_LOAD_MODEL_URL_NO_MODEL_JSON'))

    def _generic_desired_decision_model_method_call(
            self, test_data_filename: str, evaluated_method_name: str,
            empty_callable_kwargs: dict, test_case_key: str = 'test_case',
            test_output_key: str = 'test_output',
            variants_key: str = 'variants', givens_key: str = 'givens',
            predictor_filename_key: str = 'model_filename',
            scores_key: str = 'scores', scores_seed_key: str = 'scores_seed'):

        path_to_test_json = \
            ('{}' + os.sep + '{}').format(
                self.test_cases_directory, test_data_filename)

        test_data = get_test_data(path_to_test_json=path_to_test_json, method='read')

        test_case = test_data.get(test_case_key, None)

        if test_case is None:
            raise ValueError('Test case can`t be None')

        variants = test_case.get(variants_key, None)

        if variants_key in empty_callable_kwargs:
            empty_callable_kwargs[variants_key] = variants

        if variants is None:
            raise ValueError('Variants can`t be None')

        givens = test_case.get(givens_key, None)

        if givens_key in empty_callable_kwargs:
            empty_callable_kwargs[givens_key] = givens

        if givens is None:
            raise ValueError('Context can`t be None')

        score_seed = test_data.get(scores_seed_key, None)

        if score_seed is None:
            raise ValueError('`scores_seed` can`t be empty')

        predictor_filename = test_case.get(predictor_filename_key, None)

        if predictor_filename is None:
            raise ValueError('`model_filename` can`t be empty')

        expected_output = test_data.get(test_output_key, None)

        if expected_output is None:
            raise ValueError('`test_output` can`t be None')

        model_url = \
            ('{}' + os.sep + '{}').format(
                self.predictors_fs_directory, predictor_filename)

        decision_model = dm.DecisionModel(model_name=None).load(model_url=model_url)

        expected_scores = None
        if evaluated_method_name in ['score', '_score']:
            expected_scores = expected_output.get('scores', None)

        if evaluated_method_name == 'score':
            np.random.seed(score_seed)
            calculated_scores = decision_model.score(variants=variants)
            calculated_scores_float32 = convert_values_to_float32(calculated_scores)

            assert expected_scores is not None
            expected_output_float32 = convert_values_to_float32(expected_scores)
            np.testing.assert_array_equal(calculated_scores_float32, expected_output_float32)
            return

        np.random.seed(score_seed)
        # calculated_scores = decision_model.score(**empty_callable_kwargs)
        calculated_scores = decision_model._score(variants=variants, givens=givens)

        calculated_scores_float32 = convert_values_to_float32(calculated_scores)

        if evaluated_method_name == '_score':
            assert expected_scores is not None
            expected_output_float32 = convert_values_to_float32(expected_scores)

            np.testing.assert_array_equal(
                calculated_scores_float32, expected_output_float32)
            return

        if scores_key in empty_callable_kwargs:
            empty_callable_kwargs[scores_key] = calculated_scores_float32

        np.random.seed(score_seed)
        # evaluated_callable = getattr(dm.DecisionModel, evaluated_method_name)
        if evaluated_method_name == 'top_scoring_variant':
            calculated_best = \
                dm.DecisionModel.top_scoring_variant(variants=variants, scores=calculated_scores)
            expected_best = expected_output.get('best', None)
            assert expected_best is not None
            assert calculated_best == expected_best
        elif evaluated_method_name == 'rank':
            calculated_ranked_variants = \
                dm.DecisionModel.rank(variants=variants, scores=calculated_scores)
            expected_ranked_variants = expected_output.get('ranked_variants', None)
            np.testing.assert_array_equal(
                expected_ranked_variants, calculated_ranked_variants)
        elif evaluated_method_name == 'choose_first':
            # get decision object
            decision = decision_model.choose_first(variants=variants)

            expected_scores = expected_output.get('scores', None)
            assert expected_scores is not None
            expected_best = expected_output.get('best', None)
            assert expected_best is not None

            # assert that returned decision is correct
            assert_valid_decision(
                decision=decision, expected_variants=variants, expected_givens=givens,
                expected_scores=expected_scores, expected_best=expected_best)
        elif evaluated_method_name == 'first':
            # assert that returned variant is correct
            best_variant, decision_id = decision_model.first(*variants)
            # assert that best variant is equal to expected output

            expected_best = expected_output.get('best', None)
            assert expected_best is not None

            assert best_variant == expected_best
            assert is_valid_ksuid(decision_id)

            # assert that returned variant is correct
            best_variant, decision_id = decision_model.first(variants)
            # assert that best variant is equal to expected output

            expected_best = expected_output.get('best', None)
            assert expected_best is not None

            assert best_variant == expected_best
            assert is_valid_ksuid(decision_id)

        elif evaluated_method_name == 'choose_random':
            decision = decision_model.choose_random(variants=variants)

            expected_scores = expected_output.get('scores', None)
            assert expected_scores is not None
            expected_best = expected_output.get('best', None)
            assert expected_best is not None

            # assert that returned decision is correct
            assert_valid_decision(
                decision=decision, expected_variants=variants, expected_givens=givens,
                expected_scores=expected_scores, expected_best=expected_best)
        elif evaluated_method_name == 'random':
            # assert that returned variant is correct
            best_variant, decision_id = decision_model.random(*variants)
            # assert that best variant is equal to expected output

            expected_best = expected_output.get('best', None)
            assert expected_best is not None

            assert best_variant == expected_best
            assert is_valid_ksuid(decision_id)

            best_variant, decision_id = decision_model.random(*variants)

            expected_best = expected_output.get('best', None)
            assert expected_best is not None

            assert best_variant == expected_best
            assert is_valid_ksuid(decision_id)
        else:
            raise ValueError('Unsupported method: {}'.format(evaluated_method_name))

    def _generic_desired_decision_model_method_call_no_model(
            self, test_data_filename: str, evaluated_method_name: str,
            test_case_key: str = 'test_case', test_output_key: str = 'test_output',
            variants_key: str = 'variants', variants_input_type: str = 'list',
            givens_key: str = 'givens',
            scores_seed_key: str = 'scores_seed'):

        path_to_test_json = \
            ('{}' + os.sep + '{}').format(
                self.test_cases_directory, test_data_filename)

        test_data = get_test_data(path_to_test_json=path_to_test_json, method='read')

        test_case = test_data.get(test_case_key, None)
        assert test_case is not None

        variants = test_case.get(variants_key, None)
        assert variants is not None

        if variants_input_type == 'numpy':
            variants = np.array(variants)
        elif variants_input_type == 'tuple':
            variants = tuple(variants)

        givens = test_case.get(givens_key, None)

        score_seed = test_data.get(scores_seed_key, None)
        assert score_seed is not None

        # predictor_filename = test_case.get(predictor_filename_key, None)
        # assert predictor_filename is not None

        expected_output = test_data.get(test_output_key, None)
        assert expected_output is not None

        decision_model = dm.DecisionModel(model_name='dummy-model')

        expected_scores = None
        if evaluated_method_name in ['score', '_score']:
            expected_scores = expected_output.get('scores', None)

        if evaluated_method_name == 'score':
            np.random.seed(score_seed)
            calculated_scores = decision_model.score(variants=variants)
            calculated_scores_float32 = convert_values_to_float32(calculated_scores)

            assert expected_scores is not None
            expected_output_float32 = convert_values_to_float32(expected_scores)
            np.testing.assert_array_equal(calculated_scores_float32, expected_output_float32)
            return

        np.random.seed(score_seed)
        calculated_scores = decision_model._score(variants=variants, givens=givens)

        calculated_scores_float32 = convert_values_to_float32(calculated_scores)

        if evaluated_method_name == '_score':
            assert expected_scores is not None
            expected_output_float32 = convert_values_to_float32(expected_scores)

            np.testing.assert_array_equal(
                calculated_scores_float32, expected_output_float32)
            return

        np.random.seed(score_seed)
        if evaluated_method_name == 'top_scoring_variant':
            calculated_best = \
                dm.DecisionModel.top_scoring_variant(variants=variants, scores=calculated_scores)
            expected_best = expected_output.get('best', None)
            assert expected_best is not None
            assert calculated_best == expected_best
        elif evaluated_method_name == 'rank':
            calculated_ranked_variants = \
                dm.DecisionModel.rank(variants=variants, scores=calculated_scores)
            expected_ranked_variants = expected_output.get('ranked_variants', None)
            np.testing.assert_array_equal(
                expected_ranked_variants, calculated_ranked_variants)
        elif evaluated_method_name == 'choose_first':
            # get decision object
            np.random.seed(score_seed)
            decision = decision_model.choose_first(variants=variants)

            expected_scores = expected_output.get('scores', None)
            assert expected_scores is not None
            expected_best = expected_output.get('best', None)
            assert expected_best is not None

            # assert that returned decision is correct
            assert_valid_decision(
                decision=decision, expected_variants=variants, expected_givens=givens,
                expected_scores=expected_scores, expected_best=expected_best)
        elif evaluated_method_name == 'first':
            # assert that returned variant is correct
            with rqm.Mocker() as m:
                m.post(self.track_url, text='success')
                decision_model = dm.DecisionModel(model_name='dummy-model', track_url=self.track_url)

                np.random.seed(score_seed)
                best_variant, decision_id = decision_model.first(*variants)
                # assert that best variant is equal to expected output
                expected_best = expected_output.get('best', None)

                assert best_variant == expected_best
                assert is_valid_ksuid(decision_id)

                np.random.seed(score_seed)
                best_variant, decision_id = decision_model.first(variants)
                # assert that best variant is equal to expected output
                expected_best = expected_output.get('best', None)

                assert best_variant == expected_best
                assert is_valid_ksuid(decision_id)

        elif evaluated_method_name == 'choose_random':
            np.random.seed(score_seed)
            decision = decision_model.choose_random(variants=variants)

            expected_scores = expected_output.get('scores', None)
            assert expected_scores is not None
            expected_best = expected_output.get('best', None)
            assert expected_best is not None

            # assert that returned decision is correct
            assert_valid_decision(
                decision=decision, expected_variants=variants, expected_givens=givens,
                expected_scores=expected_scores, expected_best=expected_best)
        elif evaluated_method_name == 'random':
            # assert that returned variant is correct
            with rqm.Mocker() as m:
                m.post(self.track_url, text='success')
                decision_model = dm.DecisionModel(model_name='dummy-model', track_url=self.track_url)

                np.random.seed(score_seed)
                best_variant, decision_id = decision_model.random(*variants)

                # assert that best variant is equal to expected output
                expected_best = expected_output.get('best', None)
                assert expected_best is not None

                assert best_variant == expected_best
                assert is_valid_ksuid(decision_id)

                np.random.seed(score_seed)
                best_variant, decision_id = decision_model.random(*variants)

                # assert that best variant is equal to expected output
                expected_best = expected_output.get('best', None)
                assert expected_best is not None

                assert best_variant == expected_best
                assert is_valid_ksuid(decision_id)

        else:
            raise ValueError('Unsupported method: {}'.format(evaluated_method_name))

    def test__score_no_model(self):
        self._generic_desired_decision_model_method_call_no_model(
            test_data_filename=os.getenv(
                'DECISION_MODEL_TEST__SCORE_NATIVE_NO_MODEL_JSON'),
            evaluated_method_name='_score')

    def test__score(self):
        self._generic_desired_decision_model_method_call(
            test_data_filename=os.getenv(
                'DECISION_MODEL_TEST__SCORE_NATIVE_JSON'),
            evaluated_method_name='_score',
            empty_callable_kwargs={'variants': None, 'givens': None})

    def test_score_no_model(self):
        self._generic_desired_decision_model_method_call_no_model(
            test_data_filename=os.getenv(
                'DECISION_MODEL_TEST_SCORE_NATIVE_NO_MODEL_JSON'),
            evaluated_method_name='score')

    def test_score(self):
        self._generic_desired_decision_model_method_call(
            test_data_filename=os.getenv(
                'DECISION_MODEL_TEST_SCORE_NATIVE_JSON'),
            evaluated_method_name='score',
            empty_callable_kwargs={'variants': None, 'givens': None})

    def test_top_scoring_variant_no_model(self):
        self._generic_desired_decision_model_method_call_no_model(
            test_data_filename=os.getenv(
                'DECISION_MODEL_TEST_TOP_SCORING_VARIANT_NATIVE_NO_MODEL_JSON'),
            evaluated_method_name='top_scoring_variant')

    def test_top_scoring_variant(self):
        self._generic_desired_decision_model_method_call(
            test_data_filename=os.getenv(
                'DECISION_MODEL_TEST_TOP_SCORING_VARIANT_NATIVE_JSON'),
            evaluated_method_name='top_scoring_variant',
            empty_callable_kwargs={
                'variants': None, 'givens': None, 'scores': None})

    def test_ranked_no_model(self):
        self._generic_desired_decision_model_method_call_no_model(
            test_data_filename=os.getenv(
                'DECISION_MODEL_TEST_RANK_NATIVE_NO_MODEL_JSON'),
            evaluated_method_name='rank')

    def test_ranked(self):
        self._generic_desired_decision_model_method_call(
            test_data_filename=os.getenv(
                'DECISION_MODEL_TEST_RANK_NATIVE_JSON'),
            evaluated_method_name='rank',
            empty_callable_kwargs={
                'variants': None, 'givens': None, 'scores': None})

    def test_generate_descending_gaussians(self):
        path_to_test_json = \
            ('{}' + os.sep + '{}').format(
                self.test_cases_directory,
                os.getenv('DECISION_MODEL_TEST_GENERATE_DESCENDING_GAUSSIANS_JSON'))

        test_data = get_test_data(path_to_test_json=path_to_test_json, method='read')

        test_case = test_data.get("test_case", None)

        if test_case is None:
            raise ValueError('Test case can`t be None')

        variants_count = test_case.get("count", None)

        if variants_count is None:
            raise ValueError('Variants count can`t be None')

        score_seed = test_data.get("scores_seed", None)

        expected_output = test_data.get("test_output", None)
        assert expected_output is not None

        expected_scores = expected_output.get('scores', None)
        assert expected_scores is not None

        np.random.seed(score_seed)
        calculated_gaussians = \
            dm.DecisionModel.generate_descending_gaussians(count=variants_count)

        np.testing.assert_array_equal(calculated_gaussians, expected_scores)

    def test_choose_from(self):

        path_to_test_json = \
            ('{}' + os.sep + '{}').format(
                self.test_cases_directory,
                os.getenv('DECISION_MODEL_TEST_CHOOSE_FROM_JSON'))

        test_data = get_test_data(path_to_test_json=path_to_test_json, method='read')

        test_case = test_data.get("test_case", None)

        if test_case is None:
            raise ValueError('Test case can`t be None')

        variants = test_case.get("variants", None)

        if variants is None:
            raise ValueError('Variants can`t be None')

        expected_output = test_data.get("test_output", None)
        assert expected_output is not None

        expected_scores = expected_output.get('scores', None)
        assert expected_scores is not None

        expected_best = expected_output.get('best', None)
        assert expected_best is not None

        scores_seed = test_data.get('scores_seed', None)
        assert scores_seed is not None

        np.random.seed(scores_seed)
        decision = \
            dm.DecisionModel(model_name='test_choose_from_model')\
            .choose_from(variants=variants)

        assert isinstance(decision, d.Decision)
        assert_valid_decision(
            decision=decision, expected_variants=variants, expected_givens=None,
            expected_scores=expected_scores, expected_best=expected_best)

    def test_given(self):
        path_to_test_json = \
            ('{}' + os.sep + '{}').format(
                self.test_cases_directory,
                os.getenv('DECISION_MODEL_TEST_GIVEN_JSON'))

        test_data = get_test_data(path_to_test_json=path_to_test_json, method='read')

        test_case = test_data.get("test_case", None)

        if test_case is None:
            raise ValueError('Test case can`t be None')

        givens = test_case.get("givens", None)

        if givens is None:
            raise ValueError('givens can`t be None')

        expected_output = test_data.get("test_output", None)
        assert expected_output is not None

        expected_givens = expected_output.get('givens', None)
        assert expected_givens is not None

        decision_context = \
            dm.DecisionModel(model_name='test_choose_from_model').given(givens=givens)

        assert isinstance(decision_context, dc.DecisionContext)
        assert hasattr(decision_context, 'givens')
        assert isinstance(decision_context.givens, dict)
        assert decision_context.givens == expected_givens

    def test_no_model__score_and_sort(self):

        variants = [el for el in range(100)]

        decision_model = dm.DecisionModel(model_name='no-model')

        scores_for_variants = decision_model._score(variants=variants, givens={})

        sorted_with_scores = \
            [v for _, v in
             sorted(zip(scores_for_variants, variants), reverse=True)]

        assert variants == sorted_with_scores

    # set model name from constructor:
    # - test that regexp compliant model name passes regexp
    def test_good_model_name(self):
        good_model_names = \
            [None, 'a', '0', '0-', 'a1-', 'x23yz_', 'a01sd.', 'abc3-xy2z_as4d.'] + \
            [''.join('a' for _ in range(64))]

        for good_model_name in good_model_names:
            dm.DecisionModel(model_name=good_model_name)

    # - test that regexp non-compliant model name raises AssertionError
    def test_bad_model_name(self):
        bad_model_names = \
            ['', '-', '_', '.', '-a1', '.x2z', '_x2z'] + \
            [''.join(
                np.random.choice(TestDecisionModel.ALNUM_CHARS, 2).tolist() + [sc] +
                np.random.choice(TestDecisionModel.ALNUM_CHARS, 2).tolist())
             for sc in TestDecisionModel.BAD_SPECIAL_CHARACTERS] + \
            [''.join('a' for _ in range(65))]

        for bad_model_name in bad_model_names:
            with raises(AssertionError) as aerr:
                dm.DecisionModel(model_name=bad_model_name)

    def test_none_model_name_overwritten(self):
        path_to_test_json = \
            os.sep.join([
                self.test_cases_directory, os.getenv('DECISION_MODEL_TEST_MODEL_NAME_SET_TO_NONE_JSON')])
        test_data = get_test_data(path_to_test_json=path_to_test_json, method='read')

        model_url = \
            os.sep.join([self.predictors_fs_directory, test_data['test_case']['model_filename']])

        decision_model = dm.DecisionModel(model_name=None).load(model_url=model_url)
        assert decision_model.model_name is not None

        expected_output = test_data['test_output']['model_name']

        assert decision_model.model_name == expected_output

    def test_not_none_model_name_warns(self):
        path_to_test_json = \
            os.sep.join([
                self.test_cases_directory, os.getenv('DECISION_MODEL_TEST_MODEL_NAME_SET_TO_NOT_NONE_JSON')])
        test_data = get_test_data(path_to_test_json=path_to_test_json, method='read')

        model_url = \
            os.sep.join([self.predictors_fs_directory, test_data['test_case']['model_filename']])

        tested_model_name = test_data['test_case']['model_name']
        chooser = XGBChooser()
        chooser.load_model(model_url)

        with warnings.catch_warnings(record=True) as w:
            decision_model = dm.DecisionModel(model_name=tested_model_name).load(model_url=model_url)
            assert len(w) != 0
        assert decision_model.model_name is not None
        assert decision_model.model_name != chooser.model_name

        expected_output = test_data['test_output']['model_name']

        assert decision_model.model_name == expected_output

    def test_which_valid_list_variants(self):
        path_to_test_json = \
            os.sep.join([
                self.test_cases_directory, os.getenv('DECISION_MODEL_TEST_WHICH_JSON')])
        test_case_json = get_test_data(path_to_test_json)

        test_case = test_case_json.get('test_case', None)
        assert test_case is not None

        predictor_filename = test_case.get('model_filename', None)

        model_url = \
            ('{}' + os.sep + '{}').format(
                self.predictors_fs_directory, predictor_filename)

        assert model_url is not None

        decision_model = \
            dm.DecisionModel(model_name=None, track_url=self.track_url) \
            .load(model_url=model_url)

        variants = test_case.get('variants', None)
        assert variants is not None
        scores_seed = test_case_json.get('scores_seed', None)
        assert scores_seed is not None

        expected_output = test_case_json.get('test_output', None)
        assert expected_output is not None
        expected_best = expected_output.get('best', None)
        assert expected_best is not None

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success')
            np.random.seed(scores_seed)
            best, decision_id = decision_model.which(*variants)
            assert is_valid_ksuid(decision_id)

            np.random.seed(scores_seed)
            best, decision_id = decision_model.which(variants)
            assert is_valid_ksuid(decision_id)

        assert best == expected_best

    def test_which_valid_list_variants_no_model(self):
        path_to_test_json = \
            os.sep.join([
                self.test_cases_directory, os.getenv('DECISION_MODEL_TEST_WHICH_NO_MODEL_JSON')])
        test_case_json = get_test_data(path_to_test_json)

        test_case = test_case_json.get('test_case', None)
        assert test_case is not None

        decision_model = \
            dm.DecisionModel(model_name=None, track_url=self.track_url)

        variants = test_case.get('variants', None)
        assert variants is not None
        scores_seed = test_case_json.get('scores_seed', None)
        assert scores_seed is not None

        expected_output = test_case_json.get('test_output', None)
        assert expected_output is not None
        expected_best = expected_output.get('best', None)
        assert expected_best is not None

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success')
            np.random.seed(scores_seed)
            best, decision_id = decision_model.which(*variants)
            assert is_valid_ksuid(decision_id)

            np.random.seed(scores_seed)
            best, decision_id = decision_model.which(variants)
            assert is_valid_ksuid(decision_id)

        assert best == expected_best

    def test_which_valid_tuple_variants(self):
        path_to_test_json = \
            os.sep.join([
                self.test_cases_directory, os.getenv('DECISION_MODEL_TEST_WHICH_JSON')])
        test_case_json = get_test_data(path_to_test_json)

        test_case = test_case_json.get('test_case', None)
        assert test_case is not None

        predictor_filename = test_case.get('model_filename', None)

        model_url = \
            ('{}' + os.sep + '{}').format(
                self.predictors_fs_directory, predictor_filename)

        assert model_url is not None

        decision_model = \
            dm.DecisionModel(model_name=None, track_url=self.track_url) \
            .load(model_url=model_url)

        variants = test_case.get('variants', None)
        assert variants is not None
        variants = tuple(variants)
        scores_seed = test_case_json.get('scores_seed', None)
        assert scores_seed is not None

        expected_output = test_case_json.get('test_output', None)
        assert expected_output is not None
        expected_best = expected_output.get('best', None)
        assert expected_best is not None

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success')
            np.random.seed(scores_seed)
            best, decision_id = decision_model.which(*variants)
            assert is_valid_ksuid(decision_id)
            assert best == expected_best

            np.random.seed(scores_seed)
            best, decision_id = decision_model.which(variants)
            assert is_valid_ksuid(decision_id)
            assert best == expected_best

    def test_which_valid_ndarray_variants(self):
        path_to_test_json = \
            os.sep.join([
                self.test_cases_directory, os.getenv('DECISION_MODEL_TEST_WHICH_JSON')])
        test_case_json = get_test_data(path_to_test_json)

        test_case = test_case_json.get('test_case', None)
        assert test_case is not None

        predictor_filename = test_case.get('model_filename', None)

        model_url = \
            ('{}' + os.sep + '{}').format(
                self.predictors_fs_directory, predictor_filename)

        assert model_url is not None

        decision_model = \
            dm.DecisionModel(model_name=None, track_url=self.track_url) \
            .load(model_url=model_url)

        variants = test_case.get('variants', None)
        assert variants is not None
        variants = np.array(variants)
        scores_seed = test_case_json.get('scores_seed', None)
        assert scores_seed is not None

        expected_output = test_case_json.get('test_output', None)
        assert expected_output is not None
        expected_best = expected_output.get('best', None)
        assert expected_best is not None

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success')
            np.random.seed(scores_seed)
            best, decision_id = decision_model.which(*variants)
            assert is_valid_ksuid(decision_id)
            assert best == expected_best

            np.random.seed(scores_seed)
            best, decision_id = decision_model.which(variants)
            assert is_valid_ksuid(decision_id)
            assert best == expected_best

    def test_which_invalid_variants(self):
        path_to_test_json = \
            os.sep.join([
                self.test_cases_directory, os.getenv('DECISION_MODEL_TEST_WHICH_JSON')])
        test_case_json = get_test_data(path_to_test_json)

        test_case = test_case_json.get('test_case', None)
        assert test_case is not None

        predictor_filename = test_case.get('model_filename', None)

        model_url = \
            ('{}' + os.sep + '{}').format(
                self.predictors_fs_directory, predictor_filename)
        assert model_url is not None

        decision_model = \
            dm.DecisionModel(model_name=None, track_url=self.track_url) \
            .load(model_url=model_url)

        invalid_variants = ['a', 1, 1.123]
        for ivs in invalid_variants:
            with raises(AssertionError) as aerr:
                decision_model.which(*[ivs])

    def test_which_zero_length_variants(self):
        path_to_test_json = \
            os.sep.join([
                self.test_cases_directory, os.getenv('DECISION_MODEL_TEST_WHICH_JSON')])
        test_case_json = get_test_data(path_to_test_json)

        test_case = test_case_json.get('test_case', None)
        assert test_case is not None

        predictor_filename = test_case.get('model_filename', None)

        model_url = \
            ('{}' + os.sep + '{}').format(
                self.predictors_fs_directory, predictor_filename)
        assert model_url is not None

        decision_model = \
            dm.DecisionModel(model_name=None, track_url=self.track_url) \
            .load(model_url=model_url)

        invalid_variants = [[], np.array([])]
        for ivs in invalid_variants:
            with raises(ValueError) as verr:
                decision_model.which(*[ivs])

    def test_which_none_track_url(self):
        best_variant, decision_id = \
            dm.DecisionModel(model_name='dummy-model', track_url=None).which(1, 2, 3, 4, 5)
        assert best_variant == 1
        assert decision_id is None

    def test_choose_first_valid_variants_list(self):
        self._generic_desired_decision_model_method_call_no_model(
            test_data_filename=os.getenv('DECISION_MODEL_TEST_CHOOSE_FIRST_VALID_VARIANTS_JSON'),
            evaluated_method_name='choose_first')

    def test_choose_first_valid_variants_numpy(self):
        self._generic_desired_decision_model_method_call_no_model(
            test_data_filename=os.getenv('DECISION_MODEL_TEST_CHOOSE_FIRST_VALID_VARIANTS_JSON'),
            evaluated_method_name='choose_first', variants_input_type='numpy')

    def test_choose_first_valid_variants_tuple(self):
        self._generic_desired_decision_model_method_call_no_model(
            test_data_filename=os.getenv('DECISION_MODEL_TEST_CHOOSE_FIRST_VALID_VARIANTS_JSON'),
            evaluated_method_name='choose_first', variants_input_type='tuple')

    def test_choose_first_raises_for_string_variants(self):
        with raises(AssertionError) as aerr:
            dm.DecisionModel('dummy-model').choose_first(variants='abc')

    def test_choose_first_raises_for_numeric_variants(self):
        with raises(AssertionError) as aerr:
            dm.DecisionModel('dummy-model').choose_first(variants=123.123)

    def test_choose_first_raises_for_bool_variants(self):
        with raises(AssertionError) as aerr:
            dm.DecisionModel('dummy-model').choose_first(variants=True)

    def test_choose_first_raises_for_empty_variants(self):
        with raises(ValueError) as verr:
            dm.DecisionModel('dummy-model').choose_first(variants=[])

        with raises(ValueError) as verr:
            dm.DecisionModel('dummy-model').choose_first(variants=np.array([]))

        with raises(ValueError) as verr:
            dm.DecisionModel('dummy-model').choose_first(variants=tuple())

    def test_choose_first_raises_for_none_variants(self):
        with raises(AssertionError) as aerr:
            dm.DecisionModel('dummy-model').choose_first(variants=None)

    def test_first_valid_variants_list(self):
        self._generic_desired_decision_model_method_call_no_model(
            test_data_filename=os.getenv('DECISION_MODEL_TEST_FIRST_VALID_VARIANTS_JSON'),
            evaluated_method_name='first')

    def test_first_valid_variants_numpy(self):
        self._generic_desired_decision_model_method_call_no_model(
            test_data_filename=os.getenv('DECISION_MODEL_TEST_FIRST_VALID_VARIANTS_JSON'),
            evaluated_method_name='first', variants_input_type='numpy')

    def test_first_valid_variants_tuple(self):
        self._generic_desired_decision_model_method_call_no_model(
            test_data_filename=os.getenv('DECISION_MODEL_TEST_FIRST_VALID_VARIANTS_JSON'),
            evaluated_method_name='first', variants_input_type='tuple')

    def test_first_raises_for_string_variants(self):
        with raises(AssertionError) as aerr:
            dm.DecisionModel('dummy-model').first('abc')

    def test_first_raises_for_numeric_variants(self):
        with raises(AssertionError) as aerr:
            dm.DecisionModel('dummy-model').first(123.123)

    def test_first_raises_for_bool_variants(self):
        with raises(AssertionError) as aerr:
            dm.DecisionModel('dummy-model').first(True)

    def test_first_raises_for_empty_variants(self):
        with raises(ValueError) as verr:
            dm.DecisionModel('dummy-model').first(*[])

        with raises(ValueError) as verr:
            dm.DecisionModel('dummy-model').first(*np.array([]))

        with raises(ValueError) as verr:
            dm.DecisionModel('dummy-model').first(*tuple())

    def test_first_raises_for_none_variants(self):
        with raises(AssertionError) as aerr:
            dm.DecisionModel('dummy-model').first(None)

    def test_first_none_track_url(self):
        best_variant, decision_id = \
            dm.DecisionModel(model_name='dummy-model', track_url=None).first(1, 2, 3, 4, 5)
        assert best_variant == 1
        assert decision_id is None

    def test_choose_random_valid_variants_list(self):
        self._generic_desired_decision_model_method_call_no_model(
            test_data_filename=os.getenv('DECISION_MODEL_TEST_CHOOSE_RANDOM_VALID_VARIANTS_JSON'),
            evaluated_method_name='choose_random')

    def test_choose_random_valid_variants_numpy(self):
        self._generic_desired_decision_model_method_call_no_model(
            test_data_filename=os.getenv('DECISION_MODEL_TEST_CHOOSE_RANDOM_VALID_VARIANTS_JSON'),
            evaluated_method_name='choose_random', variants_input_type='numpy')

    def test_choose_random_valid_variants_tuple(self):
        self._generic_desired_decision_model_method_call_no_model(
            test_data_filename=os.getenv('DECISION_MODEL_TEST_CHOOSE_RANDOM_VALID_VARIANTS_JSON'),
            evaluated_method_name='choose_random', variants_input_type='tuple')

    def test_choose_random_raises_for_string_variants(self):
        with raises(AssertionError) as aerr:
            dm.DecisionModel('dummy-model').choose_random(variants='abc')

    def test_choose_random_raises_for_numeric_variants(self):
        with raises(AssertionError) as aerr:
            dm.DecisionModel('dummy-model').choose_random(variants=123.123)

    def test_choose_random_raises_for_bool_variants(self):
        with raises(AssertionError) as aerr:
            dm.DecisionModel('dummy-model').choose_random(variants=True)

    def test_choose_random_raises_for_empty_variants(self):
        with raises(ValueError) as verr:
            dm.DecisionModel('dummy-model').choose_random(variants=[])

        with raises(ValueError) as verr:
            dm.DecisionModel('dummy-model').choose_random(variants=np.array([]))

        with raises(ValueError) as verr:
            dm.DecisionModel('dummy-model').choose_random(variants=tuple())

    def test_choose_random_raises_for_none_variants(self):
        with raises(AssertionError) as aerr:
            dm.DecisionModel('dummy-model').choose_random(variants=None)

    def test_choose_random_orders_runners_up_randomly(self):
        variants = [1, 2, 3, 4, 5]
        decision_model = dm.DecisionModel('dummy-model', track_url=self.track_url)
        np.random.seed(1)
        decision = decision_model.choose_random(variants=variants)
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
            m.post(self.track_url, text='success', additional_matcher=custom_matcher)

            np.random.seed(int(tracks_runners_up_seed))
            best = decision.get()
            assert decision.chosen is True
            assert decision.tracked is True
            assert best == 1

    def test_random_valid_variants_list(self):
        self._generic_desired_decision_model_method_call_no_model(
            test_data_filename=os.getenv('DECISION_MODEL_TEST_RANDOM_VALID_VARIANTS_JSON'),
            evaluated_method_name='random')

    def test_random_valid_variants_numpy(self):
        self._generic_desired_decision_model_method_call_no_model(
            test_data_filename=os.getenv('DECISION_MODEL_TEST_RANDOM_VALID_VARIANTS_JSON'),
            evaluated_method_name='random', variants_input_type='numpy')

    def test_random_valid_variants_tuple(self):
        self._generic_desired_decision_model_method_call_no_model(
            test_data_filename=os.getenv('DECISION_MODEL_TEST_RANDOM_VALID_VARIANTS_JSON'),
            evaluated_method_name='random', variants_input_type='tuple')

    def test_random_raises_for_string_variants(self):
        with raises(AssertionError) as aerr:
            dm.DecisionModel('dummy-model').random('abc')

    def test_random_raises_for_numeric_variants(self):
        with raises(AssertionError) as aerr:
            dm.DecisionModel('dummy-model').random(123.123)

    def test_random_raises_for_bool_variants(self):
        with raises(AssertionError) as aerr:
            dm.DecisionModel('dummy-model').random(True)

    def test_random_raises_for_empty_variants(self):
        with raises(ValueError) as verr:
            dm.DecisionModel('dummy-model').random(*[])

        with raises(ValueError) as verr:
            dm.DecisionModel('dummy-model').random(*np.array([]))

        with raises(ValueError) as verr:
            dm.DecisionModel('dummy-model').random(*tuple())

    def test_random_raises_for_none_variants(self):
        with raises(AssertionError) as aerr:
            dm.DecisionModel('dummy-model').random(None)

    def test_random_none_track_url(self):
        best_variant, decision_id = \
            dm.DecisionModel(model_name='dummy-model', track_url=None).random(1, 2, 3, 4, 5)
        np.random.seed(1)
        assert best_variant == 3
        assert decision_id is None

    def test_add_reward_inf(self):
        # V6_DUMMY_MODEL_PATH
        model_url = os.getenv('DUMMY_MODEL_PATH', None)
        assert model_url is not None

        decision_model = \
            dm.DecisionModel(model_name=None, track_url=self.track_url)\
            .load(model_url=model_url)

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success')
            decision = decision_model.choose_from(list(range(10)))
            decision.get()

        reward = math.inf

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success')
            with raises(AssertionError) as aerr:
                decision_model.add_reward(reward=reward, decision_id=decision.id_)

        reward = -math.inf

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success')
            with raises(AssertionError) as aerr:
                decision_model.add_reward(reward=reward, decision_id=decision.id_)

    def test_add_reward_none(self):
        # V6_DUMMY_MODEL_PATH
        model_url = os.getenv('DUMMY_MODEL_PATH', None)
        assert model_url is not None

        decision_model = \
            dm.DecisionModel(model_name=None, track_url=self.track_url)\
            .load(model_url=model_url)

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success')
            decision = decision_model.choose_from(list(range(10)))
            decision.get()

        reward = None

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success')
            with raises(AssertionError) as aerr:
                decision_model.add_reward(reward=reward, decision_id=decision.id_)

        reward = np.nan

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success')
            with raises(AssertionError) as aerr:
                decision_model.add_reward(reward=reward, decision_id=decision.id_)

    def test_add_reward(self):
        model_url = os.getenv('DUMMY_MODEL_PATH', None)
        assert model_url is not None

        decision_model = \
            dm.DecisionModel(model_name=None, track_url=self.track_url)\
            .load(model_url=model_url)
        reward = 1.0

        expected_add_reward_body = {
            decision_model.tracker.TYPE_KEY: decision_model.tracker.REWARD_TYPE,
            decision_model.tracker.MODEL_KEY: decision_model.model_name,
            decision_model.tracker.REWARD_KEY: reward,
            decision_model.tracker.DECISION_ID_KEY: None}

        def grab_decision_id_matcher(request):
            request_dict = deepcopy(request.json())
            expected_add_reward_body[decision_model.tracker.DECISION_ID_KEY] = \
                request_dict[decision_model.tracker.MESSAGE_ID_KEY]

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success', additional_matcher=grab_decision_id_matcher)
            decision = decision_model.choose_from(list(range(10)))
            decision.get()

        expected_request_json = json.dumps(expected_add_reward_body, sort_keys=False)

        def custom_matcher(request):
            request_dict = deepcopy(request.json())
            del request_dict[decision_model.tracker.MESSAGE_ID_KEY]

            if json.dumps(request_dict, sort_keys=False) != expected_request_json:

                print('raw request body:')
                print(request.text)
                print('compared request string')
                print(json.dumps(request_dict, sort_keys=False))
                print('expected body:')
                print(expected_request_json)
                return None
            return True

        with rqm.Mocker() as m:
            m.post(self.track_url, text='success', additional_matcher=custom_matcher)
            decision_model.add_reward(reward=reward, decision_id=decision.id_)

    def test_add_rewards_raises_for_none_model_name(self):
        model = dm.DecisionModel(model_name=None, track_url=self.track_url)
        with rqm.Mocker() as m:
            m.post(self.track_url, text='success')
            with raises(AssertionError) as aerr:
                model.add_reward(1.0, decision_id=str(Ksuid))
