import coremltools as ct
import json
import numpy as np
import os
from pytest import fixture, raises
import sys
import time
from unittest import TestCase
import xgboost as xgb

sys.path.append(
    os.sep.join(str(os.path.abspath(__file__)).split(os.sep)[:-3]))

import improveai.decision as d
import improveai.decision_model as dm
import improveai.decision_tracker as dt
from improveai.utils.general_purpose_tools import read_jsonstring_from_file


class TestDecisionModel(TestCase):

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
            os.getenv('V6_DECISION_MODEL_TEST_CASES_DIRECTORY')

        self.predictors_fs_directory = \
            os.getenv('V6_DECISION_MODEL_PREDICTORS_DIR')

    def _get_test_data(
            self, path_to_test_json: str, method: str = 'readlines') -> object:

        loaded_jsonstring = read_jsonstring_from_file(
            path_to_file=path_to_test_json, method=method)

        loaded_json = json.loads(loaded_jsonstring)

        return loaded_json

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

        test_data = \
            self._get_test_data(
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
        if load_mode == 'sync':
            decision_model = dm.DecisionModel.load(model_url=model_url)
        elif load_mode == 'async':
            decision_model = dm.DecisionModel(model_name='dummy-model-0')
            decision_model.load_async(model_url=model_url)
            time.sleep(3)
        else:
            raise ValueError('Allowed values for `load_mode` are sync and async')

        # is returned object a decision model
        assert isinstance(decision_model, dm.DecisionModel)

        # is predictor of a desired type
        print('decision_model.chooser')
        print(decision_model.chooser)
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
            self._get_test_data(
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
                decision_model = dm.DecisionModel.load(model_url=model_url)
            elif load_mode == 'async':
                decision_model = dm.DecisionModel(model_name='dummy-model')
                decision_model.load_async(model_url=model_url)
                time.sleep(3)
            else:
                raise RuntimeError(
                    'Allowed values for `load_mode` are sync and async')

            # print(verr)
            assert False
            assert str(verr.value)

        # is returned object a decision model
        # assert isinstance(decision_model, DecisionModel)
        # assert decision_model.chooser is None

    def _generic_test_loaded_url_none_model(
            self, test_data_filename: str, test_case_key: str = 'test_case',
            model_url_key: str = 'model_url', load_mode: str = 'sync'):

        path_to_test_json = \
            ('{}' + os.sep + '{}').format(
                self.test_cases_directory, test_data_filename)

        test_data = \
            self._get_test_data(
                path_to_test_json=path_to_test_json, method='read')

        test_case = test_data.get(test_case_key, None)

        if test_case is None:
            raise ValueError('Test case can`t be None')

        model_url = test_case.get(model_url_key, None)

        if model_url is None:
            raise ValueError('Model filename can`t be None')

        # loading model
        with raises(ValueError) as verr:
            if load_mode == 'sync':
                decision_model = dm.DecisionModel.load(model_url=model_url)
            elif load_mode == 'async':
                decision_model = dm.DecisionModel(model_name='dummy-model')
                decision_model.load_async(model_url=model_url)
            else:
                raise RuntimeError(
                    'Allowed values for `load_mode` are sync and async')

            assert str(verr.value)

        # is returned object a decision model
        # assert isinstance(decision_model, DecisionModel)
        # assert decision_model.chooser is None

    # test model loading
    def test_load_model_sync_native_fs(self):

        self._generic_test_loaded_model(
            test_data_filename=os.getenv(
                'V6_DECISION_MODEL_TEST_LOAD_MODEL_FS_NATIVE_JSON'),
            expected_predictor_type=xgb.Booster)

    def test_load_model_sync_mlmodel_fs(self):
        self._generic_test_loaded_model(
            test_data_filename=os.getenv(
                'V6_DECISION_MODEL_TEST_LOAD_MODEL_FS_MLMODEL_JSON'),
            expected_predictor_type=ct.models.MLModel)

    def test_load_model_sync_native_fs_no_model(self):

        self._generic_test_loaded_fs_none_model(
            test_data_filename=os.getenv(
                'V6_DECISION_MODEL_TEST_LOAD_MODEL_FS_NATIVE_NO_MODEL_JSON'))

    def test_load_model_sync_mlmodel_fs_no_model(self):
        self._generic_test_loaded_fs_none_model(
            test_data_filename=os.getenv(
                'V6_DECISION_MODEL_TEST_LOAD_MODEL_FS_MLMODEL_NO_MODEL_JSON'))

    # TODO uncomment once v6 model url is available
    # def test_load_model_sync_native_url(self):
    #     pass

    def test_load_model_sync_native_url_no_model(self):
        self._generic_test_loaded_url_none_model(
            test_data_filename=os.getenv(
                'V6_DECISION_MODEL_TEST_LOAD_MODEL_URL_NO_MODEL_JSON'))

    # ASYNC tests
    # test model loading
    def test_load_model_async_native_fs(self):
        self._generic_test_loaded_model(
            test_data_filename=os.getenv(
                'V6_DECISION_MODEL_TEST_LOAD_MODEL_FS_NATIVE_JSON'),
            expected_predictor_type=xgb.Booster, load_mode='async')

    def test_load_model_async_mlmodel_fs(self):
        self._generic_test_loaded_model(
            test_data_filename=os.getenv(
                'V6_DECISION_MODEL_TEST_LOAD_MODEL_FS_MLMODEL_JSON'),
            expected_predictor_type=ct.models.MLModel, load_mode='async')

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

        test_data = \
            self._get_test_data(
                path_to_test_json=path_to_test_json, method='read')

        test_case = test_data.get(test_case_key, None)

        if test_case is None:
            raise ValueError('Test case can`t be None')

        variants = test_case.get(variants_key, None)

        if variants_key in empty_callable_kwargs:
            empty_callable_kwargs[variants_key] = variants

        if variants is None:
            raise ValueError('Variants can`t be None')

        context = test_case.get(givens_key, None)

        if givens_key in empty_callable_kwargs:
            empty_callable_kwargs[givens_key] = context

        if context is None:
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

        decision_model = dm.DecisionModel.load(model_url=model_url)

        np.random.seed(score_seed)
        calculated_scores = decision_model.score(**empty_callable_kwargs)

        if evaluated_method_name == 'score':

            np.testing.assert_array_equal(
                calculated_scores.tolist(), expected_output)
            return

        if scores_key in empty_callable_kwargs:
            empty_callable_kwargs[scores_key] = calculated_scores

        np.random.seed(score_seed)
        evaluated_callable = getattr(dm.DecisionModel, evaluated_method_name)
        calculated_output = evaluated_callable(**empty_callable_kwargs)

        if isinstance(expected_output, list):
            np.testing.assert_array_equal(
                expected_output, calculated_output)
        else:
            assert expected_output == calculated_output

    def _generic_desired_decision_model_method_call_no_model(
            self, test_data_filename: str, evaluated_method_name: str,
            empty_callable_kwargs: dict, test_case_key: str = 'test_case',
            test_output_key: str = 'test_output',
            variants_key: str = 'variants', givens_key: str = 'givens',
            predictor_filename_key: str = 'model_filename',
            scores_key: str = 'scores', scores_seed_key: str = 'scores_seed'):

        path_to_test_json = \
            ('{}' + os.sep + '{}').format(
                self.test_cases_directory, test_data_filename)

        test_data = \
            self._get_test_data(
                path_to_test_json=path_to_test_json, method='read')

        test_case = test_data.get(test_case_key, None)

        if test_case is None:
            raise ValueError('Test case can`t be None')

        variants = test_case.get(variants_key, None)

        if variants_key in empty_callable_kwargs:
            empty_callable_kwargs[variants_key] = variants

        if variants is None:
            raise ValueError('Variants can`t be None')

        context = test_case.get(givens_key, None)

        if givens_key in empty_callable_kwargs:
            empty_callable_kwargs[givens_key] = context

        if context is None:
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

        decision_model = dm.DecisionModel(model_name='dummy-model')

        np.random.seed(score_seed)
        calculated_scores = decision_model.score(**empty_callable_kwargs)

        if evaluated_method_name == 'score':

            np.testing.assert_array_equal(
                calculated_scores.tolist(), expected_output)
            return

        if scores_key in empty_callable_kwargs:
            empty_callable_kwargs[scores_key] = calculated_scores

        np.random.seed(score_seed)
        evaluated_callable = getattr(dm.DecisionModel, evaluated_method_name)
        calculated_output = evaluated_callable(**empty_callable_kwargs)

        if isinstance(expected_output, list):
            np.testing.assert_array_equal(
                expected_output, calculated_output)
        else:
            assert expected_output == calculated_output

    def test_score_no_model(self):
        self._generic_desired_decision_model_method_call_no_model(
            test_data_filename=os.getenv(
                'V6_DECISION_MODEL_TEST_SCORE_NATIVE_NO_MODEL_JSON'),
            evaluated_method_name='score',
            empty_callable_kwargs={'variants': None, 'givens': None})

    def test_score(self):
        self._generic_desired_decision_model_method_call(
            test_data_filename=os.getenv(
                'V6_DECISION_MODEL_TEST_SCORE_NATIVE_JSON'),
            evaluated_method_name='score',
            empty_callable_kwargs={'variants': None, 'givens': None})

    def test_top_scoring_variant_no_model(self):
        self._generic_desired_decision_model_method_call_no_model(
            test_data_filename=os.getenv(
                'V6_DECISION_MODEL_TEST_TOP_SCORING_VARIANT_NATIVE_NO_MODEL_JSON'),
            evaluated_method_name='top_scoring_variant',
            empty_callable_kwargs={
                'variants': None, 'givens': None, 'scores': None})

    def test_top_scoring_variant(self):
        self._generic_desired_decision_model_method_call(
            test_data_filename=os.getenv(
                'V6_DECISION_MODEL_TEST_TOP_SCORING_VARIANT_NATIVE_JSON'),
            evaluated_method_name='top_scoring_variant',
            empty_callable_kwargs={
                'variants': None, 'givens': None, 'scores': None})

    def test_ranked_no_model(self):
        self._generic_desired_decision_model_method_call_no_model(
            test_data_filename=os.getenv(
                'V6_DECISION_MODEL_TEST_RANK_NATIVE_NO_MODEL_JSON'),
            evaluated_method_name='rank',
            empty_callable_kwargs={
                'variants': None, 'givens': None, 'scores': None})

    def test_ranked(self):
        self._generic_desired_decision_model_method_call(
            test_data_filename=os.getenv(
                'V6_DECISION_MODEL_TEST_RANK_NATIVE_JSON'),
            evaluated_method_name='rank',
            empty_callable_kwargs={
                'variants': None, 'givens': None, 'scores': None})

    def test_generate_descending_gaussians(self):
        path_to_test_json = \
            ('{}' + os.sep + '{}').format(
                self.test_cases_directory,
                os.getenv('V6_DECISION_MODEL_TEST_GENERATE_DESCENDING_GAUSSIANS_JSON'))

        test_data = \
            self._get_test_data(
                path_to_test_json=path_to_test_json, method='read')

        test_case = test_data.get("test_case", None)

        if test_case is None:
            raise ValueError('Test case can`t be None')

        variants_count = test_case.get("count", None)

        if variants_count is None:
            raise ValueError('Variants count can`t be None')

        score_seed = test_data.get("scores_seed", None)

        expected_output = test_data.get("test_output", None)

        if expected_output is None:
            raise ValueError('`test_output` can`t be None')

        np.random.seed(score_seed)
        calculated_gaussians = \
            dm.DecisionModel.generate_descending_gaussians(count=variants_count)

        np.testing.assert_array_equal(calculated_gaussians, expected_output)

    def test_choose_from(self):

        path_to_test_json = \
            ('{}' + os.sep + '{}').format(
                self.test_cases_directory,
                os.getenv('V6_DECISION_MODEL_TEST_CHOOSE_FROM_JSON'))

        test_data = \
            self._get_test_data(
                path_to_test_json=path_to_test_json, method='read')

        test_case = test_data.get("test_case", None)

        if test_case is None:
            raise ValueError('Test case can`t be None')

        variants = test_case.get("variants", None)

        if variants is None:
            raise ValueError('Variants can`t be None')

        expected_output = test_data.get("test_output", None)

        if expected_output is None:
            raise ValueError('`test_output` can`t be None')

        decision = \
            dm.DecisionModel(model_name='test_choose_from_model')\
            .choose_from(variants=variants)

        assert isinstance(decision, d.Decision)
        assert hasattr(decision, 'variants')
        np.testing.assert_array_equal(decision.variants, expected_output)
        assert decision.givens is None

    def test_given(self):
        path_to_test_json = \
            ('{}' + os.sep + '{}').format(
                self.test_cases_directory,
                os.getenv('V6_DECISION_MODEL_TEST_GIVEN_JSON'))

        test_data = \
            self._get_test_data(
                path_to_test_json=path_to_test_json, method='read')

        test_case = test_data.get("test_case", None)

        if test_case is None:
            raise ValueError('Test case can`t be None')

        givens = test_case.get("givens", None)

        if givens is None:
            raise ValueError('givens can`t be None')

        expected_output = test_data.get("test_output", None)

        if expected_output is None:
            raise ValueError('`test_output` can`t be None')

        decision = \
            dm.DecisionModel(model_name='test_choose_from_model') \
            .given(givens=givens)

        assert isinstance(decision, d.Decision)
        assert hasattr(decision, 'givens')
        assert isinstance(decision.givens, dict)
        assert decision.givens == expected_output
        assert decision.variants == [None]

    def test_set_tracker(self):
        tracker = dt.DecisionTracker(track_url='dummy_tracker_url')
        decision_model = \
            dm.DecisionModel(model_name='test_choose_from_model')

        assert decision_model.tracker is None

        decision_model.set_tracker(tracker=tracker)

        assert decision_model.tracker is not None
        assert decision_model.tracker == tracker

    def test_set_tracker_raises(self):
        decision_model = \
            dm.DecisionModel(model_name='test_choose_from_model')

        assert decision_model.tracker is None

        with raises(TypeError) as terr:

            decision_model.set_tracker(tracker=None)

            assert str(terr.value)

    def test_no_model_score_and_sort(self):

        variants = [el for el in range(100)]

        decision_model = dm.DecisionModel(model_name='no-model')

        scores_for_variants = decision_model.score(variants=variants, givens={})

        sorted_with_scores = \
            [v for _, v in
             sorted(zip(scores_for_variants, variants), reverse=True)]

        assert variants == sorted_with_scores
