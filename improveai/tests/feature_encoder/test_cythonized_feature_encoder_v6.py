from copy import deepcopy
import json
import numpy as np
import os
from pprint import pprint
from pytest import fixture, raises
import sys
from unittest import TestCase
import xgboost as xgb

sys.path.append(
    os.sep.join(str(os.path.abspath(__file__)).split(os.sep)[:-3]))

from improveai.cythonized_feature_encoding import cfe

FeatureEncoder = cfe.FeatureEncoder
sprinkle = cfe.sprinkle
shrink = cfe.shrink
reverse_sprinkle = cfe.reverse_sprinkle
_get_previous_value = cfe._get_previous_value

# add_noise = cfe.add_noise


import improveai.settings as improve_settings
from improveai.utils.general_purpose_tools import read_jsonstring_from_file
from improveai.tests.test_utils import convert_values_to_float32, assert_dicts_identical


class TestEncoder(TestCase):

    @property
    def feature_encoder(self) -> FeatureEncoder:
        return self._feature_encoder

    @feature_encoder.setter
    def feature_encoder(self, value: FeatureEncoder):
        self._feature_encoder = value

    @property
    def encoder_seed(self) -> int:
        return self._encoder_seed

    @encoder_seed.setter
    def encoder_seed(self, value: int):
        self._encoder_seed = value

    @property
    def v6_test_suite_data_directory(self) -> str:
        return self._v6_test_data_directory

    @v6_test_suite_data_directory.setter
    def v6_test_suite_data_directory(self, value):
        self._v6_test_data_directory = value

    @property
    def noise_seed(self):
        return self._noise_seed

    @noise_seed.setter
    def noise_seed(self, value):
        self._noise_seed = value

    @property
    def noise(self):
        return self._noise

    @noise.setter
    def noise(self, value):
        self._noise = value

    @property
    def feature_names(self):
        return self._feature_names

    @feature_names.setter
    def feature_names(self, value):
        self._feature_names = value

    @fixture(autouse=True)
    def prepare_artifacts(self):
        # self.encoder_seed = int(os.getenv("FEATURE_ENCODER_MODEL_SEED"))
        # self.noise_seed = int(os.getenv("FEATURE_ENCODER_NOISE_SEED"))

        self.v6_test_suite_data_directory = \
            os.getenv("FEATURE_ENCODER_TEST_SUITE_JSONS_DIR")

        self.v6_test_python_specific_data_directory = \
            os.getenv("FEATURE_ENCODER_TEST_PYTHON_SPECIFIC_JSONS_DIR")
        # self.feature_encoder = FeatureEncoder(model_seed=self.encoder_seed)

        # np.random.seed(self.noise_seed)
        # self.noise = np.random.rand()

        self._set_feature_names()

    def _set_feature_names(self):
        b = xgb.Booster()
        b.load_model(os.getenv("DUMMY_MODEL_PATH"))

        user_defined_metadata = json.loads(b.attr('user_defined_metadata'))[
            'json']
        self.feature_names = user_defined_metadata['feature_names']

    def _get_sprinkled_value_and_noise(self):
        x = 1.0
        noise = 0.1
        small_noise = shrink(noise)

        sprinkled_x = sprinkle(x, small_noise=small_noise)
        return x, sprinkled_x, small_noise

    def _get_test_data(
            self, path_to_test_json: str, method: str = 'readlines') -> object:

        loaded_jsonstring = read_jsonstring_from_file(
                    path_to_file=path_to_test_json, method=method)

        loaded_json = json.loads(loaded_jsonstring)

        return loaded_json

    def _set_model_properties_from_test_case(self, test_case: dict):
        # set model_seed
        self.encoder_seed = test_case.get("model_seed", None)

        if self.encoder_seed is None:
            raise ValueError("model_seed is missing from a test case")

        self.noise = test_case.get("noise", None)

        if self.noise is None:
            raise ValueError("noise is missing from a test case")

        self.feature_encoder = FeatureEncoder(model_seed=self.encoder_seed)

    def _generic_test_encode_record_from_json_data(
            self, test_case_filename: str, input_data_key: str = 'test_case',
            variant_key: str = 'variant', givens_key: str = 'givens',
            expected_output_data_key: str = 'test_output'):

        test_case_path = os.sep.join(
            [self.v6_test_suite_data_directory, test_case_filename])

        test_case = self._get_test_data(path_to_test_json=test_case_path)

        self._set_model_properties_from_test_case(test_case=test_case)

        test_input = test_case.get(input_data_key, None)

        if test_input is None:
            raise ValueError('Test input is empty')

        variant_input = test_input.get(variant_key, None)

        if variant_input is None:
            raise ValueError('Test input for variant is empty')

        given_input = test_input.get(givens_key, None)

        expected_output = test_case.get(expected_output_data_key, None)

        if expected_output is None:
            raise ValueError('Expected output is empty')

        tested_record_output_float64 = \
            self.feature_encoder.encode_variant(
                variant=variant_input, noise=self.noise,
                into=self.feature_encoder.encode_givens(
                    givens=deepcopy(given_input), noise=self.noise))

        tested_record_output_float32 = \
            convert_values_to_float32(tested_record_output_float64)
        expected_output_float32 = convert_values_to_float32(expected_output)

        assert_dicts_identical(
            expected=expected_output_float32, calculated=tested_record_output_float32)

        # for expected_key in expected_output_float32.keys():
        #     expected_value = expected_output_float32[expected_key]
        #     calculated_value = tested_record_output_float32[expected_key]
        #     assert expected_value == calculated_value

        assert expected_output_float32 == tested_record_output_float32

    def _generic_test_encode_record_for_same_output_from_json_data(
            self, first_test_case_filename: str, second_test_case_filename: str,
            provided_first_test_case: dict = None,
            provided_second_test_case: dict = None,
            input_data_key: str = 'test_case', variant_key: str = 'variant',
            givens_key: str = 'givens'):

        first_test_case_path = os.sep.join(
            [self.v6_test_suite_data_directory, first_test_case_filename])

        if not provided_first_test_case:
            first_test_case = \
                self._get_test_data(path_to_test_json=first_test_case_path)
        else:
            first_test_case = provided_first_test_case

        self._set_model_properties_from_test_case(test_case=first_test_case)

        first_test_input = first_test_case.get(input_data_key, None)

        if first_test_input is None:
            raise ValueError(
                "First test input for is empty")

        first_variant_input = first_test_input.get(variant_key, None)
        first_given_input = first_test_input.get(givens_key, None)

        pprint(first_variant_input)

        if first_variant_input is None:
            raise ValueError(
                "First test variant input for is empty")

        first_output_float64 = \
            self.feature_encoder.encode_variant(
                variant=first_variant_input, noise=self.noise,
                into=self.feature_encoder.encode_givens(
                    givens=deepcopy(first_given_input), noise=self.noise))

        first_output_float32 = convert_values_to_float32(first_output_float64)

        second_test_case_path = os.sep.join(
            [self.v6_test_suite_data_directory, second_test_case_filename])

        if not provided_second_test_case:
            second_test_case = \
                self._get_test_data(path_to_test_json=second_test_case_path)
        else:
            second_test_case = provided_second_test_case

        self._set_model_properties_from_test_case(test_case=second_test_case)

        second_test_input = second_test_case.get(input_data_key, None)

        if second_test_input is None:
            raise ValueError(
                "Second test input for is empty")

        second_variant_input = second_test_input.get(variant_key, None)
        second_given_input = second_test_input.get(givens_key, None)

        print(second_variant_input)

        if second_variant_input is None:
            raise ValueError(
                "Second test input for is empty")

        second_output_float64 = \
            self.feature_encoder.encode_variant(
                variant=second_variant_input, noise=self.noise,
                into=self.feature_encoder.encode_givens(
                    givens=deepcopy(second_given_input), noise=self.noise))

        second_output_float32 = convert_values_to_float32(second_output_float64)

        assert first_output_float32 == second_output_float32

    def _generic_test_encode_record_for_different_output_from_json_data(
            self, first_test_case_filename: str, second_test_case_filename: str,
            provided_first_test_case: dict = None,
            provided_second_test_case: dict = None,
            input_data_key: str = 'test_case', variant_key: str = 'variant',
            givens_key: str = 'givens'):

        first_test_case_path = os.sep.join(
            [self.v6_test_suite_data_directory, first_test_case_filename])

        if not provided_first_test_case:
            first_test_case = \
                self._get_test_data(path_to_test_json=first_test_case_path)
        else:
            first_test_case = provided_first_test_case

        self._set_model_properties_from_test_case(test_case=first_test_case)

        first_test_input = first_test_case.get(input_data_key, None)

        if first_test_input is None:
            raise ValueError(
                "First test input for is empty")

        first_variant_input = first_test_input.get(variant_key, None)
        first_given_input = first_test_input.get(givens_key, None)

        pprint(first_variant_input)

        if first_variant_input is None:
            raise ValueError(
                "First test variant input for is empty")

        first_output_float_64 = \
            self.feature_encoder.encode_variant(
                variant=first_variant_input, noise=self.noise,
                into=self.feature_encoder.encode_givens(
                    givens=deepcopy(first_given_input), noise=self.noise))

        first_output_float_32 = convert_values_to_float32(first_output_float_64)

        second_test_case_path = os.sep.join(
            [self.v6_test_suite_data_directory, second_test_case_filename])

        if not provided_second_test_case:
            second_test_case = \
                self._get_test_data(path_to_test_json=second_test_case_path)
        else:
            second_test_case = provided_second_test_case

        self._set_model_properties_from_test_case(test_case=second_test_case)

        second_test_input = second_test_case.get(input_data_key, None)

        if second_test_input is None:
            raise ValueError(
                "Second test input for is empty")

        second_variant_input = second_test_input.get(variant_key, None)
        second_given_input = second_test_input.get(givens_key, None)

        print(second_variant_input)

        if second_variant_input is None:
            raise ValueError(
                "Second test input for is empty")

        second_output_float64 = \
            self.feature_encoder.encode_variant(
                variant=second_variant_input, noise=self.noise,
                into=self.feature_encoder.encode_givens(
                    givens=deepcopy(second_given_input), noise=self.noise))

        second_output_float32 = \
            convert_values_to_float32(second_output_float64)

        pprint('### outputs ###')
        pprint(first_output_float_32)
        pprint(second_output_float32)

        assert first_output_float_32 != second_output_float32

    def _generic_test_external_collisions(
            self, test_case_filename: str, input_data_key: str = 'test_case',
            expected_output_data_key: str = 'test_output',
            variant_key: str = 'variant', givens_key: str = 'givens',
            data_read_method: str = 'read'):

        test_case_path = os.sep.join(
            [self.v6_test_suite_data_directory, test_case_filename])

        test_case = \
            self._get_test_data(
                path_to_test_json=test_case_path, method=data_read_method)

        self._set_model_properties_from_test_case(test_case=test_case)

        test_case_input = test_case.get(input_data_key, None)

        if test_case_input is None:
            raise ValueError("Test jsonlines are empty")

        variant = test_case_input.get(variant_key, None)
        givens = test_case_input.get(givens_key, None)

        if variant is None or givens is None:
            raise ValueError("variant or givens are empty")

        expected_output = test_case.get(expected_output_data_key, None)

        if expected_output is None:
            raise ValueError("Expected output is empty")

        encoded_variant = \
            self.feature_encoder.encode_variant(variant=variant, noise=self.noise)
        encoded_givens = \
            self.feature_encoder.encode_givens(givens=givens, noise=self.noise)

        common_keys = \
            set(encoded_variant.keys()).intersection(set(encoded_givens.keys()))

        assert len(common_keys) > 0

        fully_encoded_variant_float64 = \
            self.feature_encoder.encode_variant(
                variant=variant, noise=self.noise,
                into=self.feature_encoder.encode_givens(
                    givens=deepcopy(givens), noise=self.noise))

        fully_encoded_variant_float32 = \
            convert_values_to_float32(fully_encoded_variant_float64)

        expected_output_float32 = convert_values_to_float32(expected_output)

        assert_dicts_identical(
            expected=expected_output_float32, calculated=fully_encoded_variant_float32)

        # np.testing.assert_array_equal(
        #     expected_output, fully_encoded_variant_float32)

        single_common_key = list(common_keys)[0]

        assert single_common_key in encoded_variant.keys() \
               and single_common_key in encoded_givens.keys() \
               and single_common_key in fully_encoded_variant_float32.keys()

        # TODO this assertion no longer makes sense
        # assert encoded_variant.get(single_common_key, None) + \
        #        encoded_givens.get(single_common_key, None) == \
        #        fully_encoded_variant.get(single_common_key, None)

    def _generic_test_internal_collisions(
            self, test_case_filename: str, input_data_key: str = 'test_case',
            expected_output_data_key: str = 'test_output',
            variant_key: str = 'variant', givens_key: str = 'givens',
            data_read_method: str = 'read'):

        test_case_path = os.sep.join(
            [self.v6_test_suite_data_directory, test_case_filename])

        test_case = \
            self._get_test_data(
                path_to_test_json=test_case_path, method=data_read_method)

        self._set_model_properties_from_test_case(test_case=test_case)

        test_case_input = test_case.get(input_data_key, None)

        if test_case_input is None:
            raise ValueError("Test jsonlines are empty")

        variant = test_case_input.get(variant_key, None)
        givens = test_case_input.get(givens_key, None)

        if variant is None or givens is None:
            raise ValueError("variant or givens are empty")

        expected_output = test_case.get(expected_output_data_key, None)

        if expected_output is None:
            raise ValueError("Expected output is empty")

        fully_encoded_variant_float64 = \
            self.feature_encoder.encode_variant(
                variant=variant, noise=self.noise,
                into=self.feature_encoder.encode_givens(
                    givens=deepcopy(givens), noise=self.noise))

        fully_encoded_variant_float32 = \
            convert_values_to_float32(fully_encoded_variant_float64)

        expected_output_float32 = convert_values_to_float32(expected_output)

        assert_dicts_identical(
            expected=expected_output_float32, calculated=fully_encoded_variant_float32)

        # for expected_key in expected_output_float32.keys():
        #     expected_value = expected_output_float32[expected_key]
        #     calculated_value = fully_encoded_variant_float32[expected_key]
        #     assert expected_value == calculated_value

        # assert fully_encoded_variant_float32 == expected_output

    # test all None-like types (None, [], {}, np.NaN)
    def test_none_variant(self):

        variant_input = None
        given_input = None

        test_case_path = os.sep.join(
            [self.v6_test_suite_data_directory,
             os.getenv("FEATURE_ENCODER_TEST_NONE_JSON")])

        test_case = \
            self._get_test_data(path_to_test_json=test_case_path)

        self._set_model_properties_from_test_case(test_case=test_case)

        expected_output = test_case.get('test_output', None)

        if expected_output is None:
            raise ValueError("Expected output empty for `test_none_variant`")

        tested_output_float64 = \
            self.feature_encoder.encode_variant(
                variant=variant_input, noise=self.noise,
                into=self.feature_encoder.encode_givens(
                    givens=deepcopy(given_input), noise=self.noise))

        tested_output_float32 = convert_values_to_float32(tested_output_float64)
        expected_output_float32 = convert_values_to_float32(expected_output)

        assert_dicts_identical(
            expected=expected_output_float32, calculated=tested_output_float32)

        # assert expected_output == tested_output_float32

    def _test_encode_feature_vector(
            self, test_case_envvar, variant_key: str = 'variant',
            givens_key: str = 'givens',
            extra_features_key: str = 'extra_features',
            feature_names_key: str = 'feature_names',
            test_input_key: str = 'test_case',
            test_output_key: str = 'test_output', force_numpy: bool = False):

        test_case_filename = \
            os.getenv(test_case_envvar, None)

        if not test_case_filename:
            raise ValueError(
                'No envvar under key: {}'.format(test_case_envvar))

        test_case_path = os.sep.join(
            [self.v6_test_suite_data_directory, test_case_filename])

        test_case = self._get_test_data(path_to_test_json=test_case_path)

        self._set_model_properties_from_test_case(test_case=test_case)

        test_input = test_case.get(test_input_key, None)

        if not test_input:
            raise ValueError('Test input is None')

        test_variant = test_input.get(variant_key, None)

        test_givens = test_input.get(givens_key, None)

        test_extra_features = test_input.get(extra_features_key, None)

        test_feature_names = test_input.get(feature_names_key, None)

        if not test_feature_names:
            raise ValueError('Test feature names are missing')

        expected_output = np.array(test_case.get(test_output_key, None))

        tested_into_float64 = np.full(len(test_feature_names), np.nan)

        if force_numpy:
            orig_use_cython_backend = improve_settings.USE_CYTHON_BACKEND
            improve_settings.USE_CYTHON_BACKEND = False

        self.feature_encoder.encode_feature_vector(
            variant=test_variant, givens=test_givens,
            extra_features=test_extra_features,
            feature_names=test_feature_names, noise=self.noise,
            into=tested_into_float64)

        if force_numpy:
            improve_settings.USE_CYTHON_BACKEND = orig_use_cython_backend

        print('tested_into')

        print(tested_into_float64.dtype)
        print('expected_output')
        print(expected_output.dtype)
        print('diff')
        print(tested_into_float64 - expected_output)

        tested_into_float32 = convert_values_to_float32(tested_into_float64)

        print('tested_into_float32')
        print(tested_into_float32)

        print('expected_output')
        print(expected_output)

        expected_output_float32 = np.array(expected_output).astype(np.float32)
        np.testing.assert_array_equal(expected_output_float32, tested_into_float32)

    # def test_encode_feature_vector_with_numpy(
    #         self, variant_key: str = 'variant', givens_key: str = 'givens',
    #         extra_features_key: str = 'extra_features',
    #         feature_names_key: str = 'feature_names',
    #         test_input_key: str = 'test_case',
    #         test_output_key: str = 'test_output'):
    #
    #     test_case_filename = \
    #         os.getenv('FEATURE_ENCODER_TEST_ENCODE_FEATURE_VECTOR_JSON', None)
    #
    #     if not test_case_filename:
    #         raise ValueError(
    #             'No envvar under key: FEATURE_ENCODER_TEST_ENCODE_FEATURE_VECTOR_JSON')
    #
    #     test_case_path = os.sep.join(
    #         [self.v6_test_suite_data_directory, test_case_filename])
    #
    #     test_case = self._get_test_data(path_to_test_json=test_case_path)
    #
    #     # TODO finish up
    #
    #     self._set_model_properties_from_test_case(test_case=test_case)
    #
    #     test_input = test_case.get(test_input_key, None)
    #
    #     if not test_input:
    #         raise ValueError('Test input is None')
    #
    #     test_variant = test_input.get(variant_key, None)
    #
    #     if not test_variant:
    #         raise ValueError('Test variant is missing')
    #
    #     test_givens = test_input.get(givens_key, None)
    #
    #     if not test_givens:
    #         raise ValueError('Test givens is missing')
    #
    #     test_extra_features = test_input.get(extra_features_key, None)
    #
    #     if not test_extra_features:
    #         raise ValueError('Test extra features are missing')
    #
    #     test_feature_names = test_input.get(feature_names_key, None)
    #
    #     if not test_feature_names:
    #         raise ValueError('Test feature names are missing')
    #
    #     expected_output = test_case.get(test_output_key, None)
    #
    #     tested_into_float64 = np.full(len(test_feature_names), np.nan)
    #
    #     # done for 100% coverage
    #     orig_use_cython_backend = improve_settings.USE_CYTHON_BACKEND
    #     improve_settings.USE_CYTHON_BACKEND = False
    #
    #     self.feature_encoder.encode_feature_vector(
    #         variant=test_variant, givens=test_givens,
    #         extra_features=test_extra_features,
    #         feature_names=test_feature_names, noise=self.noise,
    #         into=tested_into_float64)
    #
    #     improve_settings.USE_CYTHON_BACKEND = orig_use_cython_backend
    #
    #     tested_into_float32 = convert_values_to_float32(tested_into_float64)
    #
    #     np.testing.assert_array_equal(expected_output, tested_into_float32)

    def test_empty_list(self):

        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_EMPTY_LIST_JSON"))

    def test_empty_dict(self):

        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_EMPTY_DICT_JSON"))

    def test_dict_with_null_value(self):

        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_PRIMITIVE_DICT_NONE_JSON"))

    def test_npnan(self):

        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_NAN_JSON"))

    # Test all primitive types: "string", true, false, 0, 0.0, 1, 1.0, -1, -1.0
    def test_true(self):

        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_BOOL_TRUE_JSON"))

    def test_false(self):

        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_BOOL_FALSE_JSON"))

    def test_string(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_STRING_JSON"))

    def test_int_0(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_INT_0_JSON"))

    def test_float_0(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_FLOAT_0_JSON"))

    def test_int_1(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_INT_1_JSON"))

    def test_int64_small(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_INT64_SMALL_JSON"))

    def test_int64_big(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_INT64_BIG_JSON"))

    def test_float_1(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_FLOAT_1_JSON"))

    def test_int_m1(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_INT_M1_JSON"))

    def test_float_m1(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_FLOAT_M1_JSON"))

    def test_big_float(self):

        test_case_path = os.sep.join(
            [self.v6_test_suite_data_directory,
             os.getenv("FEATURE_ENCODER_TEST_BIG_FLOAT_JSON")])

        test_case = \
            self._get_test_data(path_to_test_json=test_case_path)

        self._set_model_properties_from_test_case(test_case=test_case)

        test_input = test_case.get("test_case", None)

        if test_input is None:
            raise ValueError(
                "Input data for `test_big_float_variant` can`t be empty")

        variant_input = test_input.get('variant', None)
        given_input = test_input.get('given', None)

        if variant_input is None:
            raise ValueError(
                "Input variant data for `test_big_float_variant` can`t be empty")

        expected_output = test_case.get("test_output", None)

        if expected_output is None:
            raise ValueError(
                "Expected output for `test_big_float_variant` can`t be empty")

        expected_output = \
            dict((key, float(val)) for key, val in expected_output.items())

        tested_output_float64 = \
            self.feature_encoder.encode_variant(
                variant=variant_input, noise=self.noise,
                into=self.feature_encoder.encode_givens(
                    givens=deepcopy(given_input), noise=self.noise))

        tested_output_float32 = convert_values_to_float32(tested_output_float64)

        assert expected_output == tested_output_float32

    def test_small_float(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_SMALL_FLOAT_JSON"))

    def test_special_characters_string(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_SPECIAL_CHARACTERS_STRING_JSON"))

    def test_special_characters_in_key_string(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_SPECIAL_CHARACTERS_IN_KEY_STRING_JSON"))

    def test_unicode_emoji_01(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_UNICODE_EMOJI_01_JSON"))

    def test_unicode_emoji_02(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_UNICODE_EMOJI_02_JSON"))

    def test_unicode_emoji_in_key(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_UNICODE_EMOJI_IN_KEY_JSON"))

    def test_unicode_string_01(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_UNICODE_STRING_01_JSON"))

    def test_unicode_string_02(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_UNICODE_STRING_02_JSON"))

    def test_unicode_string_with_u0000(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_UNICODE_STRING_WITH_U0000_JSON"))

    def test_unicode_u0000(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_UNICODE_U0000_JSON"))

    def test_unicode_zero_length_string(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_ZERO_LENGTH_STRING_JSON"))

    def test_newline_tab_return_symbols_string(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_NEWLINE_TAB_RETURN_SYMBOLS_STRING_JSON"))

    def test_noise_0_with_float(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_NOISE_0_WITH_FLOAT_JSON"))

    def test_noise_0_with_int(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_NOISE_0_WITH_INT_JSON"))

    def test_noise_0_with_primitive_dict_float(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_NOISE_0_WITH_PRIMITIVE_DICT_FLOAT_JSON"))

    def test_noise_0_with_primitive_dict_int(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_NOISE_0_WITH_PRIMITIVE_DICT_INT_JSON"))

    def test_noise_0_with_string(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_NOISE_0_WITH_STRING_JSON"))

    def test_noise_1_with_string(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_NOISE_1_WITH_STRING_JSON"))

    def test_noise_2_128(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_NOISE_2_128_JSON"))

    def test_noise_3_128(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_NOISE_3_128_JSON"))

    def test_noise_2_256(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_NOISE_2_256_JSON"))

    def test_noise_3_256(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_NOISE_3_256_JSON"))

    def test_big_int32_seed(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_SEED_BIG_INT32_JSON"))

    def test_leading_zeros_in_feature_names_01(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_LEADING_ZEROS_IN_FEATURE_NAME_01"))

    def test_leading_zeros_in_feature_names_02(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_LEADING_ZEROS_IN_FEATURE_NAME_02"))

    def test_sprinkle_equals_zero(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_SPRINKLED_EQUALS_ZERO_JSON"))

    def test_encode_feature_vector(self):
        self._test_encode_feature_vector(
            test_case_envvar='FEATURE_ENCODER_TEST_ENCODE_FEATURE_VECTOR_JSON')

    def test_encode_feature_vector_numpy(self):
        self._test_encode_feature_vector(
            test_case_envvar='FEATURE_ENCODER_TEST_ENCODE_FEATURE_VECTOR_JSON',
            force_numpy=True)

    def test_encode_feature_vector_variant_nan(self):
        self._test_encode_feature_vector(
            test_case_envvar='FEATURE_ENCODER_TEST_ENCODE_FEATURE_VECTOR_VARIANT_NAN_JSON')


    def test_encode_feature_vector_variant_nan_givens_none_extra_features(self):
        self._test_encode_feature_vector(
            test_case_envvar='FEATURE_ENCODER_TEST_ENCODE_FEATURE_VECTOR_VARIANT_NAN_GIVENS_NONE_EXTRA_FEATURES_JSON')

    def test_encode_feature_vector_variant_none(self):
        self._test_encode_feature_vector(
            test_case_envvar='FEATURE_ENCODER_TEST_ENCODE_FEATURE_VECTOR_VARIANT_NONE_JSON')


    def test_encode_feature_vector_variant_none_givens_none_extra_features(self):
        self._test_encode_feature_vector(
            test_case_envvar='FEATURE_ENCODER_TEST_ENCODE_FEATURE_VECTOR_VARIANT_NONE_GIVENS_NONE_EXTRA_FEATURES_JSON')

    def test_noise_out_of_bounds_raises(self):
        fe = FeatureEncoder(model_seed=0)

        with raises(ValueError) as high_noise_variant_error:
            fe.encode_variant({}, noise=99)
            assert str(high_noise_variant_error.value)

        with raises(ValueError) as low_noise_variant_error:
            fe.encode_variant({}, noise=-1.0)
            assert str(low_noise_variant_error.value)

        with raises(ValueError) as high_noise_given_error:
            fe.encode_givens({}, noise=99)
            assert str(high_noise_given_error.value)

        with raises(ValueError) as low_noise_given_error:
            fe.encode_givens({}, noise=-1.0)
            assert str(low_noise_given_error.value)

    def test_same_output_int_bool_1(self):

        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_INT_1_JSON'),
            second_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_BOOL_TRUE_JSON'))

    def test_same_output_big_int64_primitive_dict_big_int64(self):

        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_INT64_BIG_JSON'),
            second_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_PRIMITIVE_DICT_INT64_BIG_JSON'))

    def test_same_output_small_int64_primitive_dict_small_int64(self):

        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_INT64_SMALL_JSON'),
            second_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_PRIMITIVE_DICT_INT64_SMALL_JSON'))

    def test_same_output_int_bool_0(self):
        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_INT_0_JSON'),
            second_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_BOOL_FALSE_JSON'))

    def test_same_output_float_bool_1(self):

        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_FLOAT_1_JSON'),
            second_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_BOOL_TRUE_JSON'))

    def test_same_output_float_bool_0(self):

        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_FLOAT_0_JSON'),
            second_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_BOOL_FALSE_JSON'))

    def test_same_output_int_dict_0(self):

        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_INT_0_JSON'),
            second_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_PRIMITIVE_DICT_INT_0_JSON'))

    def test_same_output_float_dict_0(self):

        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_FLOAT_0_JSON'),
            second_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_PRIMITIVE_DICT_FLOAT_0_JSON'))

    def test_same_output_special_characters_string(self):

        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_SPECIAL_CHARACTERS_STRING_JSON'),
            second_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_PRIMITIVE_DICT_SPECIAL_CHARACTERS_STRING_JSON'))

    def test_same_output_unicode_emoji_01(self):

        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_UNICODE_EMOJI_01_JSON'),
            second_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_PRIMITIVE_DICT_UNICODE_EMOJI_01_JSON'))

    def test_same_output_unicode_emoji_02(self):

        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_UNICODE_EMOJI_02_JSON'),
            second_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_PRIMITIVE_DICT_UNICODE_EMOJI_02_JSON'))

    def test_same_output_unicode_string_01(self):

        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_UNICODE_STRING_01_JSON'),
            second_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_PRIMITIVE_DICT_UNICODE_STRING_01_JSON'))

    def test_same_output_unicode_string_02(self):

        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_UNICODE_STRING_02_JSON'),
            second_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_PRIMITIVE_DICT_UNICODE_STRING_02_JSON'))

    def test_same_output_unicode_string_with_u0000(self):

        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_UNICODE_STRING_WITH_U0000_JSON'),
            second_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_PRIMITIVE_DICT_UNICODE_STRING_WITH_U0000_JSON'))

    def test_same_output_unicode_u0000(self):

        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_UNICODE_U0000_JSON'),
            second_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_PRIMITIVE_DICT_UNICODE_U0000_JSON'))

    def test_same_output_zero_length_string(self):

        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_ZERO_LENGTH_STRING_JSON'),
            second_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_PRIMITIVE_DICT_ZERO_LENGTH_STRING_JSON'))

    def test_same_output_newline_tab_return_symbols_string(self):

        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_NEWLINE_TAB_RETURN_SYMBOLS_STRING_JSON'),
            second_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_PRIMITIVE_DICT_NEWLINE_TAB_RETURN_SYMBOLS_STRING_JSON'))

    def test_same_output_noise_2_256_3_256(self):
        pass

        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_NOISE_2_256_JSON'),
            second_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_NOISE_3_256_JSON'))

    def test_different_output_noise_2_128_3_128(self):

        self._generic_test_encode_record_for_different_output_from_json_data(
            first_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_NOISE_2_128_JSON'),
            second_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_NOISE_3_128_JSON'))

    # Test all primitive dicts: "string", true, false, 0, 0.0, 1, 1.0, -1, -1.0
    def test_primitive_dict_big_float(self):

        test_case_path = os.sep.join(
            [self.v6_test_suite_data_directory,
             os.getenv("FEATURE_ENCODER_TEST_PRIMITIVE_DICT_BIG_FLOAT_JSON")])

        test_case = \
            self._get_test_data(path_to_test_json=test_case_path)

        self._set_model_properties_from_test_case(test_case=test_case)

        test_input = test_case.get("test_case", None)

        if test_input is None:
            raise ValueError(
                "Input data for `test_big_float_variant` can`t be empty")

        variant_input = test_input.get('variant', None)
        given_input = test_input.get('given', None)

        if variant_input is None:
            raise ValueError(
                "Input variant data for `test_big_float_variant` can`t be empty")

        expected_output = test_case.get("test_output", None)

        if expected_output is None:
            raise ValueError(
                "Expected output for `test_big_float_variant` can`t be empty")

        expected_output = \
            dict((key, float(val)) for key, val in expected_output.items())

        tested_output_float64 = \
            self.feature_encoder.encode_variant(
                variant=variant_input, noise=self.noise,
                into=self.feature_encoder.encode_givens(
                    givens=deepcopy(given_input), noise=self.noise))

        tested_output_float32 = convert_values_to_float32(tested_output_float64)

        assert expected_output == tested_output_float32

    def test_primitive_dict_big_int_negative(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_PRIMITIVE_DICT_BIG_INT_NEGATIVE_JSON"))

    def test_primitive_dict_big_int_positive(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_PRIMITIVE_DICT_BIG_INT_POSITIVE_JSON"))

    def test_primitive_dict_big_int64(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_PRIMITIVE_DICT_INT64_BIG_JSON"))

    def test_primitive_dict_small_int64(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_PRIMITIVE_DICT_INT64_SMALL_JSON"))

    def test_primitive_dict_bool_false(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_PRIMITIVE_DICT_BOOL_FALSE_JSON"))

    def test_primitive_dict_bool_true(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_PRIMITIVE_DICT_BOOL_TRUE_JSON"))

    def test_primitive_dict_float_0(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_PRIMITIVE_DICT_FLOAT_0_JSON"))

    def test_primitive_dict_float_1(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_PRIMITIVE_DICT_FLOAT_1_JSON"))

    def test_primitive_dict_float_m1(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_PRIMITIVE_DICT_FLOAT_M1_JSON"))

    def test_foo_bar(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_FOO_BAR_JSON"))

    def test_dit_foo_bar(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_DICT_FOO_BAR_JSON"))

    def test_primitive_dict_foo_bar(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_PRIMITIVE_DICT_FOO_BAR_JSON"))

    def test_foo_bar_primitive_dict_equals_list(self):
        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_FOO_BAR_JSON'),
            second_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_PRIMITIVE_DICT_FOO_BAR_JSON'))

    def test_foo_bar_dict_equals_list(self):

        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv(
                'FEATURE_ENCODER_TEST_FOO_BAR_JSON'),
            second_test_case_filename=os.getenv(
                'FEATURE_ENCODER_TEST_DICT_FOO_BAR_JSON'))

    def test_same_output_string_dict(self):

        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_STRING_JSON'),
            second_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_PRIMITIVE_DICT_STRING_JSON'))

    def test_primitive_dict_int_0(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_PRIMITIVE_DICT_INT_0_JSON"))

    def test_primitive_dict_int_1(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_PRIMITIVE_DICT_INT_1_JSON"))

    def test_primitive_dict_int_m1(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_PRIMITIVE_DICT_INT_M1_JSON"))

    def test_primitive_dict_small_float(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_PRIMITIVE_DICT_SMALL_FLOAT_JSON"))

    def test_primitive_dict_string(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_PRIMITIVE_DICT_STRING_JSON"))

    def test_primitive_dict_special_characters_string(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_PRIMITIVE_DICT_SPECIAL_CHARACTERS_STRING_JSON"))

    def test_primitive_dict_unicode_emoji_01(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_PRIMITIVE_DICT_UNICODE_EMOJI_01_JSON"))

    def test_primitive_dict_unicode_emoji_02(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_PRIMITIVE_DICT_UNICODE_EMOJI_02_JSON"))

    def test_primitive_dict_unicode_string_01(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_PRIMITIVE_DICT_UNICODE_STRING_01_JSON"))

    def test_primitive_dict_unicode_string_02(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_PRIMITIVE_DICT_UNICODE_STRING_02_JSON"))

    def test_primitive_dict_unicode_string_with_u0000(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_PRIMITIVE_DICT_UNICODE_STRING_WITH_U0000_JSON"))

    def test_primitive_dict_unicode_u0000(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_PRIMITIVE_DICT_UNICODE_U0000_JSON"))

    def test_primitive_dict_zero_length_string(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_PRIMITIVE_DICT_ZERO_LENGTH_STRING_JSON"))

    def test_primitive_dict_newline_tab_return_symbols_string(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_PRIMITIVE_DICT_NEWLINE_TAB_RETURN_SYMBOLS_STRING_JSON"))

    def test_nested_list(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_NESTED_LIST_JSON"))

    def test_nested_dict_string_keys(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_NESTED_DICT_STRING_KEYS_JSON"))

    def test_external_collisions_01(self):
        self._generic_test_external_collisions(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_EXTERNAL_COLLISIONS_01_JSON"),
            input_data_key='test_case', expected_output_data_key='test_output',
            data_read_method='read')

    def test_external_collisions_02(self):
        self._generic_test_external_collisions(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_EXTERNAL_COLLISIONS_02_JSON"),
            input_data_key='test_case', expected_output_data_key='test_output',
            data_read_method='read')

    def test_internal_collision_01(self):
        self._generic_test_internal_collisions(
            test_case_filename=os.getenv(
                    "FEATURE_ENCODER_TEST_INTERNAL_COLLISIONS_01_JSON"),
            input_data_key='test_case', expected_output_data_key='test_output',
            data_read_method='read')

    def test_internal_collision_02(self):
        self._generic_test_internal_collisions(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_INTERNAL_COLLISIONS_02_JSON"),
            input_data_key='test_case', expected_output_data_key='test_output',
            data_read_method='read')

    def test_internal_collision_03(self):
        self._generic_test_internal_collisions(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_INTERNAL_COLLISIONS_03_JSON"),
            input_data_key='test_case', expected_output_data_key='test_output',
            data_read_method='read')

    def test_given_primitive_type_raises_type_error(self):

        model_seed = 0
        noise = 0

        self.feature_encoder = FeatureEncoder(model_seed=model_seed)

        for illegal_primitive in [0, 0.0, False, "string"]:

            with raises(TypeError) as type_err:

                self.feature_encoder.encode_givens(
                    givens=illegal_primitive, noise=noise)

            print('type_err.value')
            print(type_err.value)
            assert os.getenv("CYTHON_FEATURE_ENCODER_CONTEXT_TYPEERROR_MSG") \
                   in str(type_err.value)

    # def test_add_noise(
    #         self, into_key: str = 'into', noise_key: str = 'noise',
    #         test_input_key: str = 'test_case', test_output_key: str = 'test_output'):
    #
    #     test_case_filename = \
    #         os.getenv('FEATURE_ENCODER_TEST_ADD_NOISE_JSON', None)
    #
    #     if not test_case_filename:
    #         raise ValueError(
    #             'No envvar under key: FEATURE_ENCODER_TEST_ADD_NOISE_JSON')
    #
    #     test_case_path = os.sep.join(
    #         [self.v6_test_python_specific_data_directory, test_case_filename])
    #
    #     test_case = self._get_test_data(path_to_test_json=test_case_path)
    #
    #     pprint(test_case)
    #
    #     self._set_model_properties_from_test_case(test_case=test_case)
    #
    #     test_input = test_case.get(test_input_key, None)
    #
    #     if not test_input:
    #         raise ValueError('Test input is None')
    #
    #     test_into_float64 = test_input.get(into_key, None)
    #
    #     if not test_into_float64:
    #         raise ValueError(
    #             'Key {} is missing from the test case'.format(into_key))
    #
    #     test_noise = test_case.get(noise_key, None)
    #
    #     if not test_noise:
    #         raise ValueError(
    #             'Key {} is missing from the test case'.format(noise_key))
    #
    #     expected_output = test_case.get(test_output_key, None)
    #
    #     if not expected_output:
    #         raise ValueError(
    #             'Key {} is missing from the test case'.format(test_output_key))
    #
    #     add_noise(into=test_into_float64, noise=test_noise)
    #
    #     test_into_float32 = convert_values_to_float32(test_into_float64)
    #     expected_output_float32 = convert_values_to_float32(expected_output)
    #
    #     assert_dicts_identical(
    #         expected=expected_output_float32, calculated=test_into_float32)
    #
    #     # assert test_into_float32 == expected_output

    def test_reverse_sprinkle(self):

        x, sprinkled_x, small_noise = \
            self._get_sprinkled_value_and_noise()

        unsprinkled_x = \
            reverse_sprinkle(sprinkled_x=sprinkled_x, small_noise=small_noise)

        assert unsprinkled_x == x

    def test_get_previous_value(self):

        x, sprinkled_x, small_noise = \
            self._get_sprinkled_value_and_noise()

        features = {'abc': sprinkled_x}

        unsprinkled_x = _get_previous_value(
            feature_name='abc', into=features, small_noise=small_noise)

        assert unsprinkled_x == x

    def test_encode_feature_vector_raises_when_into_is_none(
            self, variant_key: str = 'variant', givens_key: str = 'givens',
            extra_features_key: str = 'extra_features',
            feature_names_key: str = 'feature_names',
            test_input_key: str = 'test_case'):

        test_case_filename = \
            os.getenv('FEATURE_ENCODER_TEST_ENCODE_FEATURE_VECTOR_JSON', None)

        if not test_case_filename:
            raise ValueError(
                'No envvar under key: FEATURE_ENCODER_TEST_ENCODE_FEATURE_VECTOR_JSON')

        test_case_path = os.sep.join(
            [self.v6_test_suite_data_directory, test_case_filename])

        test_case = self._get_test_data(path_to_test_json=test_case_path)

        self._set_model_properties_from_test_case(test_case=test_case)

        test_input = test_case.get(test_input_key, None)

        if not test_input:
            raise ValueError('Test input is None')

        test_variant = test_input.get(variant_key, None)

        if not test_variant:
            raise ValueError('Test variant is missing')

        test_givens = test_input.get(givens_key, None)

        if not test_givens:
            raise ValueError('Test givens is missing')

        test_extra_features = test_input.get(extra_features_key, None)

        if not test_extra_features:
            raise ValueError('Test extra features are missing')

        test_feature_names = test_input.get(feature_names_key, None)

        if not test_feature_names:
            raise ValueError('Test feature names are missing')

        tested_into = None

        with raises(ValueError) as val_err:
            self.feature_encoder.encode_feature_vector(
                variant=test_variant, givens=test_givens,
                extra_features=test_extra_features,
                feature_names=test_feature_names,
                noise=self.noise, into=tested_into)

            assert os.getenv("FEATURE_ENCODER_ENCODE_FEATURE_VECTOR_INTO_IS_NONE_VALERROR_MSG") \
                   in str(val_err.value)

    def test_encode_feature_vector_raises_on_worng_type_of_extra_features(
            self, variant_key: str = 'variant', givens_key: str = 'givens',
            extra_features_key: str = 'extra_features',
            feature_names_key: str = 'feature_names',
            test_input_key: str = 'test_case'):

        test_case_filename = \
            os.getenv('FEATURE_ENCODER_TEST_ENCODE_FEATURE_VECTOR_JSON', None)

        if not test_case_filename:
            raise ValueError(
                'No envvar under key: FEATURE_ENCODER_TEST_ENCODE_FEATURE_VECTOR_JSON')

        test_case_path = os.sep.join(
            [self.v6_test_suite_data_directory, test_case_filename])

        test_case = self._get_test_data(path_to_test_json=test_case_path)

        # TODO finish up

        self._set_model_properties_from_test_case(test_case=test_case)

        test_input = test_case.get(test_input_key, None)

        if not test_input:
            raise ValueError('Test input is None')

        test_variant = test_input.get(variant_key, None)

        if not test_variant:
            raise ValueError('Test variant is missing')

        test_givens = test_input.get(givens_key, None)

        if not test_givens:
            raise ValueError('Test givens is missing')

        test_feature_names = test_input.get(feature_names_key, None)

        if not test_feature_names:
            raise ValueError('Test feature names are missing')

        tested_into = np.full(len(test_feature_names), np.nan)
        test_extra_features = [1.0, 2.0, 3.0, 4.0, 5.0]

        with raises(TypeError) as type_err:
            self.feature_encoder.encode_feature_vector(
                variant=test_variant, givens=test_givens,
                extra_features=test_extra_features,
                feature_names=test_feature_names,
                noise=self.noise, into=tested_into)

            assert os.getenv("FEATURE_ENCODER_ENCODE_FEATURE_VECTOR_WRONG_TYPE_OF_EXTRA_FEATURES_TYPEERROR_MSG") \
                   in str(type_err.value)

    def test_add_multiple_extra_features(self):

        dummy_encoded_variants = [{'0': 1, '1': 1} for _ in range(3)]

        dummy_extra_features = [{'3': 3} for _ in range(3)]
        dummy_extra_features[1] = {}
        dummy_extra_features[2] = None

        fe = FeatureEncoder(model_seed=0)
        fe.add_extra_features(
            encoded_variants=dummy_encoded_variants,
            extra_features=dummy_extra_features)

        expected_result = [{'0': 1, '1': 1} for _ in range(3)]
        expected_result[0] = {'0': 1, '1': 1, '3': 3}

        assert expected_result == dummy_encoded_variants

    def test_extra_features_raises(self):

        dummy_variants_count = 3

        dummy_encoded_variants_dict = {'0': 1, '1': 1}
        dummy_encoded_variants_list = \
            [dummy_encoded_variants_dict for _ in range(dummy_variants_count)]

        dummy_extra_features_dict = {'3': 3}
        dummy_extra_features_list = \
            [dummy_extra_features_dict for _ in range(dummy_variants_count)]

        fe = FeatureEncoder(model_seed=0)
        with raises(TypeError) as terr:
            fe.add_extra_features(
                encoded_variants=dummy_encoded_variants_list,
                extra_features=dummy_extra_features_dict)
            assert terr.value

        with raises(TypeError) as terr:
            fe.add_extra_features(
                encoded_variants=dummy_encoded_variants_dict,
                extra_features=dummy_extra_features_list)
            assert terr.value

    def test_add_none_extra_features(self):

        dummy_encoded_variants = [{'0': 1, '1': 1} for _ in range(3)]

        dummy_extra_features = None

        fe = FeatureEncoder(model_seed=0)
        fe.add_extra_features(
            encoded_variants=dummy_encoded_variants,
            extra_features=dummy_extra_features)

        expected_result = [{'0': 1, '1': 1} for _ in range(3)]

        assert expected_result == dummy_encoded_variants
