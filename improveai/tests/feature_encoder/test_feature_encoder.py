from copy import deepcopy
import json
import numpy as np
import os
from pprint import pprint
from pytest import fixture, raises, warns
import sys
from unittest import TestCase
from warnings import catch_warnings, simplefilter
import xgboost as xgb

sys.path.append(
    os.sep.join(str(os.path.abspath(__file__)).split(os.sep)[:-3]))

from improveai.chooser import USER_DEFINED_METADATA_KEY, \
    FEATURE_NAMES_METADATA_KEY
# TODO test _encode
from improveai.feature_encoder import sprinkle, get_noise_shift_scale, FeatureEncoder
# TODO what should be imported here ?
import improveai.settings as improve_settings
from improveai.utils.general_purpose_tools import read_jsonstring_from_file
from improveai.tests.test_utils import convert_values_to_float32


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

        user_defined_metadata = json.loads(b.attr(USER_DEFINED_METADATA_KEY))
        self.feature_names = user_defined_metadata[FEATURE_NAMES_METADATA_KEY]

    def _get_test_data(
            self, path_to_test_json: str, method: str = 'readlines') -> object:

        loaded_jsonstring = read_jsonstring_from_file(
                    path_to_file=path_to_test_json, method=method)

        loaded_json = json.loads(loaded_jsonstring)

        return loaded_json

    def _set_model_properties_from_test_case(self, test_case: dict):
        # set model_seed
        self.encoder_seed = test_case.get("model_seed", None)
        assert self.encoder_seed is not None

        self.noise = float(test_case.get("noise", None))
        assert self.noise is not None

        self.string_tables = test_case.get('string_tables', None)
        assert self.string_tables is not None

        self.feature_names = test_case.get('feature_names', None)
        assert self.feature_names is not None

        self.string_tables = test_case.get('string_tables', None)
        assert self.string_tables is not None

        self.feature_encoder = FeatureEncoder(
            feature_names=self.feature_names, string_tables=self.string_tables,
            model_seed=self.encoder_seed)

    def _get_encoded_arrays(
            self, variant_input, givens_input):

        encode_feature_vector_into_float64 = np.full((len(self.feature_names),), np.nan)

        # check that encode_feature_vector returns desired output
        self.feature_encoder.encode_feature_vector(
            item=variant_input, context=givens_input,
            into=encode_feature_vector_into_float64, noise=self.noise)

        encode_feature_vector_into_float32 = \
            convert_values_to_float32(encode_feature_vector_into_float64)

        noise_shift, noise_scale = get_noise_shift_scale(self.noise)
        print(f'shift: {noise_shift} | scale: {noise_scale}')
        # check that encode_variant returns desired output
        manual_encode_into_float64 = np.full((len(self.feature_names),), np.nan)

        # encode_variant
        self.feature_encoder.encode_item(
            item=variant_input, into=manual_encode_into_float64,
            noise_shift=noise_shift, noise_scale=noise_scale)
        # encode_givens
        self.feature_encoder.encode_context(
            context=givens_input, into=manual_encode_into_float64,
            noise_shift=noise_shift, noise_scale=noise_scale)

        print('### encoded vectors ###')
        print(encode_feature_vector_into_float64)
        print(manual_encode_into_float64)

        manual_encode_into_float32 = \
            convert_values_to_float32(manual_encode_into_float64)

        return encode_feature_vector_into_float32, manual_encode_into_float32

    def _generic_test_encode_record_from_json_data(
            self, test_case_filename: str, input_data_key: str = 'test_case',
            variant_key: str = 'item', givens_key: str = 'context',
            expected_output_data_key: str = 'test_output', replace_none_with_nan: bool = False,
            big_float_case: bool = False):

        test_case_path = os.sep.join(
            [self.v6_test_suite_data_directory, test_case_filename])

        test_case = self._get_test_data(path_to_test_json=test_case_path)

        self._set_model_properties_from_test_case(test_case=test_case)

        test_input = test_case.get(input_data_key, None)
        assert test_input is not None

        item_input = test_input.get(variant_key, None)
        assert item_input is not None

        # givens and extra features can be None
        context_input = test_input.get(givens_key, None)

        expected_output = test_case.get(expected_output_data_key, None)
        assert expected_output is not None

        encode_feature_vector_into_float32, manual_encode_into_float32 = \
            self._get_encoded_arrays(variant_input=item_input, givens_input=context_input)

        if big_float_case:
            # currently a flat array of floats is expected to be the encoding's output
            expected_output = [float(eo) for eo in expected_output]

        expected_output_float32 = convert_values_to_float32(expected_output)
        if replace_none_with_nan:
            expected_output_float32[expected_output_float32 == None] = np.nan

        # T: 1.0000001192092896 | C: 1.0000001192092896
        # T: 12.000000953674316 | C: 12.000001907348633
        # T: 2.000000238418579 | C: 2.000000238418579

        # T: 1.0000001192092896 | C: 1.0000001192092896
        # T: 12.000000953674316 | C: 12.000001907348633
        # T: 2.000000238418579 | C: 2.000000238418579

        print(f'Noise: {self.noise} | seed: {self.encoder_seed}')
        for e1, e2 in zip(expected_output_float32, encode_feature_vector_into_float32):
            print(f'T: {e1} | C: {e2}')

            # print(f'T: {np.float32(e1)} | C: {np.float32(e2)}')

        # check that encode_feature_vector() output is identical with expected output
        np.testing.assert_array_equal(expected_output_float32, encode_feature_vector_into_float32)

        # check that 'manual' encoding output is identical with expected output
        np.testing.assert_array_equal(expected_output_float32, manual_encode_into_float32)

    def _generic_test_encode_record_for_same_output_from_json_data(
            self, first_test_case_filename: str, second_test_case_filename: str,
            provided_first_test_case: dict = None,
            provided_second_test_case: dict = None,
            input_data_key: str = 'test_case', item_key: str = 'item',
            context_key: str = 'context'):

        first_test_case_path = os.sep.join(
            [self.v6_test_suite_data_directory, first_test_case_filename])

        if not provided_first_test_case:
            first_test_case = \
                self._get_test_data(path_to_test_json=first_test_case_path)
        else:
            first_test_case = provided_first_test_case

        self._set_model_properties_from_test_case(test_case=first_test_case)

        first_test_input = first_test_case.get(input_data_key, None)
        assert first_test_input is not None

        first_variant_input = first_test_input.get(item_key, None)
        assert first_variant_input is not None

        first_given_input = first_test_input.get(context_key, None)

        first_encode_feature_vector_into_float32, first_manual_encode_into_float32 = \
            self._get_encoded_arrays(variant_input=first_variant_input, givens_input=first_given_input)

        second_test_case_path = os.sep.join(
            [self.v6_test_suite_data_directory, second_test_case_filename])

        if not provided_second_test_case:
            second_test_case = \
                self._get_test_data(path_to_test_json=second_test_case_path)
        else:
            second_test_case = provided_second_test_case

        self._set_model_properties_from_test_case(test_case=second_test_case)

        second_test_input = second_test_case.get(input_data_key, None)
        assert second_test_input is not None

        second_variant_input = second_test_input.get(item_key, None)
        assert second_variant_input is not None

        second_given_input = second_test_input.get(context_key, None)

        second_encode_feature_vector_into_float32, second_manual_encode_into_float32 = \
            self._get_encoded_arrays(variant_input=second_variant_input, givens_input=second_given_input)

        np.testing.assert_array_equal(
            first_encode_feature_vector_into_float32, first_manual_encode_into_float32)
        np.testing.assert_array_equal(
            first_encode_feature_vector_into_float32, second_encode_feature_vector_into_float32)
        np.testing.assert_array_equal(
            second_encode_feature_vector_into_float32, second_manual_encode_into_float32)
        np.testing.assert_array_equal(
            first_manual_encode_into_float32, second_manual_encode_into_float32)

    def _generic_test_encode_record_for_different_output_from_json_data(
            self, first_test_case_filename: str, second_test_case_filename: str,
            provided_first_test_case: dict = None,
            provided_second_test_case: dict = None,
            input_data_key: str = 'test_case', variant_key: str = 'item',
            givens_key: str = 'context'):

        first_test_case_path = os.sep.join(
            [self.v6_test_suite_data_directory, first_test_case_filename])

        if not provided_first_test_case:
            first_test_case = \
                self._get_test_data(path_to_test_json=first_test_case_path)
        else:
            first_test_case = provided_first_test_case

        self._set_model_properties_from_test_case(test_case=first_test_case)

        first_test_input = first_test_case.get(input_data_key, None)
        assert first_test_input is not None

        first_variant_input = first_test_input.get(variant_key, None)
        assert first_variant_input is not None

        first_given_input = first_test_input.get(givens_key, None)

        first_encode_feature_vector_into_float32, first_manual_encode_into_float32 = \
            self._get_encoded_arrays(variant_input=first_variant_input,
                                     givens_input=first_given_input)

        np.testing.assert_array_equal(first_encode_feature_vector_into_float32, first_manual_encode_into_float32)

        second_test_case_path = os.sep.join(
            [self.v6_test_suite_data_directory, second_test_case_filename])

        if not provided_second_test_case:
            second_test_case = \
                self._get_test_data(path_to_test_json=second_test_case_path)
        else:
            second_test_case = provided_second_test_case

        self._set_model_properties_from_test_case(test_case=second_test_case)

        second_test_input = second_test_case.get(input_data_key, None)
        assert second_test_input is not None

        second_variant_input = second_test_input.get(variant_key, None)
        assert second_variant_input is not None

        second_given_input = second_test_input.get(givens_key, None)

        second_encode_feature_vector_into_float32, second_manual_encode_into_float32 = \
            self._get_encoded_arrays(variant_input=second_variant_input,
                                     givens_input=second_given_input)
        np.testing.assert_array_equal(second_encode_feature_vector_into_float32, second_manual_encode_into_float32)

        with raises(AssertionError) as aerr:
            np.testing.assert_array_equal(
                first_encode_feature_vector_into_float32, second_encode_feature_vector_into_float32)
        assert aerr.match(r'Arrays are not equal')

        with raises(AssertionError) as aerr:
            np.testing.assert_array_equal(
                first_manual_encode_into_float32, second_manual_encode_into_float32)
        assert aerr.match(r'Arrays are not equal')



    # # TODO currently there is no chance for any collision (root keys for variant and givens are different)?
    # def _generic_test_external_collisions(
    #         self, test_case_filename: str, input_data_key: str = 'test_case',
    #         expected_output_data_key: str = 'test_output',
    #         variant_key: str = 'variant', givens_key: str = 'givens',
    #         data_read_method: str = 'read'):
    #
    #     test_case_path = os.sep.join(
    #         [self.v6_test_suite_data_directory, test_case_filename])
    #
    #     test_case = \
    #         self._get_test_data(
    #             path_to_test_json=test_case_path, method=data_read_method)
    #
    #     self._set_model_properties_from_test_case(test_case=test_case)
    #
    #     test_case_input = test_case.get(input_data_key, None)
    #
    #     if test_case_input is None:
    #         raise ValueError("Test jsonlines are empty")
    #
    #     variant = test_case_input.get(variant_key, None)
    #     givens = test_case_input.get(givens_key, None)
    #
    #     if variant is None or givens is None:
    #         raise ValueError("variant or givens are empty")
    #
    #     expected_output = test_case.get(expected_output_data_key, None)
    #
    #     if expected_output is None:
    #         raise ValueError("Expected output is empty")
    #
    #     encoded_variant = \
    #         self.feature_encoder.encode_variant(variant=variant, noise=self.noise)
    #     encoded_givens = \
    #         self.feature_encoder.encode_givens(givens=givens, noise=self.noise)
    #
    #     common_keys = \
    #         set(encoded_variant.keys()).intersection(set(encoded_givens.keys()))
    #
    #     assert len(common_keys) > 0
    #
    #     fully_encoded_variant_float64 = \
    #         self.feature_encoder.encode_variant(
    #             variant=variant, noise=self.noise,
    #             into=self.feature_encoder.encode_givens(
    #                 givens=deepcopy(givens), noise=self.noise))
    #
    #     fully_encoded_variant_float32 = \
    #         convert_values_to_float32(fully_encoded_variant_float64)
    #
    #     expected_output_float32 = convert_values_to_float32(expected_output)
    #
    #     # np.testing.assert_array_equal(
    #     #     expected_output_float32, fully_encoded_variant_float32)
    #
    #     for expected_key in expected_output.keys():
    #         expected_value = expected_output[expected_key]
    #         calculated_value = fully_encoded_variant_float32[expected_key]
    #         print(expected_value)
    #         print(calculated_value)
    #         assert np.float32(expected_value) == calculated_value
    #
    #     single_common_key = list(common_keys)[0]
    #
    #     assert single_common_key in encoded_variant.keys() \
    #            and single_common_key in encoded_givens.keys() \
    #            and single_common_key in fully_encoded_variant_float32.keys()
    #
    # def _generic_test_internal_collisions(
    #         self, test_case_filename: str, input_data_key: str = 'test_case',
    #         expected_output_data_key: str = 'test_output',
    #         variant_key: str = 'variant', givens_key: str = 'givens',
    #         data_read_method: str = 'read'):
    #
    #     test_case_path = os.sep.join(
    #         [self.v6_test_suite_data_directory, test_case_filename])
    #
    #     test_case = \
    #         self._get_test_data(
    #             path_to_test_json=test_case_path, method=data_read_method)
    #
    #     self._set_model_properties_from_test_case(test_case=test_case)
    #
    #     test_case_input = test_case.get(input_data_key, None)
    #
    #     if test_case_input is None:
    #         raise ValueError("Test jsonlines are empty")
    #
    #     variant = test_case_input.get(variant_key, None)
    #     givens = test_case_input.get(givens_key, None)
    #
    #     if variant is None or givens is None:
    #         raise ValueError("variant or givens are empty")
    #
    #     expected_output = test_case.get(expected_output_data_key, None)
    #
    #     if expected_output is None:
    #         raise ValueError("Expected output is empty")
    #
    #     fully_encoded_variant_float64 = \
    #         self.feature_encoder.encode_variant(
    #             variant=variant, noise=self.noise,
    #             into=self.feature_encoder.encode_givens(
    #                 givens=deepcopy(givens), noise=self.noise))
    #
    #     fully_encoded_variant_float32 = \
    #         convert_values_to_float32(fully_encoded_variant_float64)
    #
    #     expected_output_float32 = convert_values_to_float32(expected_output)
    #     assert fully_encoded_variant_float32 == expected_output_float32

    # test all None-like types (None, [], {}, np.NaN)

    def test_none_item(self):

        item_input = None
        context_input = None

        test_case_path = os.sep.join(
            [self.v6_test_suite_data_directory,
             os.getenv("FEATURE_ENCODER_TEST_NONE_JSON")])

        test_case = \
            self._get_test_data(path_to_test_json=test_case_path)

        self._set_model_properties_from_test_case(test_case=test_case)

        noise_shift, noise_scale = get_noise_shift_scale(self.noise)

        tested_into_float64 = np.full((2, ), np.nan)
        self.feature_encoder.encode_context(
            context=context_input, into=tested_into_float64, noise_shift=noise_shift,
            noise_scale=noise_scale)
        self.feature_encoder.encode_item(
            item=item_input, into=tested_into_float64, noise_shift=noise_shift,
            noise_scale=noise_scale)

        tested_output_float32 = convert_values_to_float32(tested_into_float64)
        assert np.isnan(tested_output_float32).all()

    # TODO update if still needed
    def _test_encode_feature_vector(
            self, test_case_envvar, item_key: str = 'item',
            context_key: str = 'context',
            test_input_key: str = 'test_case',
            test_output_key: str = 'test_output', force_numpy: bool = False,
            replace_none_with_nan: bool = False):

        test_case_filename = os.getenv(test_case_envvar, None)
        assert test_case_filename is not None

        test_case_path = os.sep.join(
            [self.v6_test_suite_data_directory, test_case_filename])

        test_case = self._get_test_data(path_to_test_json=test_case_path)

        self._set_model_properties_from_test_case(test_case=test_case)

        test_input = test_case.get(test_input_key, None)
        assert test_input is not None

        test_item = test_input.get(item_key, None)
        test_context = test_input.get(context_key, None)

        expected_output = np.array(test_case.get(test_output_key, None))
        assert expected_output is not None

        if replace_none_with_nan:
            expected_output[expected_output == None] = np.nan

        tested_into_float64 = np.full(len(self.feature_names, ), np.nan)

        if force_numpy:
            orig_use_cython_backend = improve_settings.CYTHON_BACKEND_AVAILABLE
            improve_settings.CYTHON_BACKEND_AVAILABLE = False

        # self, item: object, context: object, into: np.ndarray, noise: float = 0.0
        self.feature_encoder.encode_feature_vector(
            item=test_item, context=test_context,
            into=tested_into_float64, noise=self.noise)

        if force_numpy:
            improve_settings.CYTHON_BACKEND_AVAILABLE = orig_use_cython_backend

        # print('tested_into')
        # print(tested_into_float64.dtype)
        # print('expected_output')
        # print(expected_output.dtype)
        # print('diff')
        # print(tested_into_float64 - expected_output)

        tested_into_float32 = convert_values_to_float32(tested_into_float64)

        # print('tested_into_float32')
        # print(tested_into_float32)
        # print('expected_output')
        # print(expected_output)

        expected_output_float32 = np.array(expected_output).astype(np.float32)
        np.testing.assert_array_equal(expected_output_float32, tested_into_float32)

    def test_empty_list(self):

        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_EMPTY_LIST_JSON"),
            replace_none_with_nan=True)

    def test_empty_dict(self):

        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_EMPTY_DICT_JSON"),
            replace_none_with_nan=True)

    def test_dict_with_null_value(self):

        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_PRIMITIVE_DICT_NONE_JSON"),
            replace_none_with_nan=True)

    # def test_npnan(self):
    #
    #     self._generic_test_encode_record_from_json_data(
    #         test_case_filename=os.getenv(
    #             "FEATURE_ENCODER_TEST_NAN_JSON"))

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

    def _generic_test_big_float(self, test_case_filename: str):

        assert test_case_filename is not None

        test_case_path = os.sep.join(
            [self.v6_test_suite_data_directory, test_case_filename])

        test_case = \
            self._get_test_data(path_to_test_json=test_case_path)

        self._set_model_properties_from_test_case(test_case=test_case)

        test_input = test_case.get("test_case", None)
        assert test_input is not None

        variant_input = test_input.get('variant', None)
        assert variant_input is not None

        givens_input = test_input.get('givens', None)

        expected_output = test_case.get("test_output", None)
        assert expected_output is not None

        expected_output = \
            dict((key, float(val)) for key, val in expected_output.items())

        tested_output_float64 = \
            self.feature_encoder.encode_item(
                item=variant_input, noise=self.noise,
                into=self.feature_encoder.encode_context(
                    context=deepcopy(givens_input), noise=self.noise))

        tested_output_float32 = convert_values_to_float32(tested_output_float64)

        assert expected_output == tested_output_float32

    def test_big_positive_float(self):
        big_float_test_case_file = os.getenv('FEATURE_ENCODER_TEST_BIG_POSITIVE_FLOAT_JSON', None)
        assert big_float_test_case_file is not None

        self._generic_test_encode_record_from_json_data(
            test_case_filename=big_float_test_case_file, big_float_case=True)

    def test_big_positive_float_noise_0(self):
        big_float_test_case_file = os.getenv('FEATURE_ENCODER_TEST_BIG_POSITIVE_FLOAT_NOISE_0_JSON', None)
        assert big_float_test_case_file is not None

        self._generic_test_encode_record_from_json_data(
            test_case_filename=big_float_test_case_file, big_float_case=True)

    def test_big_negative_float(self):
        big_float_test_case_file = os.getenv('FEATURE_ENCODER_TEST_BIG_NEGATIVE_FLOAT_JSON', None)
        assert big_float_test_case_file is not None

        self._generic_test_encode_record_from_json_data(
            test_case_filename=big_float_test_case_file, big_float_case=True)

    def test_big_negative_float_noise_0(self):
        big_float_test_case_file = os.getenv('FEATURE_ENCODER_TEST_BIG_NEGATIVE_FLOAT_NOISE_0_JSON', None)
        assert big_float_test_case_file is not None

        self._generic_test_encode_record_from_json_data(
            test_case_filename=big_float_test_case_file, big_float_case=True)

        # test_case_path = os.sep.join(
        #     [self.v6_test_suite_data_directory,
        #      os.getenv("FEATURE_ENCODER_TEST_BIG_FLOAT_JSON")])
        #
        # test_case = \
        #     self._get_test_data(path_to_test_json=test_case_path)
        #
        # self._set_model_properties_from_test_case(test_case=test_case)
        #
        # test_input = test_case.get("test_case", None)
        #
        # if test_input is None:
        #     raise ValueError(
        #         "Input data for `test_big_float_variant` can`t be empty")
        #
        # variant_input = test_input.get('variant', None)
        # given_input = test_input.get('given', None)
        #
        # if variant_input is None:
        #     raise ValueError(
        #         "Input variant data for `test_big_float_variant` can`t be empty")
        #
        # expected_output = test_case.get("test_output", None)
        #
        # if expected_output is None:
        #     raise ValueError(
        #         "Expected output for `test_big_float_variant` can`t be empty")
        #
        # expected_output = \
        #     dict((key, float(val)) for key, val in expected_output.items())
        #
        # tested_output_float64 = \
        #     self.feature_encoder.encode_variant(
        #         variant=variant_input, noise=self.noise,
        #         into=self.feature_encoder.encode_givens(
        #             givens=deepcopy(given_input), noise=self.noise))
        #
        # tested_output_float32 = convert_values_to_float32(tested_output_float64)
        #
        # assert expected_output == tested_output_float32

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

    def test_unicode_zero_length_string_collides(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_ZERO_LENGTH_STRING_COLLIDES_JSON"))

    def test_newline_tab_return_symbols_string(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_NEWLINE_TAB_RETURN_SYMBOLS_STRING_JSON"))

    def test_noise_0_with_float(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_NOISE_0_WITH_FLOAT_JSON"))

    def test_noise_099_with_float(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_NOISE_099_WITH_FLOAT_JSON"))

    def test_noise_0_with_int(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_NOISE_0_WITH_INT_JSON"))

    def test_noise_099_with_int(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_NOISE_099_WITH_INT_JSON"))

    def test_noise_0_with_primitive_dict_float(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_NOISE_0_WITH_PRIMITIVE_DICT_FLOAT_JSON"))

    def test_noise_099_with_primitive_dict_float(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_NOISE_099_WITH_PRIMITIVE_DICT_FLOAT_JSON"))

    def test_noise_0_with_primitive_dict_int(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_NOISE_0_WITH_PRIMITIVE_DICT_INT_JSON"))

    def test_noise_099_with_primitive_dict_int(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_NOISE_099_WITH_PRIMITIVE_DICT_INT_JSON"))

    def test_noise_0_with_string(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_NOISE_0_WITH_STRING_JSON"))

    def test_noise_099_with_string(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_NOISE_099_WITH_STRING_JSON"))

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

    # def test_leading_zeros_in_feature_names_01(self):
    #     self._generic_test_encode_record_from_json_data(
    #         test_case_filename=os.getenv(
    #             "FEATURE_ENCODER_TEST_LEADING_ZEROS_IN_FEATURE_NAME_01"))
    #
    # def test_leading_zeros_in_feature_names_02(self):
    #     self._generic_test_encode_record_from_json_data(
    #         test_case_filename=os.getenv(
    #             "FEATURE_ENCODER_TEST_LEADING_ZEROS_IN_FEATURE_NAME_02"))

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

    def test_encode_feature_vector_item_nan_context_none_into_none(self):

        self.feature_encoder = \
            FeatureEncoder(feature_names=['item', 'context'], string_tables={}, model_seed=0)
        self.noise = 0.0

        test_item = np.nan
        test_context = None
        test_into = np.full((2,), None)

        self.feature_encoder.encode_feature_vector(
            item=test_item, context=test_context,
            into=test_into, noise=self.noise)

        np.testing.assert_array_equal(test_into, np.array([None, None]))

    def test_encode_feature_vector_item_None_context_none_into_nan(self):

        # feature_names: list, string_tables: dict, model_seed: int
        self.feature_encoder = \
            FeatureEncoder(feature_names=['item', 'context'], string_tables={}, model_seed=0)
        self.noise = 0.0

        test_item = None
        test_context = None
        test_into = np.full((2,), None)

        self.feature_encoder.encode_feature_vector(
            item=test_item, context=test_context,
            into=test_into, noise=self.noise)

        np.testing.assert_array_equal(test_into, np.array([None, None]))

    def test_encode_feature_vector_item_nan_context_none_into_nan(self):

        # feature_names: list, string_tables: dict, model_seed: int
        self.feature_encoder = \
            FeatureEncoder(feature_names=['item', 'context'], string_tables={}, model_seed=0)
        self.noise = 0.0

        test_item = None
        test_context = np.nan
        test_into = np.full((2,), np.nan)

        self.feature_encoder.encode_feature_vector(
            item=test_item, context=test_context,
            into=test_into, noise=self.noise)

        np.testing.assert_array_equal(test_into, np.array([np.nan, np.nan]))

    def test_noise_out_of_bounds_raises(self):
        # fe = FeatureEncoder(
        #     model_seed=0, feature_names=['a', 'b', 'c'], string_tables={})

        with raises(AssertionError) as aerr:
            get_noise_shift_scale(noise=99)
            assert str(aerr.value)

        with raises(AssertionError) as aerr:
            get_noise_shift_scale(noise=-1.0)
            assert str(aerr.value)

    def test_same_output_int_bool_1(self):

        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_INT_1_JSON'),
            second_test_case_filename=os.getenv('FEATURE_ENCODER_TEST_BOOL_TRUE_JSON'))

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
        assert test_input is not None

        item_input = test_input.get('item', None)
        assert item_input is not None

        context_input = test_input.get('context', None)

        expected_output = test_case.get("test_output", None)
        assert expected_output is not None

        expected_output = np.array([float(item) for item in expected_output])

        tested_output_float64 = np.full((len(self.feature_names), ), np.nan)
        noise_shift, noise_scale = get_noise_shift_scale(noise=self.noise)
        self.feature_encoder.encode_context(
            context=context_input, into=tested_output_float64,
            noise_shift=noise_shift, noise_scale=noise_scale)
        self.feature_encoder.encode_item(
            item=item_input, into=tested_output_float64, noise_shift=noise_shift,
            noise_scale=noise_scale)

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

    # def test_primitive_dict_big_int64(self):
    #     self._generic_test_encode_record_from_json_data(
    #         test_case_filename=os.getenv(
    #             "FEATURE_ENCODER_TEST_PRIMITIVE_DICT_INT64_BIG_JSON"))
    #
    # def test_primitive_dict_small_int64(self):
    #     self._generic_test_encode_record_from_json_data(
    #         test_case_filename=os.getenv(
    #             "FEATURE_ENCODER_TEST_PRIMITIVE_DICT_INT64_SMALL_JSON"))

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

    def test_dict_foo_bar(self):
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
                'FEATURE_ENCODER_TEST_DICT_FOO_BAR_1_JSON'))

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

    # def test_external_collisions_01(self):
    #     self._generic_test_external_collisions(
    #         test_case_filename=os.getenv(
    #             "FEATURE_ENCODER_TEST_EXTERNAL_COLLISIONS_01_JSON"),
    #         input_data_key='test_case', expected_output_data_key='test_output',
    #         data_read_method='read')
    #
    # def test_external_collisions_02(self):
    #     self._generic_test_external_collisions(
    #         test_case_filename=os.getenv(
    #             "FEATURE_ENCODER_TEST_EXTERNAL_COLLISIONS_02_JSON"),
    #         input_data_key='test_case', expected_output_data_key='test_output',
    #         data_read_method='read')
    #
    # def test_internal_collision_01(self):
    #     self._generic_test_internal_collisions(
    #         test_case_filename=os.getenv(
    #                 "FEATURE_ENCODER_TEST_INTERNAL_COLLISIONS_01_JSON"),
    #         input_data_key='test_case', expected_output_data_key='test_output',
    #         data_read_method='read')
    #
    # def test_internal_collision_02(self):
    #     self._generic_test_internal_collisions(
    #         test_case_filename=os.getenv(
    #             "FEATURE_ENCODER_TEST_INTERNAL_COLLISIONS_02_JSON"),
    #         input_data_key='test_case', expected_output_data_key='test_output',
    #         data_read_method='read')
    #
    # def test_internal_collision_03(self):
    #     self._generic_test_internal_collisions(
    #         test_case_filename=os.getenv(
    #             "FEATURE_ENCODER_TEST_INTERNAL_COLLISIONS_03_JSON"),
    #         input_data_key='test_case', expected_output_data_key='test_output',
    #         data_read_method='read')

    # def test_given_primitive_type_raises_type_error(self):
    #
    #     model_seed = 0
    #     noise = 0
    #
    #     self.feature_encoder = FeatureEncoder(model_seed=model_seed, feature_names=['item'], string_tables={})
    #
    #     for illegal_primitive in [np.array([]), b'abc', np.nan, np.int(123), np.float64(1.234), object, str]:
    #
    #         with raises(TypeError) as type_err:
    #
    #             self.feature_encoder.encode_context(
    #                 context=illegal_primitive, noise=noise)
    #
    #         assert os.getenv("FEATURE_ENCODER_CONTEXT_TYPEERROR_MSG") \
    #                in str(type_err.value)

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
    #     assert test_into_float32 == expected_output_float32

    def test_encode_feature_vector_raises_when_into_is_none(
            self, variant_key: str = 'item', givens_key: str = 'context',
            feature_names_key: str = 'feature_names',
            test_input_key: str = 'test_case'):

        test_case_filename = \
            os.getenv('FEATURE_ENCODER_TEST_ENCODE_FEATURE_VECTOR_JSON', None)
        assert test_case_filename is not None

        test_case_path = os.sep.join(
            [self.v6_test_suite_data_directory, test_case_filename])

        test_case = self._get_test_data(path_to_test_json=test_case_path)

        self._set_model_properties_from_test_case(test_case=test_case)

        test_input = test_case.get(test_input_key, None)
        assert test_input is not None

        test_item = test_input.get(variant_key, None)
        assert test_item is not None

        test_context = test_input.get(givens_key, None)
        assert test_context is not None

        tested_into = None

        with raises(TypeError) as terr:
            self.feature_encoder.encode_feature_vector(
                item=test_item, context=test_context,
                noise=self.noise, into=tested_into)

            # assert os.getenv("FEATURE_ENCODER_ENCODE_FEATURE_VECTOR_INTO_IS_NONE_VALERROR_MSG") \
            #        in str(val_err.value)

    # def test_encode_feature_vector_raises_on_worng_type_of_extra_features(
    #         self, variant_key: str = 'variant', givens_key: str = 'givens',
    #         feature_names_key: str = 'feature_names',
    #         test_input_key: str = 'test_case'):
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
    #     test_feature_names = test_input.get(feature_names_key, None)
    #
    #     if not test_feature_names:
    #         raise ValueError('Test feature names are missing')
    #
    #     tested_into = np.full(len(test_feature_names), np.nan)
    #     test_extra_features = [1.0, 2.0, 3.0, 4.0, 5.0]
    #
    #     with raises(TypeError) as type_err:
    #         self.feature_encoder.encode_feature_vector(
    #             item=test_variant, context=test_givens,
    #             feature_names=test_feature_names,
    #             noise=self.noise, into=tested_into)
    #
    #         assert os.getenv("FEATURE_ENCODER_ENCODE_FEATURE_VECTOR_WRONG_TYPE_OF_EXTRA_FEATURES_TYPEERROR_MSG") \
    #                in str(type_err.value)

    def test_encode_valid_types(self):
        fe = FeatureEncoder(
            feature_names=['a', 'b', 'c'],
            string_tables={'a': [], 'b': [], 'c': []},
            model_seed=0)
        valid_objects = \
            [1, 1.123, True, False, 'abc', None, [1, 2, 3], {'1': 2, '3': '4'}, (1, 2, 3, 4)]
        for vo in valid_objects:
            # object_, seed, small_noise, features
            into = np.full((100,), np.nan, dtype=float)
            fe._encode(obj=vo, path='a', into=into, noise_shift=0.0, noise_scale=1.0)

    def test_encode_raises_for_invalid_types(self):
        # feature_names: list, string_tables: dict, model_seed: int
        fe = FeatureEncoder(
            feature_names=['a', 'b', 'c'],
            string_tables={'a': [1, 2, 3, 4, 5],
                           'b': [1, 2, 3, 4, 5],
                           'c': [1, 2, 3, 4, 5]},
            model_seed=0)
        # test for custom object
        class CustomObject:
            pass

        # example types which are not JSON serializable
        # obj: object, path: str, into: np.ndarray, noise_shift: float = 0.0, noise_scale: float = 1.0
        with raises(ValueError) as aerr:
            fe._encode(obj=CustomObject(), path='a', into=np.array([]), noise_shift=0.0, noise_scale=0.0)

        with raises(ValueError) as aerr:
            fe._encode(obj=np.array([1, 2, 3]),  path='a', into=np.array([]), noise_shift=0.0, noise_scale=0.0)

        with raises(ValueError) as aerr:
            fe._encode(obj=object, path='a', into=np.array([]), noise_shift=0.0, noise_scale=0.0)

    def test_encode_raises_for_non_string_keys(self):
        fe = FeatureEncoder(
            feature_names=['a', 'b', 'c'],
            string_tables={'a': [1, 2, 3, 4, 5],
                           'b': [1, 2, 3, 4, 5],
                           'c': [1, 2, 3, 4, 5]},
            model_seed=0)
        with raises(TypeError) as aerr:
            fe._encode(obj={'a': 1, 'b': 2, 3: 3},  path='a', into=np.array([]), noise_shift=0.0, noise_scale=0.0)

        with raises(TypeError) as aerr:
            fe._encode(obj={'a': 1, 'b': 2, 3.3: 3},  path='a', into=np.array([]), noise_shift=0.0, noise_scale=0.0)

        with raises(TypeError) as aerr:
            fe._encode(obj={'a': 1, 'b': 2, True: 3},  path='a', into=np.array([]), noise_shift=0.0, noise_scale=0.0)

        with raises(TypeError) as aerr:
            fe._encode(obj={'a': 1, 'b': 2, False: 3},  path='a', into=np.array([]), noise_shift=0.0, noise_scale=0.0)

        with raises(TypeError) as aerr:
            fe._encode(obj={'a': 1, 'b': 2, None: 3},  path='a', into=np.array([]), noise_shift=0.0, noise_scale=0.0)

        with raises(TypeError) as aerr:
            fe._encode(obj={'a': 1, 'b': 2, (1, 2, 3): 3},  path='a', into=np.array([]), noise_shift=0.0, noise_scale=0.0)

        with raises(TypeError) as aerr:
            fe._encode(obj={'a': 1, 'b': 2, 'c': {'1': 1, 2: 2}},  path='a', into=np.array([]), noise_shift=0.0, noise_scale=0.0)
