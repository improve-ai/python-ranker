from copy import deepcopy
import json
import numpy as np
import os
from pytest import fixture, raises
import sys
from unittest import TestCase
import xgboost as xgb

sys.path.append(
    os.sep.join(str(os.path.abspath(__file__)).split(os.sep)[:-3]))

from improveai.chooser import USER_DEFINED_METADATA_KEY, \
    FEATURE_NAMES_METADATA_KEY
# TODO test _encode
from improveai.feature_encoder import scale, get_mask, sprinkle, \
    get_noise_shift_scale, FeatureEncoder
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
    def test_suite_data_directory(self) -> str:
        return self._test_data_directory

    @test_suite_data_directory.setter
    def test_suite_data_directory(self, value):
        self._test_data_directory = value

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

        self.test_suite_data_directory = \
            os.getenv("FEATURE_ENCODER_TEST_SUITE_JSONS_DIR")

        print('### self.test_suite_data_directory ###')
        print(self.test_suite_data_directory)

        self.test_python_specific_data_directory = \
            os.getenv("FEATURE_ENCODER_TEST_PYTHON_SPECIFIC_JSONS_DIR")

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

        manual_encode_into_float32 = \
            convert_values_to_float32(manual_encode_into_float64)

        return encode_feature_vector_into_float32, manual_encode_into_float32

    def _generic_test_encode_record_from_json_data(
            self, test_case_filename: str, input_data_key: str = 'test_case',
            variant_key: str = 'item', givens_key: str = 'context',
            expected_output_data_key: str = 'test_output', replace_none_with_nan: bool = False,
            big_float_case: bool = False):

        test_case_path = os.sep.join(
            [self.test_suite_data_directory, test_case_filename])

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

        print('### expected_output ###')
        print(expected_output)

        encode_feature_vector_into_float32, manual_encode_into_float32 = \
            self._get_encoded_arrays(variant_input=item_input, givens_input=context_input)

        if big_float_case:
            # currently a flat array of floats is expected to be the encoding's output
            print('Converting expected output to float')
            expected_output = [float(eo) for eo in expected_output]

        expected_output_float32 = convert_values_to_float32(expected_output)
        if replace_none_with_nan:
            print('### replace_none_with_nan ###')
            expected_output_float32[expected_output_float32 == None] = np.nan

        # check that encode_feature_vector() output is identical with expected output
        print('### expected_output_float32 ###')
        print(expected_output_float32)
        print('### encode_feature_vector_into_float32 ###')
        print(encode_feature_vector_into_float32)
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
            [self.test_suite_data_directory, first_test_case_filename])

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
            [self.test_suite_data_directory, second_test_case_filename])

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
            [self.test_suite_data_directory, first_test_case_filename])

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
            [self.test_suite_data_directory, second_test_case_filename])

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

    def test_none_item(self):

        item_input = None
        context_input = None

        test_case_path = os.sep.join(
            [self.test_suite_data_directory,
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
            [self.test_suite_data_directory, test_case_filename])

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

        tested_into_float32 = convert_values_to_float32(tested_into_float64)

        expected_output_float32 = np.array(expected_output).astype(np.float32)
        np.testing.assert_array_equal(expected_output_float32, tested_into_float32)

    def _generic_test_collisions(
            self, test_case_envvar, items_key: str = 'items',
            context_key: str = 'contexts', test_input_key: str = 'test_case',
            test_output_key: str = 'test_output', force_numpy: bool = False,
            replace_none_with_nan: bool = False):

        test_case_filename = os.getenv(test_case_envvar, None)
        assert test_case_filename is not None

        test_case_path = os.sep.join(
            [self.test_suite_data_directory, test_case_filename])

        test_case = self._get_test_data(path_to_test_json=test_case_path)

        self._set_model_properties_from_test_case(test_case=test_case)

        test_input = test_case.get(test_input_key, None)
        assert test_input is not None

        test_items = test_input.get(items_key, None)
        test_contexts = test_input.get(context_key, None)

        if test_contexts is None:
            test_contexts = [None] * len(test_items)

        expected_output = test_case.get(test_output_key, None)
        assert expected_output is not None
        expected_output = np.array(expected_output)

        if replace_none_with_nan:
            expected_output[expected_output == None] = np.nan

        tested_into_float64 = \
            np.full((len(test_items), len(self.feature_names)), np.nan)

        if force_numpy:
            orig_use_cython_backend = improve_settings.CYTHON_BACKEND_AVAILABLE
            improve_settings.CYTHON_BACKEND_AVAILABLE = False

        for item, context, into_row in zip(test_items, test_contexts, tested_into_float64):
            # self, item: object, context: object, into: np.ndarray, noise: float = 0.0
            self.feature_encoder.encode_feature_vector(
                item=item, context=context, into=into_row, noise=self.noise)

        if force_numpy:
            improve_settings.CYTHON_BACKEND_AVAILABLE = orig_use_cython_backend

        tested_into_float32 = convert_values_to_float32(tested_into_float64)

        expected_output_float32 = np.array(expected_output).astype(np.float32)
        np.testing.assert_array_equal(expected_output_float32, tested_into_float32)

    def test_empty_list(self):

        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_EMPTY_LIST_JSON"),
            replace_none_with_nan=True)

    def test_big_int_negative(self):

        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_BIG_INT_NEGATIVE_JSON"),
            replace_none_with_nan=False)

    def test_primitive_dict_int64_negative(self):

        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_PRIMITIVE_DICT_INT64_BIG_NEGATIVE_JSON"),
            replace_none_with_nan=False)

    def test_primitive_dict_int64_positive(self):

        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_PRIMITIVE_DICT_INT64_BIG_POSITIVE_JSON"),
            replace_none_with_nan=False)

    def test_big_int_positive(self):

        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "FEATURE_ENCODER_TEST_BIG_INT_POSITIVE_JSON"),
            replace_none_with_nan=False)

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
            [self.test_suite_data_directory, test_case_filename])

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
        print(test_into.dtype)

        with raises(ValueError) as verr:
            self.feature_encoder.encode_feature_vector(
                item=test_item, context=test_context,
                into=test_into, noise=self.noise)
        assert str(verr.value) == "into must be a float64 array"

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
            [self.test_suite_data_directory,
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

    def test_encode_item_raises_for_non_json_encodable(self):
        feature_encoder = FeatureEncoder(
            feature_names=['item', 'context'], string_tables={}, model_seed=1)
        with raises(ValueError) as verr:
            feature_encoder.encode_item(
                item=object, into=np.array([np.nan, np.nan]))

        with raises(ValueError) as verr:
            feature_encoder.encode_item(
                item=np.array([1, 2, 3]), into=np.array([np.nan, np.nan]))

    def test_encode_context_raises_for_non_json_encodable(self):
        feature_encoder = FeatureEncoder(
            feature_names=['item', 'context'], string_tables={}, model_seed=1)
        with raises(ValueError) as verr:
            feature_encoder.encode_context(
                context=object, into=np.array([np.nan, np.nan]))

        with raises(ValueError) as verr:
            feature_encoder.encode_context(
                context=np.array([1, 2, 3]), into=np.array([np.nan, np.nan]))

    def test_encode_feature_vector_raises_when_into_is_none(
            self, variant_key: str = 'item', givens_key: str = 'context',
            feature_names_key: str = 'feature_names',
            test_input_key: str = 'test_case'):

        test_case_filename = \
            os.getenv('FEATURE_ENCODER_TEST_ENCODE_FEATURE_VECTOR_JSON', None)
        assert test_case_filename is not None

        test_case_path = os.sep.join(
            [self.test_suite_data_directory, test_case_filename])

        test_case = self._get_test_data(path_to_test_json=test_case_path)

        self._set_model_properties_from_test_case(test_case=test_case)

        test_input = test_case.get(test_input_key, None)
        assert test_input is not None

        test_item = test_input.get(variant_key, None)
        assert test_item is not None

        test_context = test_input.get(givens_key, None)
        assert test_context is not None

        tested_into = None

        with raises(ValueError) as verr:
            self.feature_encoder.encode_feature_vector(
                item=test_item, context=test_context,
                noise=self.noise, into=tested_into)
        assert str(verr.value) == "into must be a float64 array"

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

    def test__encode_raises_for_bad_into_int_types(self):
        fe = FeatureEncoder(feature_names=['item', 'context'], string_tables={}, model_seed=0)

        for bad_into_dtype in [np.int, np.int32, bool, int]:
            into = np.array([-1, -1], dtype=bad_into_dtype)
            with raises(ValueError) as verr:
                fe._encode(obj=1, path='item', into=into)

    def test__encode_raises_for_bad_into_float_types(self):
        fe = FeatureEncoder(feature_names=['item', 'context'],
                            string_tables={}, model_seed=0)

        for bad_into_dtype in [np.float16, np.float32]:
            into = np.array([np.nan, np.nan], dtype=bad_into_dtype)
            with raises(ValueError) as verr:
                fe._encode(obj=1, path='item', into=into)

    def test__encode_raises_for_bad_into_string_types(self):
        fe = FeatureEncoder(feature_names=['item', 'context'],
                            string_tables={}, model_seed=0)

        for bad_into_dtype in [object, str, '<U16']:
            into = np.array([None, None], dtype=bad_into_dtype)
            with raises(ValueError) as verr:
                fe._encode(obj=1, path='item', into=into)

    def test_encode_feature_vector_raises_for_bad_into_int_types_item_none_context_none(self):
        fe = FeatureEncoder(feature_names=['item', 'context'], string_tables={}, model_seed=0)

        for bad_into_dtype in [np.int, np.int32, bool, int]:
            into = np.array([-1, -1], dtype=bad_into_dtype)
            with raises(ValueError) as verr:
                fe.encode_feature_vector(item=None, context=None, into=into)

    def test_encode_feature_vector_raises_for_bad_into_float_types_item_none_context_none(self):
        fe = FeatureEncoder(feature_names=['item', 'context'],
                            string_tables={}, model_seed=0)

        for bad_into_dtype in [np.float16, np.float32]:
            into = np.array([np.nan, np.nan], dtype=bad_into_dtype)
            with raises(ValueError) as verr:
                fe.encode_feature_vector(item=None, context=None, into=into)

    def test_encode_feature_vector_raises_for_bad_into_string_types_item_none_context_none(self):
        fe = FeatureEncoder(feature_names=['item', 'context'],
                            string_tables={}, model_seed=0)

        for bad_into_dtype in [object, str, '<U16']:
            into = np.array([None, None], dtype=bad_into_dtype)
            with raises(ValueError) as verr:
                fe.encode_feature_vector(item=None, context=None, into=into)

    def test_collisions_valid_items_no_context(self):
        self._generic_test_collisions(
            test_case_envvar='FEATURE_ENCODER_TEST_INTERNAL_COLLISIONS_VALID_ITEMS_NO_CONTEXT_JSON')

    def test_collisions_valid_items_and_context(self):
        self._generic_test_collisions(
            test_case_envvar='FEATURE_ENCODER_TEST_INTERNAL_COLLISIONS_VALID_ITEMS_AND_CONTEXT_JSON')

    def test_collisions_none_items_valid_context(self):
        self._generic_test_collisions(
            test_case_envvar='FEATURE_ENCODER_TEST_INTERNAL_COLLISIONS_NONE_ITEMS_AND_CONTEXT_JSON')


# TODO add sprinkle tests
def test_get_noise_shift_scale():
    noise_shift, noise_scale = get_noise_shift_scale(0.0)
    assert noise_shift == 0.0 and noise_scale == 1.0

    noise_shift, noise_scale = get_noise_shift_scale(0.5)
    # (8.96831017167883e-44, 1.0000038146972656)
    assert noise_shift == 8.96831017167883e-44 and noise_scale == 1.0000038146972656


def test_get_noise_shift_scale_raises_for_negative_noise():
    with raises(AssertionError) as aerr:
        get_noise_shift_scale(-0.1)


def test_get_noise_shift_scale_raises_for_noise_ge_1():
    with raises(AssertionError) as aerr:
        get_noise_shift_scale(1)

    with raises(AssertionError) as aerr:
        get_noise_shift_scale(5)


def test_sprinkle():
    # x, noise_shift, noise_scale
    sprinkled_x = sprinkle(1.0, 0.0, 1.0)
    assert sprinkled_x == 1.0

    sprinkled_x = sprinkle(1.0, 0.0, 0.0)
    assert sprinkled_x == 0.0

    sprinkled_x = sprinkle(1.0, 0.25, 1.25)
    assert sprinkled_x == 1.5625


def test_scale():
    scaled_miss = scale(0.0, 0.0)
    assert scaled_miss == 0.0

    scaled_miss = scale(1.0)
    assert scaled_miss == 1.0


def test_scale_raises_for_negative_width():
    with raises(AssertionError):
        scale(1.0, -0.1)


def test_get_mask():
    mask = get_mask([1])
    assert mask == 1

    mask = get_mask([2])
    assert mask == 3

    mask = get_mask([123])
    assert mask == 127

    mask = get_mask([])
    assert mask == 0

    mask = get_mask([0])
    assert mask == 0


def test_get_mask_raises_for_negative_string_hash():
    with raises(ValueError):
        get_mask([-1])
