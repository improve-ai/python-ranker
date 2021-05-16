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

from improveai.feature_encoder import FeatureEncoder
from improveai.utils.general_purpose_utils import read_jsonstring_from_file


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
        # self.encoder_seed = int(os.getenv("V6_FEATURE_ENCODER_MODEL_SEED"))
        # self.noise_seed = int(os.getenv("V6_FEATURE_ENCODER_NOISE_SEED"))

        self.v6_test_suite_data_directory = \
            os.getenv("V6_FEATURE_ENCODER_TEST_SUITE_JSONS_DIR")

        self.v6_test_python_specific_data_directory = \
            os.getenv("V6_FEATURE_ENCODER_TEST_PYTHON_SPECIFIC_JSONS_DIR")
        # self.feature_encoder = FeatureEncoder(model_seed=self.encoder_seed)

        # np.random.seed(self.noise_seed)
        # self.noise = np.random.rand()

        self._set_feature_names()

    def _set_feature_names(self):
        b = xgb.Booster()
        b.load_model(os.getenv("V6_DUMMY_MODEL_PATH"))

        user_defined_metadata = json.loads(b.attr('user_defined_metadata'))[
            'json']
        self.feature_names = np.array(user_defined_metadata['feature_names'])

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
            variant_key: str = 'variant', given_key: str = 'context',
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

        given_input = test_input.get(given_key, None)

        expected_output = test_case.get(expected_output_data_key, None)

        if expected_output is None:
            raise ValueError('Expected output is empty')

        tested_record_output = \
            self.feature_encoder.encode_variants(
                variants=[variant_input], context=given_input, noise=self.noise)

        # print(variant_input)
        # print(expected_output)
        # print(tested_record_output)

        assert expected_output == tested_record_output

    def _generic_test_encode_record_for_same_output_from_json_data(
            self, first_test_case_filename: str, second_test_case_filename: str,
            provided_first_test_case: dict = None,
            provided_second_test_case: dict = None,
            input_data_key: str = 'test_case', variant_key: str = 'variant',
            given_key: str = 'context'):

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
        first_given_inputs = first_test_input.get(given_key, None)

        pprint(first_variant_input)

        if first_variant_input is None:
            raise ValueError(
                "First test variant input for is empty")

        first_output = \
            self.feature_encoder.encode_variants(
                variants=[first_variant_input], context=first_given_inputs,
                noise=self.noise)

        second_test_case_path = os.sep.join(
            [self.v6_test_suite_data_directory, second_test_case_filename])

        if not provided_second_test_case:
            second_test_case = \
                self._get_test_data(path_to_test_json=second_test_case_path)
        else:
            second_test_case = provided_second_test_case

        second_test_input = second_test_case.get(input_data_key, None)

        if second_test_input is None:
            raise ValueError(
                "Second test input for is empty")

        second_variant_input = second_test_input.get(variant_key, None)
        second_given_input = second_test_input.get(given_key, None)

        print(second_variant_input)

        if second_variant_input is None:
            raise ValueError(
                "Second test input for is empty")

        second_output = \
            self.feature_encoder.encode_variants(
                variants=[second_variant_input], context=second_given_input,
                noise=self.noise)

        assert first_output == second_output

    def _generic_test_batch_input_encoding(
            self, test_case_filename: str, input_data_key: str = 'test_case',
            expected_output_data_key: str = 'single_given_test_output',
            variant_key: str = 'variant', given_key: str = 'given',
            single_given_encoding: bool = True,
            plain_jsonlines_encoding: bool = False,
            data_read_method: str = 'read'):

        test_case_path = os.sep.join(
            [self.v6_test_python_specific_data_directory, test_case_filename])

        test_case = \
            self._get_test_data(
                path_to_test_json=test_case_path, method=data_read_method)

        self._set_model_properties_from_test_case(test_case=test_case)

        test_jsonlines = test_case.get(input_data_key, None)

        if test_jsonlines is None:
            raise ValueError("Test jsonlines are empty")

        test_variants = np.array([jl[variant_key] for jl in test_jsonlines])
        test_givens = \
            np.array([jl.get(given_key, {}) for jl in test_jsonlines])

        if single_given_encoding:
            test_givens = test_givens[0]

        expected_output = test_case.get(expected_output_data_key, None)

        if expected_output is None:
            raise ValueError("Expected output is empty")

        if not plain_jsonlines_encoding:
            tested_output = \
                self.feature_encoder.encode_variants(
                    variants=test_variants, context=test_givens,
                    noise=self.noise)
        else:
            tested_output = \
                self.feature_encoder.encode_jsonlines(
                    jsonlines=np.array(test_jsonlines), noise=self.noise,
                    variant_key=variant_key, context_key=given_key)

        np.testing.assert_array_equal(expected_output, tested_output)

    def _generic_test_external_collisions(
            self, test_case_filename: str, input_data_key: str = 'test_case',
            expected_output_data_key: str = 'test_output',
            variant_key: str = 'variant', given_key: str = 'context',
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
        given = test_case_input.get(given_key, None)

        if variant is None or given is None:
            raise ValueError("variant or given are empty")

        expected_output = test_case.get(expected_output_data_key, None)

        if expected_output is None:
            raise ValueError("Expected output is empty")

        encoded_variant = \
            self.feature_encoder.encode_variant(variant=variant, noise=self.noise)
        encoded_given = \
            self.feature_encoder.encode_context(context=given, noise=self.noise)

        common_keys = \
            set(encoded_variant.keys()).intersection(set(encoded_given.keys()))

        assert len(common_keys) > 0

        fully_encoded_variant = \
            self.feature_encoder.encode_variants(
                variants=[variant], context=given, noise=self.noise)[0]

        np.testing.assert_array_equal(expected_output, fully_encoded_variant)

        single_common_key = list(common_keys)[0]

        assert single_common_key in encoded_variant.keys() \
               and single_common_key in encoded_given.keys() \
               and single_common_key in fully_encoded_variant.keys()

        assert encoded_variant.get(single_common_key, None) + \
               encoded_given.get(single_common_key, None) == \
               fully_encoded_variant.get(single_common_key, None)

    def _generic_test_internal_collisions(
            self, test_case_filename: str, input_data_key: str = 'test_case',
            expected_output_data_key: str = 'test_output',
            variant_key: str = 'variant', given_key: str = 'context',
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
        given = test_case_input.get(given_key, None)

        if variant is None or given is None:
            raise ValueError("variant or given are empty")

        expected_output = test_case.get(expected_output_data_key, None)

        if expected_output is None:
            raise ValueError("Expected output is empty")

        fully_encoded_variant = \
            self.feature_encoder.encode_variants(
                variants=[variant], context=given, noise=self.noise)[0]

        assert fully_encoded_variant == expected_output

    # test all None-like types (None, [], {}, np.NaN)
    def test_none_variant(self):

        variant_input = None
        given_input = None

        test_case_path = os.sep.join(
            [self.v6_test_suite_data_directory,
             os.getenv("V6_FEATURE_ENCODER_TEST_NONE_JSON")])

        test_case = \
            self._get_test_data(path_to_test_json=test_case_path)

        self._set_model_properties_from_test_case(test_case=test_case)

        expected_output = test_case.get('test_output', None)

        if expected_output is None:
            raise ValueError("Expected output empty for `test_none_variant`")

        tested_output = \
            self.feature_encoder.encode_variants(
                variants=[variant_input], context=given_input, noise=self.noise)

        assert expected_output == tested_output

    def test_empty_list(self):

        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_EMPTY_LIST_JSON"))

    def test_empty_dict(self):

        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_EMPTY_DICT_JSON"))

    def test_npnan(self):

        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_NAN_JSON"))

    # Test all primitive types: "string", true, false, 0, 0.0, 1, 1.0, -1, -1.0
    def test_true(self):

        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_BOOL_TRUE_JSON"))

    def test_false(self):

        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_BOOL_FALSE_JSON"))

    def test_string(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_STRING_JSON"))

    def test_int_0(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_INT_0_JSON"))

    def test_float_0(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_FLOAT_0_JSON"))

    def test_int_1(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_INT_1_JSON"))

    def test_float_1(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_FLOAT_1_JSON"))

    def test_int_m1(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_INT_M1_JSON"))

    def test_float_m1(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_FLOAT_M1_JSON"))

    def test_big_float(self):

        test_case_path = os.sep.join(
            [self.v6_test_suite_data_directory,
             os.getenv("V6_FEATURE_ENCODER_TEST_BIG_FLOAT_JSON")])

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

        tested_output = \
            self.feature_encoder.encode_variants(
                variants=[variant_input], context=given_input,
                noise=self.noise)

        assert expected_output == tested_output

    def test_small_float(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_SMALL_FLOAT_JSON"))

    def test_special_characters_string(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_SPECIAL_CHARACTERS_STRING_JSON"))

    def test_special_characters_in_key_string(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_SPECIAL_CHARACTERS_IN_KEY_STRING_JSON"))

    def test_unicode_emoji_01(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_UNICODE_EMOJI_01_JSON"))

    def test_unicode_emoji_02(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_UNICODE_EMOJI_02_JSON"))

    def test_unicode_emoji_in_key(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_UNICODE_EMOJI_IN_KEY_JSON"))

    def test_unicode_string_01(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_UNICODE_STRING_01_JSON"))

    def test_unicode_string_02(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_UNICODE_STRING_02_JSON"))

    def test_unicode_string_with_u0000(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_UNICODE_STRING_WITH_U0000_JSON"))

    def test_unicode_u0000(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_UNICODE_U0000_JSON"))

    def test_unicode_zero_length_string(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_ZERO_LENGTH_STRING_JSON"))

    def test_newline_tab_return_symbols_string(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_NEWLINE_TAB_RETURN_SYMBOLS_STRING_JSON"))

    def test_noise_0_with_string(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_NOISE_0_WITH_STRING_JSON"))

    def test_noise_1_with_string(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_NOISE_1_WITH_STRING_JSON"))

    def test_noise_out_of_bounds_raises(self):
        fe = FeatureEncoder(model_seed=0)

        with raises(ValueError) as high_noise_variant_error:
            fe.encode_variant({}, noise=99)
            assert str(high_noise_variant_error.value)

        with raises(ValueError) as low_noise_variant_error:
            fe.encode_variant({}, noise=-1.0)
            assert str(low_noise_variant_error.value)
            
        with raises(ValueError) as high_noise_given_error:
            fe.encode_context({}, noise=99)
            assert str(high_noise_given_error.value)

        with raises(ValueError) as low_noise_given_error:
            fe.encode_context({}, noise=-1.0)
            assert str(low_noise_given_error.value)

    def test_same_output_int_bool_1(self):

        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv('V6_FEATURE_ENCODER_TEST_INT_1_JSON'),
            second_test_case_filename=os.getenv('V6_FEATURE_ENCODER_TEST_BOOL_TRUE_JSON'))

    def test_same_output_int_bool_0(self):
        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv('V6_FEATURE_ENCODER_TEST_INT_0_JSON'),
            second_test_case_filename=os.getenv('V6_FEATURE_ENCODER_TEST_BOOL_FALSE_JSON'))

    def test_same_output_float_bool_1(self):

        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv('V6_FEATURE_ENCODER_TEST_FLOAT_1_JSON'),
            second_test_case_filename=os.getenv('V6_FEATURE_ENCODER_TEST_BOOL_TRUE_JSON'))

    def test_same_output_float_bool_0(self):

        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv('V6_FEATURE_ENCODER_TEST_FLOAT_0_JSON'),
            second_test_case_filename=os.getenv('V6_FEATURE_ENCODER_TEST_BOOL_FALSE_JSON'))

    def test_same_output_int_dict_0(self):

        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv('V6_FEATURE_ENCODER_TEST_INT_0_JSON'),
            second_test_case_filename=os.getenv('V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_INT_0_JSON'))

    def test_same_output_float_dict_0(self):

        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv('V6_FEATURE_ENCODER_TEST_FLOAT_0_JSON'),
            second_test_case_filename=os.getenv('V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_FLOAT_0_JSON'))

    def test_same_output_special_characters_string(self):

        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv('V6_FEATURE_ENCODER_TEST_SPECIAL_CHARACTERS_STRING_JSON'),
            second_test_case_filename=os.getenv('V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_SPECIAL_CHARACTERS_STRING_JSON'))

    def test_same_output_unicode_emoji_01(self):

        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv('V6_FEATURE_ENCODER_TEST_UNICODE_EMOJI_01_JSON'),
            second_test_case_filename=os.getenv('V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_UNICODE_EMOJI_01_JSON'))

    def test_same_output_unicode_emoji_02(self):

        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv('V6_FEATURE_ENCODER_TEST_UNICODE_EMOJI_02_JSON'),
            second_test_case_filename=os.getenv('V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_UNICODE_EMOJI_02_JSON'))

    def test_same_output_unicode_string_01(self):

        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv('V6_FEATURE_ENCODER_TEST_UNICODE_STRING_01_JSON'),
            second_test_case_filename=os.getenv('V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_UNICODE_STRING_01_JSON'))

    def test_same_output_unicode_string_02(self):

        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv('V6_FEATURE_ENCODER_TEST_UNICODE_STRING_02_JSON'),
            second_test_case_filename=os.getenv('V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_UNICODE_STRING_02_JSON'))

    def test_same_output_unicode_string_with_u0000(self):

        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv('V6_FEATURE_ENCODER_TEST_UNICODE_STRING_WITH_U0000_JSON'),
            second_test_case_filename=os.getenv('V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_UNICODE_STRING_WITH_U0000_JSON'))

    def test_same_output_unicode_u0000(self):

        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv('V6_FEATURE_ENCODER_TEST_UNICODE_U0000_JSON'),
            second_test_case_filename=os.getenv('V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_UNICODE_U0000_JSON'))

    def test_same_output_zero_length_string(self):

        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv('V6_FEATURE_ENCODER_TEST_ZERO_LENGTH_STRING_JSON'),
            second_test_case_filename=os.getenv('V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_ZERO_LENGTH_STRING_JSON'))

    def test_same_output_newline_tab_return_symbols_string(self):

        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv('V6_FEATURE_ENCODER_TEST_NEWLINE_TAB_RETURN_SYMBOLS_STRING_JSON'),
            second_test_case_filename=os.getenv('V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_NEWLINE_TAB_RETURN_SYMBOLS_STRING_JSON'))

    # Test all primitive dicts: "string", true, false, 0, 0.0, 1, 1.0, -1, -1.0
    def test_primitive_dict_big_float(self):

        test_case_path = os.sep.join(
            [self.v6_test_suite_data_directory,
             os.getenv("V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_BIG_FLOAT_JSON")])

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

        tested_output = \
            self.feature_encoder.encode_variants(
                variants=[variant_input], context=given_input,
                noise=self.noise)

        assert expected_output == tested_output

    def test_primitive_dict_big_int_negative(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_BIG_INT_NEGATIVE_JSON"))

    def test_primitive_dict_big_int_positive(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_BIG_INT_POSITIVE_JSON"))

    def test_primitive_dict_bool_false(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_BOOL_FALSE_JSON"))

    def test_primitive_dict_bool_true(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_BOOL_TRUE_JSON"))

    def test_primitive_dict_float_0(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_FLOAT_0_JSON"))

    def test_primitive_dict_float_1(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_FLOAT_1_JSON"))

    def test_primitive_dict_float_m1(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_FLOAT_M1_JSON"))

    def test_primitive_dict_foo_bar(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_FOO_BAR_JSON"))

    def test_foo_bar_primitive_dict_equals_list(self):
        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv('V6_FEATURE_ENCODER_TEST_FOO_BAR_JSON'),
            second_test_case_filename=os.getenv('V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_FOO_BAR_JSON'))

    def test_foo_bar_dict_equals_list(self):

        # foo_bar_dict_test_case = \
        #     {
        #         "test_case": {
        #             "variant": {"$value":
        #                         {"\0\0\0\0\0\0\0\0": "foo",
        #                          "\0\0\0\0\0\0\0\1": "bar"}}
        #         },
        #         "test_output": {
        #             "fb2b2ee3": -4750.032350104165,
        #             "90804277": 2851.019427773728,
        #             "627617c9": 13673.093147046791,
        #             "e9bac36b": 11840.080660683132
        #         },
        #         "model_seed": 1,
        #         "noise": 0.8928601514360016,
        #         "variant_seed": 2675988294294598568,
        #         "value_seed": 6818340268807889528,
        #         "context_seed": 5164679660109946987
        #     }

        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv(
                'V6_FEATURE_ENCODER_TEST_FOO_BAR_JSON'),
            second_test_case_filename=os.getenv(
                'V6_FEATURE_ENCODER_TEST_DICT_FOO_BAR_JSON'))  # ,
            # provided_second_test_case=foo_bar_dict_test_case)

    def test_same_output_string_dict(self):

        self._generic_test_encode_record_for_same_output_from_json_data(
            first_test_case_filename=os.getenv('V6_FEATURE_ENCODER_TEST_STRING_JSON'),
            second_test_case_filename=os.getenv('V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_STRING_JSON'))

    def test_primitive_dict_int_0(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_INT_0_JSON"))

    def test_primitive_dict_int_1(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_INT_1_JSON"))

    def test_primitive_dict_int_m1(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_INT_M1_JSON"))

    def test_primitive_dict_small_float(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_SMALL_FLOAT_JSON"))

    def test_primitive_dict_string(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_STRING_JSON"))

    def test_primitive_dict_special_characters_string(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_SPECIAL_CHARACTERS_STRING_JSON"))

    def test_primitive_dict_unicode_emoji_01(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_UNICODE_EMOJI_01_JSON"))

    def test_primitive_dict_unicode_emoji_02(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_UNICODE_EMOJI_02_JSON"))

    def test_primitive_dict_unicode_string_01(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_UNICODE_STRING_01_JSON"))

    def test_primitive_dict_unicode_string_02(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_UNICODE_STRING_02_JSON"))

    def test_primitive_dict_unicode_string_with_u0000(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_UNICODE_STRING_WITH_U0000_JSON"))

    def test_primitive_dict_unicode_u0000(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_UNICODE_U0000_JSON"))

    def test_primitive_dict_zero_length_string(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_ZERO_LENGTH_STRING_JSON"))

    def test_primitive_dict_newline_tab_return_symbols_string(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_NEWLINE_TAB_RETURN_SYMBOLS_STRING_JSON"))

    def test_nested_list(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_NESTED_LIST_JSON"))

    def test_nested_dict_string_keys(self):
        self._generic_test_encode_record_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_NESTED_DICT_STRING_KEYS_JSON"))

    def test_batch_variants_encoding_with_single_given(self):
        self._generic_test_batch_input_encoding(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_BATCH_ENCODING_JSONLINES"),
            expected_output_data_key="single_given_test_output",
            single_given_encoding=True, data_read_method='read')

    def test_batch_variants_encoding_with_multiple_givens(self):
        self._generic_test_batch_input_encoding(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_BATCH_ENCODING_JSONLINES"),
            expected_output_data_key="multiple_given_test_output",
            single_given_encoding=False, data_read_method='read')

    def test_batch_jsonlines_encoding(self):
        self._generic_test_batch_input_encoding(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_BATCH_ENCODING_JSONLINES"),
            expected_output_data_key="plain_jsonlines_test_output",
            single_given_encoding=False, plain_jsonlines_encoding=True,
            data_read_method='read')

    def test_missing_features_filler_method_01(self):
        test_case_path = os.sep.join(
            [self.v6_test_python_specific_data_directory,
             os.getenv("V6_FEATURE_ENCODER_TEST_BATCH_FILLING_MISSING_FEATURES_01")])

        test_case = \
            self._get_test_data(path_to_test_json=test_case_path, method="read")

        self._set_model_properties_from_test_case(test_case=test_case)

        test_jsonlines = test_case.get("test_case", None)

        if test_jsonlines is None:
            raise ValueError("Test input empty")

        test_variants = np.array([jl["variant"] for jl in test_jsonlines])
        test_givens = [jl.get("given", {}) for jl in test_jsonlines][0]

        expected_output = test_case.get("test_output", None)

        if expected_output is None:
            raise ValueError("Expected output is empty")

        encoded_variants = \
            self.feature_encoder.encode_variants(
                variants=test_variants, context=test_givens,
                noise=self.noise)

        missings_filled_array = self.feature_encoder.fill_missing_features(
            encoded_variants=encoded_variants, feature_names=self.feature_names)

        np.testing.assert_array_equal(expected_output, missings_filled_array)

    def test_missing_features_filler_method_02(self):
        test_case_path = os.sep.join(
            [self.v6_test_python_specific_data_directory,
             os.getenv("V6_FEATURE_ENCODER_TEST_BATCH_FILLING_MISSING_FEATURES_02")])

        test_case = \
            self._get_test_data(path_to_test_json=test_case_path, method="read")

        self._set_model_properties_from_test_case(test_case=test_case)

        test_jsonlines = test_case.get("test_case", None)

        if test_jsonlines is None:
            raise ValueError("Test input empty")

        test_variants = np.array([jl["variant"] for jl in test_jsonlines])
        test_givens = [jl.get("given", {}) for jl in test_jsonlines][0]

        expected_output = test_case.get("test_output", None)

        if expected_output is None:
            raise ValueError("Expected output is empty")

        encoded_variants = \
            self.feature_encoder.encode_variants(
                variants=test_variants, context=test_givens,
                noise=self.noise)

        missings_filled_array = self.feature_encoder.fill_missing_features(
            encoded_variants=encoded_variants, feature_names=self.feature_names)

        np.testing.assert_array_equal(expected_output, missings_filled_array)

    def test_external_collisions_01(self):
        self._generic_test_external_collisions(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_EXTERNAL_COLLISIONS_01_JSON"),
            input_data_key='test_case', expected_output_data_key='test_output',
            data_read_method='read')

    def test_external_collisions_02(self):
        self._generic_test_external_collisions(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_EXTERNAL_COLLISIONS_02_JSON"),
            input_data_key='test_case', expected_output_data_key='test_output',
            data_read_method='read')

    def test_internal_collision_01(self):
        self._generic_test_internal_collisions(
            test_case_filename=os.getenv(
                    "V6_FEATURE_ENCODER_TEST_INTERNAL_COLLISIONS_01_JSON"),
            input_data_key='test_case', expected_output_data_key='test_output',
            data_read_method='read')

    def test_internal_collision_02(self):
        self._generic_test_internal_collisions(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_INTERNAL_COLLISIONS_02_JSON"),
            input_data_key='test_case', expected_output_data_key='test_output',
            data_read_method='read')

    def test_internal_collision_03(self):
        self._generic_test_internal_collisions(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_INTERNAL_COLLISIONS_02_JSON"),
            input_data_key='test_case', expected_output_data_key='test_output',
            data_read_method='read')

    def test_given_primitive_type_raises_type_error(self):

        model_seed = 0
        noise = 0

        self.feature_encoder = FeatureEncoder(model_seed=model_seed)

        for illegal_primitive in [0, 0.0, False, "string"]:

            with raises(TypeError) as type_err:

                self.feature_encoder.encode_context(
                    context=illegal_primitive, noise=noise)

            assert os.getenv("V6_FEATURE_ENCODER_CONTEXT_TYPEERROR_MSG") \
                   in str(type_err.value)
