import json
import numpy as np
import os
from pytest import fixture, raises
import sys
from unittest import TestCase
import xgboost as xgb

sys.path.append(
    os.sep.join(str(os.path.abspath(__file__)).split(os.sep)[:-3]))

from feature_encoders.v6 import FeatureEncoder
from utils.general_purpose_utils import read_jsonstring_from_file


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
    def v6_test_data_directory(self) -> str:
        return self._v6_test_data_directory

    @v6_test_data_directory.setter
    def v6_test_data_directory(self, value):
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

        self.v6_test_data_directory = \
            os.getenv("V6_FEATURE_ENCODER_TEST_JSONS_DIR")
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
            self, path_to_test_json: str,
            # test_data_key: str = 'test_case',
            method: str = 'readlines') -> object:

        loaded_json = \
            json.loads(
                read_jsonstring_from_file(
                    path_to_file=path_to_test_json, method=method))

        return loaded_json

        # test_case = loaded_json.get(test_data_key, None)
        # if test_case is None:
        #     raise ValueError(
        #         'No `{}` key / test data found in provided json: {}'
        #         .format(test_data_key, path_to_test_json))
        # return test_case

    def _set_model_properties_from_test_case(self, test_case: dict):
        # set model_seed
        self.encoder_seed = test_case.get("model_seed", None)

        if self.encoder_seed is None:
            raise ValueError("model_seed is missing from a test case")

        self.noise = test_case.get("noise", None)

        if self.noise is None:
            raise ValueError("noise is missing from a test case")

        self.feature_encoder = FeatureEncoder(model_seed=self.encoder_seed)

    def _generic_test_encode_variant_from_json_data(
            self, test_case_filename: str, input_data_key: str = 'test_case',
            expected_output_data_key: str = 'variant_test_output'):

        test_case_path = os.sep.join(
            [self.v6_test_data_directory, test_case_filename])

        test_case = self._get_test_data(path_to_test_json=test_case_path)

        self._set_model_properties_from_test_case(test_case=test_case)

        test_input = test_case.get(input_data_key, None)

        if test_input is None:
            raise ValueError('Test input is empty')

        expected_output = test_case.get(expected_output_data_key, None)

        if expected_output is None:
            raise ValueError('Expected output is empty')

        tested_output = \
            self.feature_encoder.encode_variant(
                variant=test_input, noise=self.noise)

        assert expected_output == tested_output

    def _generic_test_encode_context_from_json_data(
            self, test_case_filename: str, input_data_key: str = 'test_case',
            expected_output_data_key: str = 'context_test_output'):

        test_case_path = os.sep.join(
            [self.v6_test_data_directory, test_case_filename])

        test_case = self._get_test_data(path_to_test_json=test_case_path)

        self._set_model_properties_from_test_case(test_case=test_case)

        test_input = test_case.get(input_data_key, None)

        if test_input is None:
            raise ValueError('Test input is empty')

        expected_output = test_case.get(expected_output_data_key, None)

        if expected_output is None:
            raise ValueError('Expected output is empty')

        if test_input and not isinstance(test_input, dict):
            with raises(TypeError) as type_err:

                self.feature_encoder.encode_context(
                    context=test_input, noise=self.noise)
            assert os.getenv("V6_FEATURE_ENCODER_CONTEXT_TYPEERROR_MSG") \
                   in str(type_err.value)
            return

        tested_output = \
            self.feature_encoder.encode_context(
                context=test_input, noise=self.noise)

        assert expected_output == tested_output

    def _generic_test_batch_input_encoding(
            self, test_case_filename: str, input_data_key: str = 'test_case',
            expected_output_data_key: str = 'single_context_test_output',
            variant_key: str = 'variant', context_key: str = 'context',
            single_context_encoding: bool = True,
            plain_jsonlines_encoding: bool = False,
            data_read_method: str = 'read'):

        test_case_path = os.sep.join(
            [self.v6_test_data_directory, test_case_filename])

        test_case = \
            self._get_test_data(
                path_to_test_json=test_case_path, method=data_read_method)

        self._set_model_properties_from_test_case(test_case=test_case)

        test_jsonlines = test_case.get(input_data_key, None)

        if test_jsonlines is None:
            raise ValueError("Test jsonlines are empty")

        test_variants = np.array([jl[variant_key] for jl in test_jsonlines])
        test_contexts = \
            np.array([jl.get(context_key, {}) for jl in test_jsonlines])

        if single_context_encoding:
            test_contexts = test_contexts[0]

        expected_output = test_case.get(expected_output_data_key, None)

        if expected_output is None:
            raise ValueError("Expected output is empty")

        if not plain_jsonlines_encoding:
            tested_output = \
                self.feature_encoder.encode_variants(
                    variants=test_variants, contexts=test_contexts,
                    noise=self.noise)
        else:
            tested_output = \
                self.feature_encoder.encode_jsonlines(
                    jsonlines=np.array(test_jsonlines), noise=self.noise,
                    variant_key=variant_key, context_key=context_key)

        np.testing.assert_array_equal(expected_output, tested_output)

    # test all None-like types (None, [], {}, np.NaN)
    def test_none_variant(self):

        test_input = None

        test_case_path = os.sep.join(
            [self.v6_test_data_directory,
             os.getenv("V6_FEATURE_ENCODER_TEST_NONE_JSON")])

        test_case = self._get_test_data(path_to_test_json=test_case_path)

        self._set_model_properties_from_test_case(test_case=test_case)

        expected_output = test_case.get('variant_test_output', None)

        if expected_output is None:
            raise ValueError("Expected output empty for `test_none_variant`")

        # expected_output = self._get_test_data(
        #     path_to_test_json=expected_output_path, test_data_key='test_output')

        tested_output = \
            self.feature_encoder.encode_variant(
                variant=test_input, noise=self.noise)

        assert expected_output == tested_output

    def test_empty_list_variant(self):

        self._generic_test_encode_variant_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_EMPTY_LIST_JSON"))

    def test_empty_dict_variant(self):

        self._generic_test_encode_variant_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_EMPTY_DICT_JSON"))

    def test_npnan_variant(self):

        self._generic_test_encode_variant_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_NAN_JSON"))

    def test_none_context(self):

        test_input = None

        test_case_path = os.sep.join(
            [self.v6_test_data_directory,
             os.getenv("V6_FEATURE_ENCODER_TEST_NONE_JSON")])

        test_case = self._get_test_data(path_to_test_json=test_case_path)

        self._set_model_properties_from_test_case(test_case=test_case)

        expected_output = test_case.get('context_test_output', None)

        if expected_output is None:
            raise ValueError("Expected output for `test_none_context` is empty")

        tested_output = \
            self.feature_encoder.encode_context(
                context=test_input, noise=self.noise)

        assert expected_output == tested_output

    def test_empty_list_context(self):

        self._generic_test_encode_context_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_EMPTY_LIST_JSON"))

    def test_empty_dict_context(self):

        self._generic_test_encode_context_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_EMPTY_DICT_JSON"))

    def test_npnan_context(self):

        self._generic_test_encode_context_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_NAN_JSON"))

    # Test all primitive types: "string", true, false, 0, 0.0, 1, 1.0, -1, -1.0
    def test_true_variant(self):

        self._generic_test_encode_variant_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_BOOL_TRUE_JSON"))

    def test_false_variant(self):

        self._generic_test_encode_variant_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_BOOL_FALSE_JSON"))

    def test_true_context(self):

        self._generic_test_encode_context_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_BOOL_TRUE_JSON"))

    def test_false_context(self):

        self._generic_test_encode_context_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_BOOL_FALSE_JSON"))

    def test_string_variant(self):
        self._generic_test_encode_variant_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_STRING_JSON"))

    def test_string_context(self):
        self._generic_test_encode_context_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_STRING_JSON"))

    def test_int_0_variant(self):
        self._generic_test_encode_variant_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_INT_0_JSON"))

    def test_int_0_context(self):
        self._generic_test_encode_context_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_INT_0_JSON"))

    def test_float_0_variant(self):
        self._generic_test_encode_variant_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_FLOAT_0_JSON"))

    def test_float_0_context(self):
        self._generic_test_encode_context_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_FLOAT_0_JSON"))

    def test_int_1_variant(self):
        self._generic_test_encode_variant_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_INT_1_JSON"))

    def test_int_1_context(self):
        self._generic_test_encode_context_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_INT_1_JSON"))

    def test_float_1_variant(self):
        self._generic_test_encode_variant_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_FLOAT_1_JSON"))

    def test_float_1_context(self):
        self._generic_test_encode_context_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_FLOAT_1_JSON"))

    def test_int_m1_variant(self):
        self._generic_test_encode_variant_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_INT_M1_JSON"))

    def test_int_m1_context(self):
        self._generic_test_encode_context_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_INT_M1_JSON"))

    def test_float_m1_variant(self):
        self._generic_test_encode_variant_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_FLOAT_M1_JSON"))

    def test_float_m1_context(self):
        self._generic_test_encode_context_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_FLOAT_M1_JSON"))

    def test_big_float_variant(self):

        test_case_path = os.sep.join(
            [self.v6_test_data_directory,
             os.getenv("V6_FEATURE_ENCODER_TEST_BIG_FLOAT_JSON")])

        test_case = \
            self._get_test_data(
                path_to_test_json=test_case_path)

        self._set_model_properties_from_test_case(test_case=test_case)

        test_input = test_case.get("test_case", None)

        if test_input is None:
            raise ValueError(
                "Input data for `test_big_float_variant` can`t be empty")

        expected_output = test_case.get("variant_test_output", None)

        if expected_output is None:
            raise ValueError(
                "Expected output for `test_big_float_variant` can`t be empty")

        expected_output = \
            dict((key, float(val)) for key, val in expected_output.items())

        tested_output = \
            self.feature_encoder.encode_variant(
                variant=test_input, noise=self.noise)

        assert expected_output == tested_output

    def test_big_float_context(self):
        test_case_path = os.sep.join(
            [self.v6_test_data_directory,
             os.getenv("V6_FEATURE_ENCODER_TEST_BIG_FLOAT_JSON")])

        test_case = \
            self._get_test_data(
                path_to_test_json=test_case_path)

        self._set_model_properties_from_test_case(test_case=test_case)

        test_input = test_case.get("test_case", None)

        if test_input is None:
            raise ValueError(
                "Input data for `test_big_float_variant` can`t be empty")

        expected_output = test_case.get("context_test_output", None)

        if expected_output is None:
            raise ValueError(
                "Expected output for `test_big_float_variant` can`t be empty")

        if test_input and not isinstance(test_input, dict):
            with raises(TypeError) as type_err:

                self.feature_encoder.encode_context(
                    context=test_input, noise=self.noise)
            assert os.getenv("V6_FEATURE_ENCODER_CONTEXT_TYPEERROR_MSG") \
                   in str(type_err.value)
            return

        assert False

    def test_small_float_variant(self):
        self._generic_test_encode_variant_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_SMALL_FLOAT_JSON"))

    def test_small_float_context(self):
        self._generic_test_encode_context_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_SMALL_FLOAT_JSON"))

    def test_same_output_int_bool_1_variant(self):
        int_test_case_path = os.sep.join(
            [self.v6_test_data_directory,
             os.getenv('V6_FEATURE_ENCODER_TEST_INT_1_JSON')])

        int_test_case = \
            self._get_test_data(path_to_test_json=int_test_case_path)

        self._set_model_properties_from_test_case(test_case=int_test_case)

        int_test_input = int_test_case.get('test_case', None)

        if int_test_input is None:
            raise ValueError(
                "Test input for `test_same_output_int_bool_1_variant` is empty")

        int_output = \
            self.feature_encoder.encode_variant(
                variant=int_test_input, noise=self.noise)

        bool_test_case_path = os.sep.join(
            [self.v6_test_data_directory,
             os.getenv('V6_FEATURE_ENCODER_TEST_BOOL_TRUE_JSON')])

        bool_test_case = \
            self._get_test_data(path_to_test_json=bool_test_case_path)

        bool_test_input = bool_test_case.get('test_case', None)

        if bool_test_input is None:
            raise ValueError(
                "Test input for `test_same_output_int_bool_1_variant` is empty")

        bool_output = \
            self.feature_encoder.encode_variant(
                variant=bool_test_input, noise=self.noise)

        assert int_output == bool_output

    def test_same_output_int_bool_0_variant(self):

        int_test_case_path = os.sep.join(
            [self.v6_test_data_directory,
             os.getenv('V6_FEATURE_ENCODER_TEST_INT_0_JSON')])

        int_test_case = \
            self._get_test_data(path_to_test_json=int_test_case_path)

        self._set_model_properties_from_test_case(test_case=int_test_case)

        int_test_input = int_test_case.get('test_case', None)

        if int_test_input is None:
            raise ValueError(
                "Test input for `test_same_output_int_bool_0_variant` is empty")

        int_output = \
            self.feature_encoder.encode_variant(
                variant=int_test_input, noise=self.noise)

        bool_test_case_path = os.sep.join(
            [self.v6_test_data_directory,
             os.getenv('V6_FEATURE_ENCODER_TEST_BOOL_FALSE_JSON')])

        bool_test_case = \
            self._get_test_data(path_to_test_json=bool_test_case_path)

        bool_test_input = bool_test_case.get('test_case', None)

        if bool_test_input is None:
            raise ValueError(
                "Test input for `test_same_output_int_bool_0_variant` is empty")

        bool_output = \
            self.feature_encoder.encode_variant(
                variant=bool_test_input, noise=self.noise)

        assert int_output == bool_output

    def test_same_output_float_bool_1_variant(self):
        float_test_case_path = os.sep.join(
            [self.v6_test_data_directory,
             os.getenv('V6_FEATURE_ENCODER_TEST_FLOAT_1_JSON')])

        float_test_case = \
            self._get_test_data(path_to_test_json=float_test_case_path)

        self._set_model_properties_from_test_case(test_case=float_test_case)

        float_test_input = float_test_case.get("test_case", None)

        if float_test_input is None:
            raise ValueError("Test input is empty")

        float_output = \
            self.feature_encoder.encode_variant(
                variant=float_test_input, noise=self.noise)

        bool_test_case_path = os.sep.join(
            [self.v6_test_data_directory,
             os.getenv('V6_FEATURE_ENCODER_TEST_BOOL_TRUE_JSON')])

        bool_test_case = \
            self._get_test_data(path_to_test_json=bool_test_case_path)

        bool_test_input = bool_test_case.get("test_case", None)

        if bool_test_input is None:
            raise ValueError("Test input is empty")

        bool_output = \
            self.feature_encoder.encode_variant(
                variant=bool_test_input, noise=self.noise)

        assert float_output == bool_output

    def test_same_output_float_bool_0_variant(self):
        float_test_case_path = os.sep.join(
            [self.v6_test_data_directory,
             os.getenv('V6_FEATURE_ENCODER_TEST_FLOAT_0_JSON')])

        float_test_case = \
            self._get_test_data(path_to_test_json=float_test_case_path)

        self._set_model_properties_from_test_case(test_case=float_test_case)

        float_test_input = float_test_case.get("test_case", None)

        float_output = \
            self.feature_encoder.encode_variant(
                variant=float_test_input, noise=self.noise)

        bool_test_case_path = os.sep.join(
            [self.v6_test_data_directory,
             os.getenv('V6_FEATURE_ENCODER_TEST_BOOL_FALSE_JSON')])

        bool_test_case = \
            self._get_test_data(path_to_test_json=bool_test_case_path)

        bool_test_input = bool_test_case.get("test_case", None)

        if bool_test_input is None:
            raise ValueError("Test input is empty")

        bool_output = \
            self.feature_encoder.encode_variant(
                variant=bool_test_input, noise=self.noise)

        assert float_output == bool_output

    # Test all primitive dicts: "string", true, false, 0, 0.0, 1, 1.0, -1, -1.0
    def test_primitive_dict_big_float_variant(self):

        test_case_path = os.sep.join(
            [self.v6_test_data_directory,
             os.getenv("V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_BIG_FLOAT_JSON")])

        test_case = self._get_test_data(path_to_test_json=test_case_path)

        self._set_model_properties_from_test_case(test_case=test_case)

        test_input = test_case.get("test_case", None)

        if not test_input:
            raise ValueError(
                "Input data for `test_big_float_variant` can`t be empty")

        expected_output = test_case.get("variant_test_output", None)

        if not expected_output:
            raise ValueError(
                "Expected output for `test_big_float_variant` can`t be empty")

        expected_output = \
            dict((key, float(val)) for key, val in expected_output.items())

        tested_output = \
            self.feature_encoder.encode_variant(
                variant=test_input, noise=self.noise)

        assert expected_output == tested_output

    def test_primitive_dict_big_float_context(self):

        test_case_path = os.sep.join(
            [self.v6_test_data_directory,
             os.getenv("V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_BIG_FLOAT_JSON")])

        test_case = self._get_test_data(path_to_test_json=test_case_path)

        self._set_model_properties_from_test_case(test_case=test_case)

        test_input = test_case.get("test_case", None)

        if not test_input:
            raise ValueError(
                "Input data for `test_big_float_variant` can`t be empty")

        expected_output = test_case.get("context_test_output", None)

        if not expected_output:
            raise ValueError(
                "Expected output for `test_big_float_variant` can`t be empty")

        expected_output = \
            dict((key, float(val)) for key, val in expected_output.items())

        tested_output = \
            self.feature_encoder.encode_context(
                context=test_input, noise=self.noise)

        assert expected_output == tested_output

    def test_primitive_dict_big_int_negative_variant(self):
        self._generic_test_encode_variant_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_BIG_INT_NEGATIVE_JSON"))

    def test_primitive_dict_big_int_negative_context(self):
        self._generic_test_encode_context_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_BIG_INT_NEGATIVE_JSON"))

    def test_primitive_dict_big_int_positive_variant(self):
        self._generic_test_encode_variant_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_BIG_INT_POSITIVE_JSON"))

    def test_primitive_dict_big_int_positive_context(self):
        self._generic_test_encode_context_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_BIG_INT_POSITIVE_JSON"))

    def test_primitive_dict_bool_false_variant(self):
        self._generic_test_encode_variant_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_BOOL_FALSE_JSON"))

    def test_primitive_dict_bool_false_context(self):
        self._generic_test_encode_context_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_BOOL_FALSE_JSON"))

    def test_primitive_dict_bool_true_variant(self):
        self._generic_test_encode_variant_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_BOOL_TRUE_JSON"))

    def test_primitive_dict_bool_true_context(self):
        self._generic_test_encode_context_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_BOOL_TRUE_JSON"))

    def test_primitive_dict_float_0_variant(self):
        self._generic_test_encode_variant_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_FLOAT_0_JSON"))

    def test_primitive_dict_float_0_context(self):
        self._generic_test_encode_context_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_FLOAT_0_JSON"))

    def test_primitive_dict_float_1_variant(self):
        self._generic_test_encode_variant_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_FLOAT_1_JSON"))

    def test_primitive_dict_float_1_context(self):
        self._generic_test_encode_context_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_FLOAT_1_JSON"))

    def test_primitive_dict_float_m1_variant(self):
        self._generic_test_encode_variant_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_FLOAT_M1_JSON"))

    def test_primitive_dict_float_m1_context(self):
        self._generic_test_encode_context_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_FLOAT_M1_JSON"))

    def test_primitive_dict_foo_bar_variant(self):
        self._generic_test_encode_variant_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_FOO_BAR_JSON"))

    def test_primitive_dict_foo_bar_context(self):
        self._generic_test_encode_context_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_FOO_BAR_JSON"))

    def test_foo_bar_dict_equals_list_variant(self):

        foo_bar_list_test_case_path = os.sep.join(
            [self.v6_test_data_directory,
             os.getenv("V6_FEATURE_ENCODER_TEST_FOO_BAR_JSON")])

        list_test_case = \
            self._get_test_data(path_to_test_json=foo_bar_list_test_case_path)

        self._set_model_properties_from_test_case(test_case=list_test_case)

        list_input = list_test_case.get("test_case", None)
        
        if list_input is None:
            raise ValueError("foo bar list input empty")

        foo_bar_dict_test_case_path = os.sep.join(
            [self.v6_test_data_directory,
             os.getenv("V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_FOO_BAR_JSON")])

        dict_test_case = \
            self._get_test_data(path_to_test_json=foo_bar_dict_test_case_path)

        dict_input = dict_test_case.get("test_case", None)

        if dict_input is None:
            raise ValueError("foo bar dict input empty")

        list_output = \
            self.feature_encoder.encode_variant(
                variant=list_input, noise=self.noise)

        dict_output = \
            self.feature_encoder.encode_variant(
                variant=dict_input, noise=self.noise)

        assert list_output == dict_output

    def test_primitive_dict_int_0_variant(self):
        self._generic_test_encode_variant_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_INT_0_JSON"))

    def test_primitive_dict_int_0_context(self):
        self._generic_test_encode_context_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_INT_0_JSON"))

    def test_primitive_dict_int_1_variant(self):
        self._generic_test_encode_variant_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_INT_1_JSON"))

    def test_primitive_dict_int_1_context(self):
        self._generic_test_encode_context_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_INT_1_JSON"))

    def test_primitive_dict_int_m1_variant(self):
        self._generic_test_encode_variant_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_INT_M1_JSON"))

    def test_primitive_dict_int_m1_context(self):
        self._generic_test_encode_context_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_INT_M1_JSON"))

    def test_primitive_dict_small_float_variant(self):
        self._generic_test_encode_variant_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_SMALL_FLOAT_JSON"))

    def test_primitive_dict_small_float_context(self):
        self._generic_test_encode_context_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_SMALL_FLOAT_JSON"))

    def test_primitive_dict_string_variant(self):
        self._generic_test_encode_variant_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_STRING_JSON"))

    def test_primitive_dict_string_context(self):
        self._generic_test_encode_context_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_PRIMITIVE_DICT_STRING_JSON"))

    def test_nested_list_variant(self):
        self._generic_test_encode_variant_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_NESTED_LIST_JSON"))

    def test_nested_list_context(self):
        self._generic_test_encode_context_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_NESTED_LIST_JSON"))

    def test_nested_dict_string_keys_variant(self):
        self._generic_test_encode_variant_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_NESTED_DICT_STRING_KEYS_JSON"))

    def test_nested_dict_string_keys_context(self):
        self._generic_test_encode_context_from_json_data(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_NESTED_DICT_STRING_KEYS_JSON"))

    # def _convert_keys_to_int_or_float(self, input_dict: dict) -> dict:
    #
    #     input_with_numeric_keys = {}
    #
    #     for key, value in input_dict.items():
    #
    #         if isinstance(value, dict):
    #             value = self._convert_keys_to_int_or_float(input_dict=value)
    #
    #         if key == "$value":
    #             input_with_numeric_keys[key] = value
    #         else:
    #             np.random.seed(self.encoder_seed)
    #             converter = np.random.choice([int, float])
    #             try:
    #                 numeric_key = converter(key)
    #             except:
    #                 numeric_key = key
    #
    #             input_with_numeric_keys[numeric_key] = value
    #
    #     return input_with_numeric_keys
    #
    # def test_nested_dict_mixed_keys_variant(self):
    #     input_path = os.sep.join(
    #         [self.v6_test_data_directory,
    #          os.getenv(
    #              "V6_FEATURE_ENCODER_TEST_NESTED_DICT_MIXED_KEYS_INPUT_JSON")])
    #
    #     test_input = \
    #         self._get_test_data(
    #             path_to_test_json=input_path, test_data_key="test_case")
    #
    #     numeric_keys_test_input = \
    #         self._convert_keys_to_int_or_float(input_dict=test_input)
    #
    #     expected_output_path = os.sep.join(
    #         [self.v6_test_data_directory,
    #          os.getenv(
    #              "V6_FEATURE_ENCODER_TEST_NESTED_DICT_MIXED_KEYS_OUTPUT_VARIANT_JSON")])
    #
    #     expected_output = self._get_test_data(
    #         path_to_test_json=expected_output_path,
    #         test_data_key="test_output")
    #
    #     tested_output = \
    #         self.feature_encoder.encode_variant(
    #             variant=numeric_keys_test_input, noise=self.noise)
    #
    #     assert expected_output == tested_output
    #
    # def test_nested_dict_mixed_keys_context(self):
    #
    #     input_path = os.sep.join(
    #         [self.v6_test_data_directory,
    #          os.getenv(
    #              "V6_FEATURE_ENCODER_TEST_NESTED_DICT_MIXED_KEYS_INPUT_JSON")])
    #
    #     test_input = \
    #         self._get_test_data(
    #             path_to_test_json=input_path, test_data_key="test_case")
    #
    #     numeric_keys_test_input = \
    #         self._convert_keys_to_int_or_float(input_dict=test_input)
    #
    #     expected_output_path = os.sep.join(
    #         [self.v6_test_data_directory,
    #          os.getenv(
    #              "V6_FEATURE_ENCODER_TEST_NESTED_DICT_MIXED_KEYS_OUTPUT_CONTEXT_JSON")])
    #
    #     expected_output = self._get_test_data(
    #         path_to_test_json=expected_output_path,
    #         test_data_key="test_output")
    #
    #     tested_output = \
    #         self.feature_encoder.encode_context(
    #             context=numeric_keys_test_input, noise=self.noise)
    #
    #     assert expected_output == tested_output

    def test_foo_bar_dict_variant(self):

        # self._generic_test_encode_variant_from_json_data(
        #     test_case_filename=os.getenv(
        #         "V6_FEATURE_ENCODER_TEST_DICT_FOO_BAR_JSON"))

        # TODO verify if this is desired test case
        test_input = {"\0\0\0\0\0\0\0\0": "foo", "\0\0\0\0\0\0\0\1": "bar"}

        test_case_path = os.sep.join(
            [self.v6_test_data_directory,
             os.getenv(
                 "V6_FEATURE_ENCODER_TEST_DICT_FOO_BAR_JSON")])

        test_case = self._get_test_data(path_to_test_json=test_case_path)

        self._set_model_properties_from_test_case(test_case=test_case)

        expected_output = test_case.get("variant_test_output", None)

        if expected_output is None:
            raise ValueError("Expected output is empty")

        tested_output = \
            self.feature_encoder.encode_variant(
                variant=test_input, noise=self.noise)

        assert expected_output == tested_output

    def test_foo_bar_dict_context(self):

        # self._generic_test_encode_context_from_json_data(
        #     test_case_filename=os.getenv(
        #         "V6_FEATURE_ENCODER_TEST_DICT_FOO_BAR_JSON"))

        test_input = {"\0\0\0\0\0\0\0\0": "foo", "\0\0\0\0\0\0\0\1": "bar"}

        test_case_path = os.sep.join(
            [self.v6_test_data_directory,
             os.getenv(
                 "V6_FEATURE_ENCODER_TEST_DICT_FOO_BAR_JSON")])

        test_case = self._get_test_data(path_to_test_json=test_case_path)

        self._set_model_properties_from_test_case(test_case=test_case)

        expected_output = test_case.get("context_test_output", None)

        if expected_output is None:
            raise ValueError("Expected output is empty")

        tested_output = \
            self.feature_encoder.encode_context(
                context=test_input, noise=self.noise)

        assert expected_output == tested_output

    # TODO test encoding with single context
    def test_batch_variants_encoding_with_single_context(self):
        self._generic_test_batch_input_encoding(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_BATCH_ENCODING_JSONLINES"),
            expected_output_data_key="single_context_test_output",
            single_context_encoding=True, data_read_method='read')

    # TODO test encoding with multiple contexts
    def test_batch_variants_encoding_with_multiple_contexts(self):
        self._generic_test_batch_input_encoding(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_BATCH_ENCODING_JSONLINES"),
            expected_output_data_key="multiple_context_test_output",
            single_context_encoding=False, data_read_method='read')

    def test_batch_jsonlines_encoding(self):
        self._generic_test_batch_input_encoding(
            test_case_filename=os.getenv(
                "V6_FEATURE_ENCODER_TEST_BATCH_ENCODING_JSONLINES"),
            expected_output_data_key="plain_jsonlines_test_output",
            single_context_encoding=False, plain_jsonlines_encoding=True,
            data_read_method='read')

    # TODO test missing features filler method
    def test_missing_features_filler_method(self):
        test_case_path = os.sep.join(
            [self.v6_test_data_directory,
             os.getenv("V6_FEATURE_ENCODER_TEST_BATCH_FILLING_MISSING_FEATURES")])

        test_case = \
            self._get_test_data(path_to_test_json=test_case_path, method="read")

        self._set_model_properties_from_test_case(test_case=test_case)

        test_jsonlines = test_case.get("test_case", None)

        if test_jsonlines is None:
            raise ValueError("Test input empty")

        test_variants = np.array([jl["variant"] for jl in test_jsonlines])
        test_contexts = [jl.get("context", {}) for jl in test_jsonlines][0]

        expected_output = test_case.get("test_output", None)

        if expected_output is None:
            raise ValueError("Expected output is empty")

        encoded_variants = \
            self.feature_encoder.encode_variants(
                variants=test_variants, contexts=test_contexts,
                noise=self.noise)

        missings_filled_array = self.feature_encoder.fill_missing_features(
            encoded_variants=encoded_variants, feature_names=self.feature_names)

        np.testing.assert_array_equal(expected_output, missings_filled_array)

