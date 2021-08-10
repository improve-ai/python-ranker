from copy import deepcopy
import json
from numbers import Number
import numpy as np
import os
from pprint import pprint
from pytest import fixture, raises
import sys
from unittest import TestCase
import xgboost as xgb

sys.path.append(
    os.sep.join(str(os.path.abspath(__file__)).split(os.sep)[:-3]))

from improveai import FeatureEncoder
from improveai.choosers.xgb_chooser import BasicNativeXGBChooser
from improveai.utils.choosers_feature_encoding_tools import \
    encoded_variants_to_np
import improveai.settings as improve_settings
from improveai.utils.general_purpose_tools import read_jsonstring_from_file
from improveai.tests.test_utils import convert_values_to_float32


class TestChooserFeatureEncoding(TestCase):

    @property
    def chooser(self):
        return self._chooser

    @chooser.setter
    def chooser(self, value):
        self._chooser = value

    @fixture(autouse=True)
    def prepare_artifacts(self):
        # self.encoder_seed = int(os.getenv("V6_FEATURE_ENCODER_MODEL_SEED"))
        # self.noise_seed = int(os.getenv("V6_FEATURE_ENCODER_NOISE_SEED"))

        self.v6_test_suite_data_directory = \
            os.getenv("V6_FEATURE_ENCODER_TEST_SUITE_JSONS_DIR")

        self.v6_test_python_specific_data_directory = \
            os.getenv("V6_FEATURE_ENCODER_TEST_PYTHON_SPECIFIC_JSONS_DIR")
        # self.feature_encoder = FeatureEncoder(model_seed=self.encoder_seed)

        self.xgb_chooser = BasicNativeXGBChooser()
        xgb_path = os.getenv("V6_DUMMY_MODEL_PATH")
        self.xgb_chooser.load_model(input_model_src=xgb_path)

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

    def _generic_test_batch_input_encoding(
            self, test_case_filename: str, input_data_key: str = 'test_case',
            expected_output_data_key: str = 'single_givens_test_output',
            variant_key: str = 'variant', givens_key: str = 'givens',
            single_given_encoding: bool = True,
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
            np.array([jl.get(givens_key, {}) for jl in test_jsonlines])

        if single_given_encoding:
            test_givens = test_givens[0]

        expected_output = test_case.get(expected_output_data_key, None)

        if expected_output is None:
            raise ValueError("Expected output is empty")

        tested_output_float64 = \
            self.xgb_chooser._encode_variants_single_givens(
                variants=test_variants, givens=test_givens,
                noise=self.noise)

        tested_output_float32 = \
            convert_values_to_float32(val=tested_output_float64)

        # np.testing.assert_array_equal(expected_output, tested_output_float32)

        for v_ref, v_calc in zip(expected_output, tested_output_float32):
            assert v_ref == v_calc

    def test_batch_variants_encoding_with_single_given(self):
        self._generic_test_batch_input_encoding(
            test_case_filename=os.getenv(
                "V6_CHOOSERS_FEATURE_ENCODER_TEST_BATCH_ENCODING_JSONLINES"),
            expected_output_data_key="single_givens_test_output",
            single_given_encoding=True, data_read_method='read')

    def test_batch_variants_encoding_with_single_given_and_numpy(self):
        # done for 100% coverage
        orig_use_cython_backend = improve_settings.USE_CYTHON_BACKEND
        improve_settings.USE_CYTHON_BACKEND = False

        self._generic_test_batch_input_encoding(
            test_case_filename=os.getenv(
                "V6_CHOOSERS_FEATURE_ENCODER_TEST_BATCH_ENCODING_JSONLINES"),
            expected_output_data_key="single_givens_test_output",
            single_given_encoding=True, data_read_method='read')

        improve_settings.USE_CYTHON_BACKEND = orig_use_cython_backend

    def test_missing_features_filler_method_01(self):
        test_case_path = os.sep.join(
            [self.v6_test_python_specific_data_directory,
             os.getenv("V6_CHOOSERS_FEATURE_ENCODER_TEST_BATCH_FILLING_MISSING_FEATURES_01")])

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
            self.xgb_chooser._encode_variants_single_givens(
                variants=test_variants, givens=test_givens,
                noise=self.noise)

        missings_filled_array_float64 = encoded_variants_to_np(
            encoded_variants=encoded_variants,
            feature_names=self.xgb_chooser.model_feature_names)

        missings_filled_array_float32 = \
            convert_values_to_float32(val=missings_filled_array_float64)

        np.testing.assert_array_equal(
            expected_output, missings_filled_array_float32)

    def test_missing_features_filler_method_02(self):
        test_case_path = os.sep.join(
            [self.v6_test_python_specific_data_directory,
             os.getenv("V6_CHOOSERS_FEATURE_ENCODER_TEST_BATCH_FILLING_MISSING_FEATURES_02")])

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
            self.xgb_chooser._encode_variants_single_givens(
                variants=test_variants, givens=test_givens,
                noise=self.noise)

        missings_filled_array_float64 = encoded_variants_to_np(
            encoded_variants=encoded_variants,
            feature_names=self.xgb_chooser.model_feature_names)

        missings_filled_array_float32 = \
            convert_values_to_float32(val=missings_filled_array_float64)

        # pprint(missings_filled_array_float32.tolist())

        np.testing.assert_array_equal(
            expected_output, missings_filled_array_float32)

    def test_missing_features_filler_method_02_with_numpy(self):
        test_case_path = os.sep.join(
            [self.v6_test_python_specific_data_directory,
             os.getenv("V6_CHOOSERS_FEATURE_ENCODER_TEST_BATCH_FILLING_MISSING_FEATURES_02")])

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
            self.xgb_chooser._encode_variants_single_givens(
                variants=test_variants, givens=test_givens,
                noise=self.noise)

        # done for 100% coverage
        orig_use_cython_backend = improve_settings.USE_CYTHON_BACKEND
        improve_settings.USE_CYTHON_BACKEND = False

        missings_filled_array_float64 = encoded_variants_to_np(
            encoded_variants=encoded_variants,
            feature_names=self.xgb_chooser.model_feature_names)

        improve_settings.USE_CYTHON_BACKEND = orig_use_cython_backend

        # pprint(missings_filled_array.tolist())

        missings_filled_array_float32 = \
            convert_values_to_float32(val=missings_filled_array_float64)

        np.testing.assert_array_equal(
            expected_output, missings_filled_array_float32)
