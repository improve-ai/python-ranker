import json
import numpy as np
import os
from pytest import fixture
import sys
from unittest import TestCase

sys.path.append(
    os.sep.join(str(os.path.abspath(__file__)).split(os.sep)[:-3]))

from improveai import FeatureEncoder
from improveai.chooser import XGBChooser
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

    def _get_test_data(
            self, path_to_test_json: str, method: str = 'readlines') -> object:

        loaded_jsonstring = read_jsonstring_from_file(
                    path_to_file=path_to_test_json, method=method)

        loaded_json = json.loads(loaded_jsonstring)

        return loaded_json

    def _set_chooser_properties_from_test_case(self, test_case_json: dict):
        # set model_seed
        self.encoder_seed = test_case_json.get("model_seed", None)
        assert self.encoder_seed is not None

        self.encoder_seed = int(self.encoder_seed)

        self.feature_names = test_case_json.get("feature_names", None)
        assert self.feature_names is not None

        self.string_tables = test_case_json.get("string_tables", None)
        assert self.string_tables is not None

        self.noise = test_case_json.get("noise", None)
        assert self.noise is not None

        self.xgb_chooser = XGBChooser()
        self.xgb_chooser.imposed_noise = self.noise
        self.xgb_chooser.feature_encoder = FeatureEncoder(
            feature_names=self.feature_names, string_tables=self.string_tables,
            model_seed=self.encoder_seed)

    def _generic_test_batch_input_encoding(
            self, test_case_filename: str, input_data_key: str = 'test_case',
            expected_output_data_key: str = 'expected_output',
            candidates_key: str = 'candidates', context_key: str = 'context',
            data_read_method: str = 'read', candidates_to_numpy: bool = False):

        self.python_specific_data_directory = \
            os.getenv("FEATURE_ENCODER_TEST_PYTHON_SPECIFIC_JSONS_DIR")

        test_case_path = os.sep.join(
            [self.python_specific_data_directory, test_case_filename])

        test_case_json = \
            self._get_test_data(
                path_to_test_json=test_case_path, method=data_read_method)

        self._set_chooser_properties_from_test_case(test_case_json=test_case_json)

        test_case = test_case_json.get(input_data_key, None)
        assert test_case is not None

        test_candidates = test_case.get(candidates_key, None)
        assert test_candidates is not None

        if candidates_to_numpy:
            test_candidates = np.array(test_candidates)

        test_context = test_case.get(context_key, None)
        assert test_context

        expected_output = test_case_json.get(expected_output_data_key, None)
        assert expected_output is not None

        tested_output_float64 = \
            self.xgb_chooser.encode_candidates_single_context(
                candidates=test_candidates, context=test_context)

        tested_output_float32 = convert_values_to_float32(val=tested_output_float64)

        for el in tested_output_float32:
            print('[' + ', '.join([str(n) for n in el]) + ']')

        expected_output_float32 = convert_values_to_float32(expected_output)

        np.testing.assert_array_equal(expected_output_float32, tested_output_float32)

    def test_batch_variants_encoding_with_single_context(self):
        self._generic_test_batch_input_encoding(
            test_case_filename=os.getenv(
                "CHOOSERS_FEATURE_ENCODER_TEST_BATCH_ENCODING_JSONLINES"),
            data_read_method='read')

    def test_batch_variants_encoding_with_single_given_and_numpy(self):
        # done for 100% coverage
        orig_use_cython_backend = improve_settings.CYTHON_BACKEND_AVAILABLE
        improve_settings.CYTHON_BACKEND_AVAILABLE = False

        self._generic_test_batch_input_encoding(
            test_case_filename=os.getenv(
                "CHOOSERS_FEATURE_ENCODER_TEST_BATCH_ENCODING_JSONLINES"),
            expected_output_data_key="expected_output",
            data_read_method='read', candidates_to_numpy=True)

        improve_settings.CYTHON_BACKEND_AVAILABLE = orig_use_cython_backend
