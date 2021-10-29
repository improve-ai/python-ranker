import numpy as np
import json
import os
import requests_mock as rqm
from tqdm import tqdm

from improveai import DecisionModel, DecisionTracker, Decision
from improveai.tests.test_utils import convert_values_to_float32

# os.environ['V6_DECISION_MODEL_PREDICTORS_DIR'] = '/home/kw/Projects/upwork/python-sdk/improveai/artifacts/models'
# os.environ['V6_DECISION_TRACKER_TEST_URL'] = 'http://test.v6.api/track'

MODELS_DIR = os.getenv('V6_DECISION_MODEL_PREDICTORS_DIR')
TRACK_URL = os.getenv('V6_DECISION_TRACKER_TEST_URL')
SYNTHETIC_MODELS_DIR = os.sep.join([MODELS_DIR, 'synthetic_models'])
SYNTHETIC_MODELS_TIEBREAKING_DIR = os.sep.join([MODELS_DIR, 'synthetic_models_tiebreaking'])
SDK_PATH = os.getenv('SDK_HOME_PATH', '/home/kw/Projects/upwork/python-sdk')


def test_sdk_against_all_synthetic_models():

    all_test_cases_dirs = os.listdir(SYNTHETIC_MODELS_DIR)

    for test_case_dir in tqdm(all_test_cases_dirs):

        test_case_json_path = \
            os.sep.join([SYNTHETIC_MODELS_DIR, test_case_dir, '{}.json'.format(test_case_dir)])
        with open(test_case_json_path, 'r') as tcf:
            test_case = json.loads(tcf.read())

        # load model
        dm = DecisionModel.load(os.sep.join([SDK_PATH, test_case['model_url']]))
        dt = DecisionTracker(track_url=TRACK_URL)
        dm.track_with(tracker=dt)

        all_givens = test_case['test_case']['givens']
        variants = test_case['test_case']['variants']
        noise = test_case['test_case']['noise']
        seed = test_case['test_case']['python_seed']
        expected_outputs = test_case['expected_output']

        # for each givens
        if all_givens is None:
            all_givens = [None]

        for givens, output in zip(all_givens, expected_outputs):
            with rqm.Mocker() as m:
                m.post(TRACK_URL, text='success')
                decision = \
                    Decision(decision_model=dm).choose_from(variants=variants) \
                    .given(givens=givens)
                np.random.seed(seed)
                chosen_variant = decision.get()
                scores = decision.scores

                assert chosen_variant == output['variant']
                float_32_scores = convert_values_to_float32(scores)
                float_32_expected_valeus = convert_values_to_float32(output['scores'])
                np.testing.assert_array_almost_equal(float_32_scores, float_32_expected_valeus, decimal=6)

                # try:
                #
                #     np.testing.assert_array_almost_equal(float_32_scores, float_32_expected_valeus, decimal=7)
                # except Exception as exc:
                #     errors[test_case_dir][json.dumps(givens)] = str(exc)
                #     print('### error ###')
                #     print('### test case: {} ###'.format(test_case_dir))
                #     print(exc)
                #     json.dumps(errors)
    # return errors


def test_sdk_against_all_synthetic_models_tiebreakers():
    all_test_cases_dirs = os.listdir(SYNTHETIC_MODELS_TIEBREAKING_DIR)

    for test_case_dir in tqdm(all_test_cases_dirs):

        test_case_json_path = \
            os.sep.join([SYNTHETIC_MODELS_TIEBREAKING_DIR, test_case_dir, '{}.json'.format(test_case_dir)])
        with open(test_case_json_path, 'r') as tcf:
            test_case = json.loads(tcf.read())

        # load model
        dm = DecisionModel.load(os.sep.join([SDK_PATH, test_case['model_url']]))
        dt = DecisionTracker(track_url=TRACK_URL)
        dm.track_with(tracker=dt)

        all_givens = test_case['test_case']['givens']
        variants = test_case['test_case']['variants']
        noise = test_case['test_case']['noise']
        seed = test_case['test_case']['python_seed']
        expected_outputs = test_case['expected_output']

        # for each givens
        if all_givens is None:
            all_givens = [None]

        for givens, output in zip(all_givens, expected_outputs):
            with rqm.Mocker() as m:
                m.post(TRACK_URL, text='success')
                decision = \
                    Decision(decision_model=dm).choose_from(variants=variants) \
                    .given(givens=givens)
                np.random.seed(seed)
                chosen_variant = decision.get()
                scores = decision.scores

                assert chosen_variant == output['variant']
                float_32_scores = convert_values_to_float32(scores)
                float_32_expected_valeus = convert_values_to_float32(output['scores'])
                np.testing.assert_array_almost_equal(float_32_scores, float_32_expected_valeus, decimal=6)

                # try:
                #
                #     np.testing.assert_array_almost_equal(float_32_scores, float_32_expected_valeus, decimal=7)
                # except Exception as exc:
                #     errors[test_case_dir][json.dumps(givens)] = str(exc)
                #     print('### error ###')
                #     print('### test case: {} ###'.format(test_case_dir))
                #     print(exc)
                #     json.dumps(errors)
    # return errors


if __name__ == '__main__':
    _sdk_against_all_synthetic_models()
