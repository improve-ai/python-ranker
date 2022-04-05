import json
import os
import requests_mock as rqm
from tqdm import tqdm

from improveai import DecisionModel


MODELS_DIR = os.getenv('DECISION_MODEL_PREDICTORS_DIR')
TRACK_URL = os.getenv('DECISION_TRACKER_TEST_URL')
SYNTHETIC_MODELS_DIR = os.sep.join([MODELS_DIR, 'synthetic_models'])
SYNTHETIC_MODELS_TIEBREAKING_DIR = os.sep.join([MODELS_DIR, 'synthetic_models_tiebreaking'])
SDK_PATH = os.getenv('SDK_HOME_PATH', '/')


def test_sdk_against_all_synthetic_models():

    all_test_cases_dirs = os.listdir(SYNTHETIC_MODELS_DIR)

    for test_case_dir in tqdm(all_test_cases_dirs):

        test_case_json_path = \
            os.sep.join([SYNTHETIC_MODELS_DIR, test_case_dir, '{}.json'.format(test_case_dir)])
        with open(test_case_json_path, 'r') as tcf:
            test_case = json.loads(tcf.read())

        # load model
        dm = DecisionModel(model_name=None, track_url=TRACK_URL)\
            .load(os.sep.join([SDK_PATH, test_case['model_url']]))

        # dt = DecisionTracker(track_url=TRACK_URL)
        # dm.track_with(tracker=dt)

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
                decision = dm.given(givens=givens).choose_from(variants=variants)
                # np.random.seed(seed)
                dm.chooser.imposed_noise = noise
                chosen_variant = decision.get()
                scores = decision.scores

                assert chosen_variant == output['variant']
                assert all(abs(scores - output['scores']) < 2 ** -22)


def test_primitive_predicts_identical_with_primitive_dicts():
    test_case_dirs = \
        ['1000_numeric_variants_20_same_nested_givens_large_binary_reward',
         'primitive_variants_no_givens_binary_reward']
    for test_case_dir in test_case_dirs:
        test_case_json_path = \
            os.sep.join([SYNTHETIC_MODELS_DIR, test_case_dir, '{}.json'.format(test_case_dir)])
        with open(test_case_json_path, 'r') as tcf:
            test_case = json.loads(tcf.read())

        # load model
        dm = DecisionModel(model_name=None, track_url=TRACK_URL)\
            .load(os.sep.join([SDK_PATH, test_case['model_url']]))

        all_givens = test_case['test_case']['givens']

        if all_givens is None:
            all_givens = [None]

        variants = test_case['test_case']['variants']
        dicts_variants = [{"$value": v} for v in test_case['test_case']['variants']]
        noise = test_case['test_case']['noise']
        seed = test_case['test_case']['python_seed']
        expected_outputs = test_case['expected_output']

        for givens, output in zip(all_givens, expected_outputs):
            with rqm.Mocker() as m:
                m.post(TRACK_URL, text='success')
                decision = dm.given(givens=givens).choose_from(variants=variants)
                # np.random.seed(seed)
                dm.chooser.imposed_noise = noise
                chosen_variant = decision.get()
                scores = decision.scores

                assert chosen_variant == output['variant']
                assert all(abs(scores - output['scores']) < 2 ** -22)

        for givens, output in zip(all_givens, expected_outputs):
            with rqm.Mocker() as m:
                m.post(TRACK_URL, text='success')
                decision = dm.given(givens=givens).choose_from(variants=dicts_variants)
                # np.random.seed(seed)
                dm.chooser.imposed_noise = noise
                chosen_variant = decision.get()
                scores = decision.scores

                assert chosen_variant == {"$value": output['variant']}
                assert all(abs(scores - output['scores']) < 2 ** -22)


def test_model_predicts_identical_for_nullish_variants():
    test_case_dir = 'primitive_variants_no_givens_binary_reward'

    test_case_json_path = \
        os.sep.join([SYNTHETIC_MODELS_DIR, test_case_dir, '{}.json'.format(test_case_dir)])

    with open(test_case_json_path, 'r') as tcf:
        test_case = json.loads(tcf.read())

    # load model
    dm = \
        DecisionModel(model_name=None, track_url=TRACK_URL)\
        .load(os.sep.join([SDK_PATH, test_case['model_url']]))
    print('### dm.tracker ###')
    print(TRACK_URL)
    print(dm.track_url)
    print(dm._tracker)

    # dt = DecisionTracker(track_url=TRACK_URL)
    # dm.track_with(tracker=dt)

    variants = [None, {}, [], {'$value': None}]
    noise = 0.1
    seed = 0

    with rqm.Mocker() as m:
        m.post(TRACK_URL, text='success')
        decision = dm.given(givens=None).choose_from(variants=variants)
        # np.random.seed(seed)
        dm.chooser.imposed_noise = noise
        # calling to calc scores
        decision.get()
        scores = decision.scores

        for bsc_index, bsc in enumerate(scores):
            for osc_index, osc in enumerate(scores):
                if bsc_index == osc_index:
                    continue
                assert abs(bsc - osc) < 2 ** -22

