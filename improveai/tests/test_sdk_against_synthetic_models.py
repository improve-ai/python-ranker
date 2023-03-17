import json
import os

import numpy as np
import requests_mock as rqm
from tqdm import tqdm

from improveai import Scorer


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

        # load scorer
        scorer = Scorer(model_url=os.sep.join([SDK_PATH, test_case['model_url']]))

        all_contexts = test_case['test_case']['contexts']
        candidates = test_case['test_case']['candidates']
        noise = test_case['test_case']['noise']
        seed = test_case['test_case']['python_seed']
        expected_outputs = test_case['expected_output']

        # for each context
        if all_contexts is None:
            all_contexts = [None]

        for context, output in zip(all_contexts, expected_outputs):
            scorer.chooser.imposed_noise = noise
            np.random.seed(seed)
            scores = scorer.score(items=candidates, context=context)
            chosen_item = candidates[np.argmax(scores)]

            assert chosen_item == output['item']
            assert all(abs(scores - output['scores']) < 2 ** -22)


# TODO this test no longer makes sense for a single model (due to feature names)
# def test_primitive_predicts_identical_with_primitive_dicts():
#     test_case_dirs = \
#         ['1000_numeric_variants_20_same_nested_givens_large_binary_reward',
#          'primitive_variants_no_givens_binary_reward']
#     for test_case_dir in test_case_dirs:
#         test_case_json_path = \
#             os.sep.join([SYNTHETIC_MODELS_DIR, test_case_dir, '{}.json'.format(test_case_dir)])
#         with open(test_case_json_path, 'r') as tcf:
#             test_case = json.loads(tcf.read())
#
#         # load model
#         scorer = Scorer(model_url=os.sep.join([SDK_PATH, test_case['model_url']]))
#
#         all_contexts = test_case['test_case']['contexts']
#
#         if all_contexts is None:
#             all_contexts = [None]
#
#         candidates = test_case['test_case']['candidates']
#         dicts_variants = [{"$value": v} for v in test_case['test_case']['candidates']]
#         noise = test_case['test_case']['noise']
#         seed = test_case['test_case']['python_seed']
#         expected_outputs = test_case['expected_output']
#
#         for context, output in zip(all_contexts, expected_outputs):
#
#             scorer.chooser.imposed_noise = noise
#             scores = scorer.score(items=candidates, context=context)
#             chosen_variant = candidates[np.argmax(scores)]
#
#             assert chosen_variant == output['item']
#             assert all(abs(scores - output['scores']) < 2 ** -22)
#
#         for context, output in zip(all_contexts, expected_outputs):
#
#             scorer.chooser.imposed_noise = noise
#             scores = scorer.score(items=dicts_variants, context=context)
#             chosen_variant = dicts_variants[np.argmax(scores)]
#
#             assert chosen_variant == {"$value": output['item']}
#             assert all(abs(scores - output['scores']) < 2 ** -22)


def test_model_predicts_identical_for_nullish_variants():
    test_case_dir = 'primitive_variants_no_givens_binary_reward'

    test_case_json_path = \
        os.sep.join([SYNTHETIC_MODELS_DIR, test_case_dir, '{}.json'.format(test_case_dir)])

    with open(test_case_json_path, 'r') as tcf:
        test_case = json.loads(tcf.read())

    scorer = Scorer(os.sep.join([SDK_PATH, test_case['model_url']]))

    candidates = [None, {}, [], {'$value': None}]
    noise = 0.1
    seed = 0

    scorer.chooser.imposed_noise = noise
    np.random.seed(seed)
    scores = scorer.score(items=candidates, context=None)

    for bsc_index, bsc in enumerate(scores):
        for osc_index, osc in enumerate(scores):
            if bsc_index == osc_index:
                continue
            assert abs(bsc - osc) < 2 ** -22

