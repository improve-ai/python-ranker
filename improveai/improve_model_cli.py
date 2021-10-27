from argparse import ArgumentParser
import json
import numpy as np
import os
import simplejson

from improveai.decision_model import DecisionModel
from utils.general_purpose_tools import read_jsonstring_from_file


if __name__ == '__main__':

    ap = ArgumentParser()
    ap.add_argument(
        '--operation', default='score',
        help='Which action of 4 available should be performed: {}'.format(
            DecisionModel.SUPPORTED_CALLS))
    ap.add_argument('--model_url', help='Path to model file', required=True)
    ap.add_argument(
        '--variants', help='Path to JSONfile with variants or a JSON itself',
        default='[]')

    ap.add_argument(
        '--givens',
        help='Path to JSONfile with givens or JSON with givens itself',
        default='{}')
    ap.add_argument(
        '--results_path',
        help='Path to file where JSON with prediction results will be dumped',
        default='result_file.json')
    ap.add_argument('--sigmoid_correction', action='store_true')

    ap.add_argument(
        '--debug_print',
        action='store_true',
        help='Printing results on screen')

    ap.add_argument(
        '--prettify_json',
        action='store_true',
        help='Pretty dump to JSON')

    ap.add_argument(
        '--full_output',
        action='store_true',
        help='dump to JSON all variants along with scores')

    pa = ap.parse_args()

    dm = DecisionModel.load(model_url=pa.model_url)

    if pa.operation not in DecisionModel.SUPPORTED_CALLS:
        raise ValueError(
            'CLI supported method_calls are {}. \n Provided method_calls: {}'
            .format(','.join(dm.SUPPORTED_CALLS), pa.operation))

    if pa.operation != 'get':
        assert hasattr(dm, pa.operation)

    # inputs = {}
    # for key, value in zip(['variants', 'givens'], [pa.variants, pa.givens]):
    #     json_loads_value = value
    #     if os.path.isfile(value):
    #         # load jsonfile
    #         json_loads_value = read_jsonstring_from_file(path_to_file=value)
    #     inputs[key] = json.loads(json_loads_value)

    variants = json.loads(
        read_jsonstring_from_file(path_to_file=pa.variants) if os.path.isfile(pa.variants) else pa.variants)
    givens = json.loads(
        read_jsonstring_from_file(path_to_file=pa.givens) if os.path.isfile(pa.givens) else pa.givens)

    dm = DecisionModel.load(model_url=pa.model_url)
    if pa.operation == 'get':
        result = \
            dm.choose_from(variants=variants)\
            .given(givens=givens).get()
    else:

        scores = \
            dm.score(variants=variants, givens=givens)

        if pa.operation == 'score':
            result = scores
        else:
            desired_operation = getattr(dm, pa.operation)
            # inputs['scores'] = scores
            result = desired_operation(**{'variants': variants, 'scores': scores})

        if isinstance(result, np.ndarray):
            result = result.tolist()

    if pa.full_output:
        if pa.operation == 'score':
            result_payload = [{'variant': k, 'score': v} for k, v in zip(variants, result)]
        elif pa.operation == 'rank':
            result_payload = [{'variant': k, 'score': v} for k, v in zip(result, list(reversed(sorted(scores))))]
        else:
            result_payload = result

    string_result = json.dumps(result_payload)

    if pa.prettify_json:
        string_result = simplejson.dumps(json.loads(string_result), indent=4)

    if pa.debug_print:
        print('\n##################################################################\n\n'
              'Result of method call: {} \non variants: {}\nwith model: {} \n\n'
              '##################################################################\n\n'
              'is: \n'
              '{}'
              .format(pa.operation, pa.variants, pa.model_url, result))

    if pa.results_path:
        with open(pa.results_path, 'w') as resf:
            resf.writelines(string_result)
