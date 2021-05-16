from argparse import ArgumentParser
import json
import numpy as np
import os
import simplejson

from improveai.model import DecisionModel
from utils.general_purpose_utils import read_jsonstring_from_file


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
        '--sigmoid_const',
        help='Intercept for the sigmoid correction',
        default=0.5)

    ap.add_argument(
        '--full_output',
        action='store_true',
        help='Form of output - full vs short')

    ap.add_argument(
        '--debug_print',
        action='store_true',
        help='Printing results on screen')

    ap.add_argument(
        '--prettify_json',
        action='store_true',
        help='Pretty dump to JSON')

    pa = ap.parse_args()

    dm = DecisionModel.load(model_url=pa.model_url)

    if pa.operation not in DecisionModel.SUPPORTED_CALLS:
        raise ValueError(
            'CLI supported method_calls are {}. \n Provided method_calls: {}'
            .format(','.join(dm.SUPPORTED_CALLS), pa.method_call))

    if pa.operation != 'get':
        assert hasattr(dm, pa.operation)

    inputs = {}
    for key, value in zip(['variants', 'givens'], [pa.variants, pa.givens]):
        json_loads_value = value
        if os.path.isfile(value):
            # load jsonfile
            json_loads_value = read_jsonstring_from_file(path_to_file=value)
        inputs[key] = json.loads(json_loads_value)

    dm = DecisionModel.load(model_url=pa.model_url)
    if pa.operation == 'get':
        result = \
            dm.choose_from(variants=inputs['variants'])\
            .given(givens=inputs['givens']).get()
    else:

        scores = \
            dm.score(**inputs)

        if pa.operation == 'score':
            result = scores
        else:
            desired_operation = getattr(dm, pa.operation)
            inputs['scores'] = scores
            result = desired_operation(**inputs)

        if isinstance(result, np.ndarray):
            result = result.tolist()

    string_result = json.dumps(result)

    if pa.prettify_json:
        string_result = simplejson.dumps(json.loads(string_result), indent=4)

    if pa.debug_print:
        print('\n##################################################################\n\n'
              'Result of method call: {} \non variants: {}\nwith model: {} \n\n'
              '##################################################################\n\n'
              'is: \n'
              '{}'
              .format(pa.method_call, pa.variants, pa.model_pth, result))

    if pa.results_path:
        with open(pa.results_path, 'w') as resf:
            resf.writelines(string_result)
