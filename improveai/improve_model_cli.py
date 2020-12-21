from argparse import ArgumentParser
import json
import simplejson

from models.decision_models import DecisionModel
from utils.gen_purp_utils import read_jsonstring_frm_file


if __name__ == '__main__':

    DEFAULT_VARIANT = \
        json.dumps(
            [{"text": "Sanity check test 1"},
             {"text": "Sanity check test 2"},
             {"text": "Completely different sanity check"}])

    with open('artifacts/test_artifacts/context.json', 'r') as mj:
        context_str = mj.read()

    DEFAULT_CONTEXT = context_str

    ap = ArgumentParser()
    ap.add_argument(
        'method_call', default='score',
        help='Which action of 4 available should be performed '
             '(score, sort, choose)')
    ap.add_argument(
        'model_kind', default='mlmodel',
        help='String indicating which model type would be used: mlmodel, '
             'xgb_native or tf_lite model')
    ap.add_argument('model_pth', help='Path to model file')
    ap.add_argument('--variants', help='JSON with variants to be scored',
                    default=DEFAULT_VARIANT)
    ap.add_argument(
        '--variants_pth',
        help='Path to file with JSON with variants to be scored',
        default='')
    ap.add_argument(
        '--context', help='JSON with context', default=DEFAULT_CONTEXT)

    ap.add_argument(
        '--context_pth', help='Path to file with JSON with context',
        default='')
    ap.add_argument(
        '--results_pth',
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

    im = DecisionModel(model_kind=pa.model_kind, model_pth=pa.model_pth)

    if pa.method_call not in im.SUPPORTED_CALLS:
        raise ValueError(
            'CLI supported method_calls are {}. \n Provided method_calls: {}'
            .format(','.join(im.SUPPORTED_CALLS), pa.method_call))

    if pa.model_kind not in im.SUPPORTED_MODEL_KINDS:
        raise ValueError(
            'Currently supported model kinds are: {} \n. '
            'You have provided following model kind: {}'
            .format(','.join(im.SUPPORTED_MODEL_KINDS), pa.model_kind))

    assert hasattr(im, pa.method_call)

    # loading files
    input_objects = {
        'variants': pa.variants,
        'context': pa.context,
        'sigmoid_correction': pa.sigmoid_correction,
        'sigmoid_const': float(pa.sigmoid_const)}

    input_pths = [pa.variants_pth, pa.context_pth]

    for input_obj_key, input_pth in zip(input_objects.keys(), input_pths):
        if input_pth:
            input_objects[input_obj_key] = read_jsonstring_frm_file(input_pth)

    input_objects['cli_call'] = True

    des_method = getattr(im, pa.method_call)

    res = \
        des_method(**input_objects)

    output_col = 1 if pa.method_call == "score" else 0

    if not pa.prettify_json:
        res = \
            json.dumps([output[output_col] for output in json.loads(res)]) \
            if not pa.full_output else res
    else:
        res = \
            simplejson.dumps(
                [output[output_col] for output in json.loads(res)], indent=4) \
            if not pa.full_output \
            else simplejson.dumps(json.loads(res), indent=4)

    if pa.debug_print:
        print('\n##################################################################\n\n'
              'Result of method call: {} \non variants: {}\nwith model: {} \n\n'
              '##################################################################\n\n'
              'is: \n'
              '{}'
              .format(pa.method_call, pa.variants, pa.model_pth, res))

    if pa.results_pth:
        with open(pa.results_pth, 'w') as resf:
            resf.writelines(res)
