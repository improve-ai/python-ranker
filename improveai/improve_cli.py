from argparse import ArgumentParser
import json
import os
import sys

sys.path.append(os.path.abspath('../'))

from improveai import Scorer, Ranker
from improveai.utils.general_purpose_tools import read_jsonstring_from_file


if __name__ == '__main__':

    ap = ArgumentParser()
    ap.add_argument(
        '--operation', default='score',
        help='Which action of 4 available should be performed: {}'.format(Ranker.SUPPORTED_CALLS))
    ap.add_argument('--model_url', help='Path to model file', required=True)
    ap.add_argument(
        '--candidates', help='Path to JSONfile with candidates or a JSON itself',
        default='[]')

    ap.add_argument(
        '--context',
        help='Path to JSONfile with context or JSON with givens itself',
        default='{}')
    ap.add_argument(
        '--results_path',
        help='Path to file where JSON with prediction results will be dumped',
        default='result_file.json')

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

    scorer = Scorer(model_url=pa.model_url)
    ranker = Ranker(scorer=scorer)

    if pa.operation not in Ranker.SUPPORTED_CALLS:
        raise ValueError(
            'CLI supported method_calls are {}. \n Provided method_calls: {}'
            .format(','.join(Ranker.SUPPORTED_CALLS), pa.operation))

    assert pa.operation in Ranker.SUPPORTED_CALLS, ValueError(
        'CLI supported method_calls are {}. \n Provided method_calls: {}'
        .format(','.join(Ranker.SUPPORTED_CALLS), pa.operation))

    candidates = json.loads(
        read_jsonstring_from_file(path_to_file=pa.candidates) if os.path.isfile(pa.candidates) else pa.candidates)
    context = json.loads(
        read_jsonstring_from_file(path_to_file=pa.context) if os.path.isfile(pa.context) else pa.context)

    scores = None
    if pa.operation == 'score':
        result = scorer.score(items=candidates, context=context).tolist()
    elif pa.operation == 'rank':
        scores = scorer.score(items=candidates, context=context)[::-1].tolist()
        result = list(ranker.rank(items=candidates, context=context))
    else:
        raise ValueError(f'Unable to perform operation: {pa.operation}')

    result_payload = result
    if pa.full_output:
        if pa.operation == 'score':
            result_payload = [{'item': k, 'score': v} for k, v in zip(candidates, result)]
        elif pa.operation == 'rank':
            result_payload = [{'item': k, 'score': v} for k, v in zip(result, list(reversed(sorted(scores))))]

    string_result = json.dumps(result_payload)

    if pa.prettify_json:
        try:
            import simplejson
        except ImportError as ierr:
            raise ImportError(
                'Please install `simplejson` package to mock track endpoint: pip3 install simplejson')
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
