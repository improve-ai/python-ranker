import json
from time import time

from decisions.v6_fast import Decision
from models.decision_models import DecisionModel


if __name__ == '__main__':
    model_kind = 'xgb_native'
    # model_pth = '../artifacts/test_artifacts/'
    xgb_model_pth = '../artifacts/xgb_models/conv_model.xgb'
    dm = DecisionModel(model_kind=model_kind, model_pth=xgb_model_pth)

    # context = frozendict({})
    with open('../artifacts/test_artifacts/sorting_context.json', 'r') as cjson:
        read_str = ''.join(cjson.readlines())
        context = json.loads(read_str)

    with open('../artifacts/data/real/meditations.json') as mjson:
        read_str = ''.join(mjson.readlines())
        variants = json.loads(read_str)

    start_time = time()
    bench_size = 1000

    for _ in range(bench_size):
        d = Decision(
            variants=variants[:100], model=None, context=context)
        d.scores()
    total_time = time() - start_time
    print(total_time / bench_size)
    print(total_time)
