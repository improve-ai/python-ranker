from copy import deepcopy
import json
import numpy as np
from time import time

from encoders.feature_encoder import FeatureEncoder
# from models.decision_models import DecisionModel
from xgboost import Booster, DMatrix


class CheckInClassVectorizatoin:
    def __init__(self):
        pass

    def sample_method(self, x, y):
        return '{} + {}'.format(x, y)


if __name__ == '__main__':

    # c = CheckInClassVectorizatoin()
    #
    # method_v = np.vectorize(pyfunc=c.sample_method, cache=True)
    # in_test = np.array(['a', 'b', 'c'])
    #
    # res = method_v(in_test, '1')
    # print(res)
    # input('sanity check')

    # model_kind = 'mlmodel'
    model_kind = 'xgb_native'
    # model_pth = '../artifacts/test_artifacts/'
    xgb_model_pth = '../artifacts/xgb_models/conv_model.xgb'
    # dm = DecisionModel(model_kind=model_kind, model_pth=xgb_model_pth)

    # context = frozendict({})
    with open('../artifacts/test_artifacts/sorting_context.json', 'r') as cjson:
        read_str = ''.join(cjson.readlines())
        context = json.loads(read_str)

    with open('../artifacts/data/real/meditations.json') as mjson:
        read_str = ''.join(mjson.readlines())
        variants = json.loads(read_str)

    # dm = DecisionModel(model_kind=model_kind, model_pth=xgb_model_pth)
    #
    # res = dm.score(variants=variants[:10], context=context)

    # print(res)
    raw_xgb_model_pth = '../artifacts/xgb_models/model.xgb'
    metadata_pth = '../artifacts/xgb_models/model.json'
    b = Booster()
    b.load_model(xgb_model_pth)

    with open(metadata_pth, 'r') as mdjson:
        read_str = ''.join(mdjson.readlines())
        model_metadata = json.loads(read_str)

    table = model_metadata['table']
    model_seed = model_metadata['model_seed']

    # print(model_metadata)

    fe = FeatureEncoder(table=table, model_seed=model_seed)

    total_enc_time = 0

    variants_count = len(variants)  # 900

    # input('sanity check')

    def get_enc_feat(variant, context):
        context_copy = deepcopy(context)
        enc_variant = fe.encode_features({'variant': variant})
        context_copy.update(enc_variant)
        return context_copy

    vectorized_fe = np.vectorize(pyfunc=get_enc_feat, cache=True)

    # enc_context = fe.encode_features({'context': context})
    # start_time = time()
    # res = vectorized_fe(np.array(variants[:variants_count]), enc_context)
    # end_time = time() - start_time
    # print(end_time / variants_count)
    # print(res[:3])
    # res[0][30] = -1
    # print(res[:3])
    # input('sanity check')

    def append_prefix(input_str, prfx):
        return '{}{}'.format(prfx, input_str)

    append_prefix_vectorized = np.vectorize(pyfunc=append_prefix, otypes=[str])

    def fill_nan(feat_n, enc_feats):
        feat_val = enc_feats.get(feat_n, None)
        if feat_val is None:
            return np.nan
        return feat_val

    fill_nan_vectorized = \
        np.vectorize(
            pyfunc=fill_nan, cache=True, otypes=[float], excluded=['enc_feats'])

    def get_batch_score(variant, context, feat_count):

        context_copy = deepcopy(context)
        enc_variant = fe.encode_features({'variant': variant})
        context_copy.update(enc_variant)

        # feat_count = max(all_feat_names_c.shape)

        # missings_filled_v = \
        #     fill_nan_vectorized(
        #         np.arange(0, feat_count, 1), context_copy)\
        #     .reshape((1, feat_count))

        missings_filled_v = np.array([
            context_copy[el] if context_copy.get(el, None) is not None
            else np.nan for el in np.arange(0, feat_count, 1)])\
            .reshape(1, feat_count)

        # print(missings_filled_v)
        # input('sanity check')

        return missings_filled_v

    get_batch_score_v = \
        np.vectorize(
            pyfunc=get_batch_score, cache=True, otypes=[np.ndarray],
            excluded=set(['context', 'all_feat_names_c']),
            signature='(),(),(1,n)->(1,n)'
        )

    # start_time = time()
    # trials = 1000
    # for _ in range(trials):
    #     all_feat_names_c = \
    #         np.core.defchararray.add('f', np.arange(len(table[1])).astype(str))
    # end_time = time()
    # print((end_time - start_time) / trials)
    #
    # start_time = time()
    # trials = 1000
    # for _ in range(trials):
    #     all_feat_names_c = \
    #         np.array(['f{}'.format(el) for el in range(len(table[1]))])
    # end_time = time()
    # print((end_time - start_time) / trials)
    #
    # input('nparr vect vs list compr')

    enc_context = fe.encode_features({'context': context})

    start_time = time()
    # all_feat_names_c = \
    #     append_prefix_vectorized(np.arange(0, len(table[1])), 'f')
    # this is faster
    batch_size = 500
    for _ in range(batch_size):
        all_feat_names_c = np.array(['f{}'.format(el) for el in range(len(table[1]))])
        # print(np.array(variants[:variants_count]).reshape(-1, 1).shape)
        # print(all_feat_names_c.reshape(-1, 1).shape)
        # print(variants_count)
        # input('sanity check')
        # res = \
        #     get_batch_score_v(
        #         np.array(variants[:variants_count]).reshape(-1, 1),
        #         enc_context,
        #         all_feat_names_c.reshape(1, -1)) \
        #     .reshape((variants_count, len(all_feat_names_c)))
        res = \
            np.array([get_batch_score(
                variant=v, context=enc_context,
                feat_count=max(all_feat_names_c.shape))
                for v in variants[:variants_count]])\
            .reshape((variants_count, len(all_feat_names_c)))
        # print(res.shape)
        # input('sanity check')
        preds = b.predict(DMatrix(res, feature_names=all_feat_names_c))
        preds[::-1].sort()
        # scores = b.predict(DMatrix(np.array(res.values())))
    end_time = time() - start_time
    print(end_time / batch_size)
    print(res.shape)
    # print(res[:10])
    # print(res.reshape((299, 44)).shape)
    # print(res.reshape(299, 44))
    # print(type(res))
    # print(res.flatten())
    # print(res[0].shape)
    # print(end_time)
    input('sanity check')

    for v_idx, _ in enumerate(variants[:variants_count]):
        start_time = time()
        enc_context = fe.encode_features({'context': context})
        enc_feats = fe.encode_features({'variant': variants[v_idx]})
        enc_context.update(enc_feats)
        total_enc_time += time() - start_time

        enc_feats = \
            dict(
                zip(['f{}'.format(k) for k in enc_context.keys()],
                    enc_context.values()))

        # print(enc_feats)
        # input('sanity check')

        all_feat_names_c = len(table[1])
        # print(all_feat_names_c)
        pred_array = []
        # print(type(enc_feats))

        for f_idx in range(all_feat_names_c):
            # print('f{}'.format(f_idx))
            curr_feat = enc_feats.get('f{}'.format(f_idx), None)
            # print(curr_feat)
            if curr_feat is None:
                curr_feat = np.nan
            pred_array.append(curr_feat)

        # print(pred_array)

        # print(enc_context)
        print(
            b.predict(
                DMatrix(
                    np.array(pred_array).reshape(1, all_feat_names_c),
                    feature_names=['f{}'.format(el) for el in range(all_feat_names_c)])))

    print(total_enc_time / variants_count)
