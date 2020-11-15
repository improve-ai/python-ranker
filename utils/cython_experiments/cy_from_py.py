import json
import numpy as np
import pyximport
from time import time

pyximport.install()

from choosers.choosers_cython_utils.fast_feat_enc import get_all_feat_names, \
    fast_get_nan_filled_encoded_variant, get_nan_filled_encoded_variants
from encoders.feature_encoder import FeatureEncoder
from models.decision_models import DecisionModel

if __name__ == '__main__':

    # model_kind = 'mlmodel'
    model_kind = 'xgb_native'
    # model_pth = '../artifacts/test_artifacts/'
    xgb_model_pth = '../../artifacts/models/12_11_2020_verses_conv.xgb'
    dm = DecisionModel(model_kind='xgb_native', model_pth=xgb_model_pth)
    # input('sanity check')

    # context = frozendict({})
    with open('../../artifacts/test_artifacts/sorting_context.json', 'r') as cjson:
        read_str = ''.join(cjson.readlines())
        context = json.loads(read_str)

    variant = {'text': 'Sample text to score'}
    with open('../../artifacts/data/real/2bs_bible_verses_full.json') as mjson:
        read_str = ''.join(mjson.readlines())
        variants = json.loads(read_str)

    all_feat_count = dm.chooser._get_features_count()
    encoded_context = \
        dm.chooser.feature_encoder.encode_features({'context': context})

    # print(encoded_context)

    # for _ in range(1000):

    st = time()
    # res = \
    #     [fast_get_nan_filled_encoded_variant(
    #         variant, encoded_context, all_feat_count, dm.chooser.feature_encoder)
    #      for v_idx in range(len(variants))]

    # res = \
    #     fast_get_nan_filled_encoded_variant(
    #         variant, encoded_context, all_feat_count,
    #         dm.chooser.feature_encoder)

    # print(np.array(variants[:10], dtype=dict).shape)
    #
    # print(type(all_feat_count))
    # input('abc')

    res = \
        np.asarray(get_nan_filled_encoded_variants(
            np.array(variants, dtype=dict), encoded_context, all_feat_count,
            dm.chooser.feature_encoder))

    et = time()
    print(et - st)
    print(np.nansum(res[:10]))

    print(encoded_context)

    # print(res[0,:].sum())
    input('sanity check')
    #
    arr_size = 3000
    a0 = np.arange(arr_size)
    print(a0[:10])
    # fast mode
    st = time()
    ac = np.asarray(get_all_feat_names(arr_size))
    # ac1 = sample_array_oper(ac)
    et = time()
    print(et - st)
    # print(ac[:10])
    # # slow mode
    # st1 = time()
    # ap = ['f{}'.format(el) for el in a0]
    # print(ap[:10])
    # et1 = time()
    # print(et1 - st1)
