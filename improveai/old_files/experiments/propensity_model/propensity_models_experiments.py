from gc import collect
import json
import numpy as np
import os
import pandas as pd
import sys
from time import time
from tqdm import tqdm
from xgboost import XGBClassifier

sys.path.append('/home/os/Projects/upwork/python-sdk/improveai')

from models.legacy.decision_models import DecisionModel

if __name__ == '__main__':

    variants_counts = [300]
    chkd_prfxs = ['not_weighted', 'weighted', 'appended']
    chkd_sfxs = ['uni', 'norm_m150_sd30', 'weib_rl5_a10']
    chkd_samples_counts = [100, 500, 1000, 2000, 4000, 5000, 10000, 20000, 30000]

    # get feature encoder
    xgb_model_url = 'https://improve-v5-resources-prod-models-117097735164.s3-us-west-2.amazonaws.com/models/mindful/latest/improve-messages-2.0.xgb.gz'

    dm = DecisionModel(model_kind='xgb_native', model_pth=xgb_model_url)
    feat_enc = dm.chooser.feature_encoder
    chooser = dm.chooser
    feats_count = chooser._get_features_count()

    all_feat_names = \
        np.array(['f{}'.format(el) for el in range(feats_count)])

    for variants_count in variants_counts:

        # read only variants
        with open(
                'data/synth_{}_variants.json'.format(variants_count), 'r') as svj:
            svj_str = svj.read()
            synth_variants = json.loads(svj_str)

        for data_prfx in chkd_prfxs:
            for data_suffix in chkd_sfxs:
                for samples_count in chkd_samples_counts:

                    if samples_count > 4000 \
                            and data_prfx in ['not_weighted', 'appended']:

                        print('NOT ENOUGH RAM TO PROCESS! SKIPPING!')
                        continue

                    read_start = time()
                    # read train data
                    with open(
                            'data/{}_data_{}_sc{}.json'.format(
                                data_prfx, data_suffix, samples_count), 'r') as nwjf:
                        nwjf_str = nwjf.read()
                        non_weighted_raw = json.loads(nwjf_str)

                    with open(
                            'data/{}_props_{}_sc{}.json'.format(
                                data_prfx, data_suffix, samples_count), 'r') as nwp:
                        nwp_str = nwp.read()
                        non_weighted_props_info = json.loads(nwp_str)

                    read_ends = time()

                    non_weighted_props = non_weighted_props_info['props']

                    prop_distr = data_suffix

                    train_starts = time()
                    # # get y
                    y = [el[1] for el in non_weighted_raw]
                    # # get weights
                    w = [el[2] for el in non_weighted_raw]
                    # # get variants
                    train_variants = [el[0] for el in non_weighted_raw]

                    dummy_context = {}

                    # enc_dummy_context = feat_enc.encode_features(context)
                    enc_dummy_context = {}

                    train_encoded_variants = \
                        np.array([chooser._get_nan_filled_encoded_variant(
                            variant=v, context=enc_dummy_context,
                            all_feats_count=feats_count, missing_filler=np.nan)
                            for v in tqdm(train_variants)])\
                        .reshape((len(train_variants), len(all_feat_names)))

                    # X = pd.DataFrame(train_encoded_variants)

                    xgbc = XGBClassifier(n_jobs=4)
                    xgbc.fit(X=train_encoded_variants, y=y, sample_weight=w)

                    train_ends = time()

                    eval_starts = time()

                    eval_encoded_variants = \
                        np.array([chooser._get_nan_filled_encoded_variant(
                            variant=v, context=enc_dummy_context,
                            all_feats_count=feats_count, missing_filler=np.nan)
                            for v in tqdm(synth_variants)]).reshape(
                            (len(synth_variants), len(all_feat_names)))

                    X_tst = pd.DataFrame(eval_encoded_variants)

                    test_preds = xgbc.predict_proba(X_tst)

                    eval_ends = time()

                    res_arr = [
                        [raw_v,
                         float(score[1]) if data_prfx != 'appended'
                         else float(score[1] / (1 - score[1])),
                         float(non_weighted_props[raw_v['text']])]
                        for score, raw_v in zip(test_preds, synth_variants)]

                    res_dict = {
                        'results_per_variant': res_arr,
                        'model_propensity_sum': sum(el[1] for el in res_arr),
                        'propensity_smape':
                            np.mean(
                                [abs(el[2] - el[1]) / (abs(el[1]) + abs(el[2]))
                                 if (abs(el[1]) + abs(el[2])) != 0 else 0
                                 for el in res_arr]) * 100,
                        'data_read_duration_mins': (read_ends - read_start) / 60,
                        'train_duration_mins': (train_ends - train_starts) / 60,
                        'eval_duration_mins': (eval_ends - eval_starts) / 60,
                        'total_duration_mins':
                            ((read_ends - read_start) + (train_ends - train_starts) +
                             (eval_ends - eval_starts)) / 60
                    }

                    dt_name_chunk = \
                        str(pd.datetime.now()).split('.')[0].replace('-', '')\
                        .replace(' ', '_').replace(':', '')
                    vlen_name_chunk = 'vc' + str(len(synth_variants))
                    samples_count_name_chunk = \
                        'sc' + str(samples_count)

                    exp_dirname = \
                        '{}-{}-{}-{}-{}'.format(
                            dt_name_chunk, data_prfx, vlen_name_chunk,
                            samples_count_name_chunk, prop_distr)

                    os.mkdir('results/{}'.format(exp_dirname))

                    with open(
                            'results/{}/results_summary.json'
                            .format(exp_dirname), 'w') as resj:
                        res_json_str = json.dumps(res_dict)
                        resj.write(res_json_str)

                    with open(
                            'results/{}/hash_table.json'
                            .format(exp_dirname), 'w') as tabj:
                        tab_json_str = json.dumps(feat_enc.table)
                        tabj.write(tab_json_str)

                    xgbc.save_model(
                        'results/{}/prop_model.xgb'.format(exp_dirname))

                    non_weighted_raw = None
                    train_encoded_variants = None
                    res_arr = None
                    res_dict = None
                    res_json_str = None
                    X = None
                    y = None
                    w = None
                    xgbc = None
                    collect()


