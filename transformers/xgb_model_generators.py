import json
from xgboost import Booster

if __name__ == '__main__':

    src_booster_pth = '../test_artifacts/model.xgb'
    orig_booster = Booster()
    orig_booster.load_model(src_booster_pth)

    # read metadata json
    with open('../test_artifacts/model.json', 'r') as mj:
        metadata_str = mj.read()
        model_metadata = json.loads(metadata_str)

    model_metadata_str = json.dumps({'json': model_metadata})

    # append metadata json
    orig_booster.set_attr(**{'user_defined_metadata': model_metadata_str})
    # orig_booster_binary_buffer = orig_booster.save_raw()
    # save model to file
    trgt_booster_pth = '../test_artifacts/model_appended.xgb'
    orig_booster.save_model(trgt_booster_pth)

    # with open(trgt_booster_pth, 'wb') as xgbw:
    #     xgbw.write(orig_booster_binary_buffer)

