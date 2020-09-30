from copy import deepcopy
import numpy as np


def constant(f):
    def fset(self, value):
        raise AttributeError

    def fget(self):
        return f()

    return property(fget, fset)


def append_prfx_to_dict_keys(input_dict: dict, prfx: str) -> dict:
    return dict(zip(
        ['{}{}'.format(prfx, key) for key in input_dict.keys()],
        list(input_dict.values())))


def impute_missing_dict_keys(
        all_des_keys: list, imputed_dict, imputer_value: float = np.nan):

    prcsd_encoded_features = deepcopy(imputed_dict)

    for feat_n in all_des_keys:
        if feat_n not in imputed_dict.keys():
            prcsd_encoded_features[feat_n] = imputer_value

    return prcsd_encoded_features
