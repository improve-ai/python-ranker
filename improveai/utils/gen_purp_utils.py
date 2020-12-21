from collections.abc import Iterable
from copy import deepcopy
from frozendict import frozendict
import numpy as np


def constant(f):
    """
    Constant decorator raising error on attempt to set decorated value

    Parameters
    ----------
    f: callable
        decorated object

    Returns
    -------
    property
        constant property

    """
    def fset(self, value):
        raise AttributeError

    def fget(self):
        return f()

    return property(fget, fset)


def append_prfx_to_dict_keys(input_dict: dict, prfx: str) -> dict:
    """
    Appends prefix to the input dict keys

    Parameters
    ----------
    input_dict: dict
        dict which keys will be appended
    prfx: str
        prf to append to input dict keys

    Returns
    -------
    dict
        dict with keys appended

    """
    return dict(zip(
        ['{}{}'.format(prfx, key) for key in input_dict.keys()],
        list(input_dict.values())))


def impute_missing_dict_keys(
        all_des_keys: list, imputed_dict, imputer_value: float = np.nan):
    """
    Imputes provided value into missing keys of input dict

    Parameters
    ----------
    all_des_keys: list
        list of all necessary keys in returned dict
    imputed_dict: dict
        input dict
    imputer_value: object
        value to be imputed for missing values

    Returns
    -------
    dict
        dict with all missing keys imputed

    """

    prcsd_encoded_features = deepcopy(imputed_dict)

    for feat_n in all_des_keys:
        if feat_n not in imputed_dict.keys():
            prcsd_encoded_features[feat_n] = imputer_value

    return prcsd_encoded_features


def read_jsonstring_frm_file(pth_to_file: str, mode: str = 'r') -> str:
    """
    Safely reads desired file with json wrapper

    Parameters
    ----------
    pth_to_file: str
        pth to loaded file
    mode: str
        read mode

    Returns
    -------
    str
        read json string

    """
    with open(pth_to_file, mode) as rf:
        contents = rf.readlines()

    if isinstance(contents, list):
        contents = ''.join(contents)

    assert isinstance(contents, str)
    return contents


def sigmoid(x: float, logit_const: float) -> float:
    """
    Calculate sigmoid value

    Parameters
    ----------
    x: float
        arg passed to sigmoid function
    logit_const: float
        constant added to sigmoid argument

    Returns
    -------
    float
        value of sigmoid function

    """
    exp_arg = logit_const - x
    return 1 / (1 + np.exp(exp_arg))


def get_immutable_iterable(
        input_val: dict or frozendict or Iterable) -> Iterable or None:
    """
    Attempts to make Iterable objects immutable

    Parameters
    ----------
    input_val: dict or frozendict or Iterable
        value to be checked and converted to immutable collection

    Returns
    -------
    Iterable
        Immutable collection

    """

    assert not isinstance(input_val, str)

    if input_val is None:
        return None

    conv_new_val = input_val
    if isinstance(input_val, frozendict):
        pass
    elif isinstance(input_val, dict):
        conv_new_val = frozendict(input_val)
    elif isinstance(input_val, Iterable):
        # conv_new_val = new_val
        if not all([isinstance(variant, frozendict) for variant in input_val]):
            conv_new_val = \
                tuple([frozendict(variant) for variant in input_val])
        if not isinstance(conv_new_val, tuple):
            conv_new_val = tuple(conv_new_val)
    else:
        raise TypeError('Unsupported vairants type: {}'.format(type(input_val)))
    return conv_new_val
