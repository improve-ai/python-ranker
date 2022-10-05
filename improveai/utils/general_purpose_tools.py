from codecs import decode
from copy import deepcopy
import datetime
from ksuid import Ksuid
import numpy as np

ALLOWED_VARIANT_TYPES = [list, tuple, np.ndarray]


def constant(f) -> property:
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


def append_prefix_to_dict_keys(input_dict: dict, prfx: str) -> dict:
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
        all_des_keys: list, imputed_dict, imputer_value: float = np.nan) -> dict:
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

    processed_encoded_features = deepcopy(imputed_dict)

    for feat_n in all_des_keys:
        if feat_n not in imputed_dict.keys():
            processed_encoded_features[feat_n] = imputer_value

    return processed_encoded_features


def read_jsonstring_from_file(
        path_to_file: str, mode: str = 'r', method: str = 'readlines',
        decode_escape_char: bool = False) -> str:
    """
    Safely reads desired file with json wrapper

    Parameters
    ----------
    path_to_file: str
        pth to loaded file
    mode: str
        read mode
    decode_escape_char: bool
        should double escape be repalced by single ?

    Returns
    -------
    str
        read json string

    """

    with open(path_to_file, mode) as rf:
        if method == 'readlines':
            contents = rf.readlines()
        elif method == 'read':
            contents = rf.read()
        else:
            raise ValueError('Wrong reading method value: {}'.format(method))

    if isinstance(contents, list):
        contents = ''.join(contents)

    if decode_escape_char:
        contents = decode(contents, 'unicode_escape')

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


def check_variants(variants: list or tuple or np.ndarray) -> list or tuple or np.ndarray:
    """
    Check if variants are of desired type and if they are not an empty collection

    Parameters
    ----------
    variants: list or tuple or np.ndarray
        checked variants

    """
    # following checks take only less than 0,001 ms
    assert variants is not None
    assert type(variants) in ALLOWED_VARIANT_TYPES
    # raise if variants are an empty list
    if len(variants) == 0:
        raise ValueError('`variants` must not be an empty collection')


def is_valid_variants_type(variants):
    return type(variants) in ALLOWED_VARIANT_TYPES


def check_variant_map(variant_map: dict):
    # make sure variant map is a dict
    assert isinstance(variant_map, dict)

    # make sure that variant map is not empty
    assert len(variant_map) > 0

    # make sure all collections of variants are of a correct type
    [check_variants(vs) for vs in variant_map.values()]


def get_variants_from_args(variants: list or tuple or np.ndarray) -> list or tuple or np.ndarray:
    """
    Extract variants from pythonic args

    Parameters
    ----------
    variants: list or tuple or np.ndarray


    Returns
    -------

    """

    if len(variants) == 1:
        # if the first element is a list -> it is ok to use such list as variants
        # if the first element is not a list / tuple / np.ndarray raise
        assert isinstance(variants[0], list) or isinstance(variants[0], tuple) or isinstance(variants[0], np.ndarray)
        return variants[0]

    return [v if not is_object_numpy_type(v) else v.item() for v in variants]


def is_valid_ksuid(id_: str) -> bool:
    """
    Checks if input value is a valid Ksuid string

    Parameters
    ----------
    id_: str
        checked string

    Returns
    -------
    bool
        True if string is a valid ksuid otherwise False

    """
    if not isinstance(id_, str):
        return False

    if len(id_) != 27:
        return False

    try:
        # Disallow KSUIDs from the future, otherwise it could severely hurt
        # the performance of the partitions by creating a huge partition in the future
        # that new records keep aggregating into. At some point that partition would
        # no longer fit in RAM and processing could seize.
        if Ksuid.from_base62(id_).datetime > datetime.datetime.now(datetime.timezone.utc):
            return False
    except:
        # there was an exception parsing the KSUID, fail
        return False

    return True


def is_object_numpy_type(checked_object: object):
    """
    Check if a passed input object is a numpy type or not

    Parameters
    ----------
    checked_object: object
        object to be checked for numpy type

    Returns
    -------
    bool
        True if object is of a nupy type False otherwise

    """
    if type(checked_object).__module__ == np.__name__:
        return True
    return False
