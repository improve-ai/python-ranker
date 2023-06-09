from codecs import decode
from copy import deepcopy
import datetime
from ksuid import Ksuid
import numpy as np

ALLOWED_VARIANT_COLLECTION_TYPES = [list, tuple, np.ndarray]


def read_jsonstring_from_file(
        path_to_file: str, mode: str = 'r', method: str = 'readlines',
        decode_escape_char: bool = False) -> str:
    """
    Safely reads JSON string from desired file into JSON dict

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


def check_candidates(candidates: list or tuple or np.ndarray) -> list or tuple or np.ndarray:
    """
    Checks if items / candidates are of desired type and if they are not an empty collection

    Parameters
    ----------
    candidates: list or tuple or np.ndarray
        checked variants
    """
    # following checks take only less than 0,001 ms
    assert candidates is not None
    assert type(candidates) in ALLOWED_VARIANT_COLLECTION_TYPES
    # raise if variants are an empty list
    assert len(candidates) > 0, '`variants` must not be an empty collection'


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


def deepcopy_args(*args):
    """
    Helper function to deepcopy request body items before request. Requests are
    now asynchronous by default and not copying those values could yield strange
    results

    Parameters
    ----------
    args: list
        lsit of args to be copied

    Returns
    -------
    list
        list of copied args
    """
    return [deepcopy(arg) for arg in args]

