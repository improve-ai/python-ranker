import numpy as np
from pytest import raises
import string

from improveai.choosers import BasicNativeXGBChooser

BAD_SPECIAL_CHARACTERS = [el for el in '`~!@#$%^&*()=+[]{};:"<>,/?' + "'"]
ALNUM_CHARS = [el for el in string.digits + string.ascii_letters]


def test_none_model_name():
    chooser = BasicNativeXGBChooser()
    with raises(AssertionError) as aerr:
        chooser.model_name = None
        assert aerr.value


# - test that regexp compliant model name passes regexp
def test_good_model_name():
    good_model_names = \
        ['a', '0', '0-', 'a1-', 'x23yz_', 'a01sd.', 'abc3-xy2z_as4d.'] + \
        [''.join('a' for _ in range(64))]

    for good_model_name in good_model_names:
        chooser = BasicNativeXGBChooser()
        chooser.model_name = good_model_name


# - test that regexp non-compliant model name raises AssertionError
def test_bad_model_name():
    bad_model_names = \
        ['', '-', '_', '.', '-a1', '.x2z', '_x2z'] + \
        [''.join(
            np.random.choice(ALNUM_CHARS, 2).tolist() + [sc] +
            np.random.choice(ALNUM_CHARS, 2).tolist())
            for sc in BAD_SPECIAL_CHARACTERS] + \
        [''.join('a' for _ in range(65))]

    for bad_model_name in bad_model_names:
        with raises(AssertionError) as aerr:
            chooser = BasicNativeXGBChooser()
            chooser.model_name = bad_model_name
            assert aerr.value
