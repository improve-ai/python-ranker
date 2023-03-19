from copy import deepcopy
import numpy as np
import os
from pytest import raises, fixture
import string

from improveai.chooser import XGBChooser, USER_DEFINED_METADATA_KEY

BAD_SPECIAL_CHARACTERS = [el for el in '`~!@#$%^&*()=+[]{};:"<>,/?' + "'"]
ALNUM_CHARS = [el for el in string.digits + string.ascii_letters]

global model_url
model_url = None

global chooser
chooser = None


global model_metadata
model_metadata = None

global valid_model_metadata
valid_model_metadata = \
    {'ai.improve.model': 'appconfig',
     'ai.improve.features': ['context.ga', 'context.gb', 'item'],
     'ai.improve.string_tables': {'item': [42, 34, 10, 7, 63, 39, 6, 19]},
     'ai.improve.seed': 257854654,
     'ai.improve.created_at': '2023-02-14T22:31:12.751902',
     'ai.improve.version': '8.0.0'}


@fixture(autouse=True)
def prep_env():
    global model_url
    model_url = os.getenv('DUMMY_MODEL_PATH', None)
    global chooser
    chooser = XGBChooser()
    chooser.load_model(input_model_src=model_url)


def test_none_model_name():
    global chooser
    with raises(AssertionError) as aerr:
        chooser.model_name = None
        # assert aerr.value


# - test that regexp compliant model name passes regexp
def test_good_model_name():
    good_model_names = \
        ['a', '0', '0-', 'a1-', 'x23yz_', 'a01sd.', 'abc3-xy2z_as4d.'] + \
        [''.join('a' for _ in range(64))]

    global chooser
    for good_model_name in good_model_names:
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
    global chooser
    for bad_model_name in bad_model_names:
        with raises(AssertionError) as aerr:
            chooser.model_name = bad_model_name
            # assert aerr.value

# TODO test IO errors
#         model_metadata = self._get_model_metadata()
#         self.model_seed = self._get_model_seed(model_metadata=model_metadata)
#         self.model_name = self._get_model_name(model_metadata=model_metadata)
#         self.feature_names = self._get_model_feature_names(model_metadata=model_metadata)
#         self.string_tables = self._get_string_tables(model_metadata=model_metadata)
#         self.improveai_major_version_from_metadata = \
#             self._get_improveai_major_version(model_metadata=model_metadata)


def test__get_model_metadata_raises_for_no_user_metadata_attr():
    global chooser
    # this will delete `USER_DEFINED_METADATA_KEY` attribute from model
    chooser.model.set_attr(**{USER_DEFINED_METADATA_KEY: None})

    # attempt to load metadata -> this should raise IOError
    with raises(IOError) as ioe:
        chooser._get_model_metadata()
        # assert ioe.value


def test__get_model_metadata_raises_for_empty_user_metadata_attr():
    global chooser
    chooser.model.set_attr(**{USER_DEFINED_METADATA_KEY: "{}"})

    # attempt to load metadata -> this should raise IOError
    with raises(IOError) as ioe:
        chooser._get_model_metadata()
        # assert ioe.value


def test__get_model_metadata_raises_for_bad_user_metadata_type():

    global chooser
    chooser.model.set_attr(**{USER_DEFINED_METADATA_KEY: "[]"})

    # attempt to load metadata -> this should raise IOError
    with raises(IOError) as ioe:
        chooser._get_model_metadata()
        # assert ioe.value

    chooser.model.set_attr(**{USER_DEFINED_METADATA_KEY: "0"})

    # attempt to load metadata -> this should raise IOError
    with raises(IOError) as ioe:
        chooser._get_model_metadata()
        # assert ioe.value


def test__get_model_metadata_raises_for_malformed_user_metadata_attr():
    global chooser
    chooser.model.set_attr(**{USER_DEFINED_METADATA_KEY: "{abc, 13]"})

    # attempt to load metadata -> this should raise IOError
    with raises(IOError) as ioe:
        chooser._get_model_metadata()
        # assert ioe.value


def test__get_model_metadata_raises_for_user_metadata_attr_with_missing_mandatory_keys():
    global chooser
    valid_model_metadata = {
        'ai.improve.model': 'appconfig',
        'ai.improve.features': ['context.ga', 'context.gb', 'item'],
        'ai.improve.string_tables':
            {'item': [42, 34, 10, 7, 63, 39, 6, 19]},
        'ai.improve.seed': 257854654,
        'ai.improve.created_at': '2023-02-14T22:31:12.751902',
        'ai.improve.version': '8.0.0'}

    model_metadata_with_missing_keys = deepcopy(valid_model_metadata)
    missing_key = 'ai.improve.model'
    del model_metadata_with_missing_keys[missing_key]

    chooser.model.set_attr(**{USER_DEFINED_METADATA_KEY: "{abc, 13]"})

    # attempt to load metadata -> this should raise IOError
    with raises(IOError) as ioe:
        chooser._get_model_metadata()
        assert missing_key in ioe.value


def test__get_model_feature_names_raises_for_none_metadata():
    with raises(IOError) as ioe:
        chooser._get_model_feature_names(model_metadata=None)


def test__get_model_feature_names_raises_for_empty_metadata():
    with raises(IOError) as ioe:
        chooser._get_model_feature_names(model_metadata={})


def test__get_model_feature_names_raises_for_no_feature_names():
    global chooser
    global valid_model_metadata
    model_metadata_with_missing_key = deepcopy(valid_model_metadata)
    del model_metadata_with_missing_key[chooser.FEATURE_NAMES_METADATA_KEY]

    with raises(IOError) as ioe:
        chooser._get_model_feature_names(
            model_metadata=model_metadata_with_missing_key)


def test__get_model_feature_names_raises_for_none_feature_names():
    with raises(IOError) as ioe:
        chooser._get_model_feature_names(model_metadata={chooser.FEATURE_NAMES_METADATA_KEY: None})


def test__get_model_feature_names_raises_for_empty_feature_names():
    with raises(IOError) as ioe:
        chooser._get_model_feature_names(model_metadata={chooser.FEATURE_NAMES_METADATA_KEY: []})


def test__get_model_feature_names_raises_for_one_feature_name_not_string():
    with raises(IOError) as ioe:
        chooser._get_model_feature_names(model_metadata={chooser.FEATURE_NAMES_METADATA_KEY: ['a', 1, '2']})

