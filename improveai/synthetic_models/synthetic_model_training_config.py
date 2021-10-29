import os

SYNTHETIC_MODELS_TRAINING_DIR = os.sep.join(str(__file__).split(os.sep)[:-1])
PYTHON_SDK_PACKAGE_NAME = 'improveai'


IMPROVE_ABS_PATH = os.sep.join(str(__file__).split(os.sep)[:-3])
print(IMPROVE_ABS_PATH)

SYNTHETIC_DATA_DEFINITIONS_DIRECTORY = os.sep.join([
    IMPROVE_ABS_PATH, PYTHON_SDK_PACKAGE_NAME,
    'artifacts/data/synthetic_models/datasets_definitions'])


SYNTHETIC_TRACKER_URL = 'http://dummy.track.url'
MODELS_DIR = SYNTHETIC_MODELS_TRAINING_DIR + os.sep + 'models/model'


INPUT_DIR = SYNTHETIC_MODELS_TRAINING_DIR + os.sep + 'input'
CONFIG_DIR = INPUT_DIR + os.sep + 'config'
DATA_DIR = INPUT_DIR + os.sep + 'data'


MODEL_NAME_ENVVAR = 'MODEL_NAME'
IMAGE_NAME = 'v6_trainer'
INPUT_CHANNEL_NAME = 'decisions'

SYNTHETIC_MODELS_TEST_CASES_DIR = \
    os.sep.join([PYTHON_SDK_PACKAGE_NAME, 'artifacts', 'data', 'test', 'v6_tests',
                 'synthetic_models'])

SYNTHETIC_MODELS_TARGET_DIR = \
    os.sep.join([PYTHON_SDK_PACKAGE_NAME, 'artifacts', 'models', 'synthetic_models'])


REINSTALL_SDK = True
CLEANUP_AFTER_TEST = True
CLEANUP_CONTAINERS = True

CONTINUE = True
