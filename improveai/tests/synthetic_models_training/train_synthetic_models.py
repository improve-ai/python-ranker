import coremltools as ct
import docker
import json
import numpy as np
import orjson
import os
import xgboost as xgb


from synthetic_models_config import SYNTHETIC_TRACKER_URL, CONFIG_DIR, \
    DATA_DIR, IMAGE_NAME, MODELS_DIR, MODEL_NAME_ENVVAR, INPUT_CHANNEL_NAME,\
    PYTHON_SDK_PACKAGE_NAME, SYNTHETIC_MODELS_TEST_CASES_DIR, \
    SYNTHETIC_MODELS_TARGET_DIR, CLEANUP_CONTAINERS, IMPROVE_ABS_PATH, \
    SYNTHETIC_DATA_DEFINITIONS_DIRECTORY, CONTINUE

from improveai import Scorer
from improveai.reward_tracker import RewardTracker
from improveai.tests.synthetic_models_training.data_generator import \
    BasicSemiRandomDataGenerator


def get_test_data_definitions(test_data_definitions_dir: str):

    test_jsons_paths = \
        [os.sep.join([test_data_definitions_dir, file_name])
         for file_name in os.listdir(test_data_definitions_dir)
         if 'pattern_' not in file_name]

    return test_jsons_paths


def prepare_synthetic_model_dirs(epochs, case_id):
    if os.getcwd() != os.sep.join(str(__file__).split(os.sep)[:-1]):
        raise RuntimeError(
            'Must be ran from local_synthetic_models_training directory '
            'otherwise might perform unexpected deletes')

    print('### PURGING DATA DIRECTORY ###')
    os.system('rm -rf input/data_{}'.format(case_id))
    print('### RECREATING DATA DIRECTORY ###')
    os.system('mkdir -p input/data_{}'.format(case_id))
    print('### PURGING MODEL DIRECTORY ###')
    os.system('rm -rf {}_{}'.format(MODELS_DIR, case_id))
    print('### PREPARING MODEL DIRECTORIES ###')
    for epoch in range(epochs):
        os.system('mkdir -p {}_{}/epoch_{}'.format(MODELS_DIR, case_id, epoch))


def get_scorer(epoch_index: int, case_id: str):
    if epoch_index == 0:
        return

    model_url = \
        os.sep.join([MODELS_DIR + '_{}'.format(case_id),
                     'epoch_{}/model.xgb'.format(epoch_index - 1)])
    return Scorer(model_url=model_url)


def train_model(epoch: int, dataset_name: str, case_id):
    """
    Runs a single training iteration with python docker API

    Parameters
    ----------
    epoch: int
        current epoch index

    Returns
    -------

    """
    dc = docker.from_env()
    volumes = {
        '{}_{}'.format(DATA_DIR, case_id):
            {"bind": f'/opt/ml/input/data/decisions', "mode": 'rw'},
        '{}_{}'.format(MODELS_DIR, case_id) + os.sep + 'epoch_{}'.format(epoch):
            {"bind": '/opt/ml/model', "mode": 'rw'},
        CONFIG_DIR:
            {"bind": '/opt/ml/input/config', "mode": 'rw'}}
    environment = {
        MODEL_NAME_ENVVAR: '{}-epoch-{}'.format(dataset_name.replace('_', '-'), epoch)
    }

    container = \
        dc.containers.run(
            IMAGE_NAME, command='train', detach=True, volumes=volumes, environment=environment)
    print('### Waiting for the model to train ###')
    container.wait()
    print(container.logs().decode("utf-8"))


def run_single_synthetic_training(
        data_definition_json_path: str, target_model_directory: str, case_id):

    synthetic_data_tracker = \
        RewardTracker(track_url=SYNTHETIC_TRACKER_URL, model_name='synthetic-model')

    data_generator = \
        BasicSemiRandomDataGenerator(
            data_definition_json_path=data_definition_json_path,
            track_url=SYNTHETIC_TRACKER_URL)

    data_set_name = data_generator.get_dataset_name()
    prepare_synthetic_model_dirs(epochs=data_generator.epochs, case_id=case_id)

    print('### TRAINING {} MODEL ###'.format(data_generator.dataset_name))
    for epoch_index in range(data_generator.epochs):
        print('\n### PROCESSING EPOCH: {}/{} ###'.format(epoch_index + 1, data_generator.epochs))

        # create scorer
        scorer = get_scorer(epoch_index=epoch_index, case_id=case_id)

        # self, epoch_index: int, reward_tracker: RewardTracker, scorer: Scorer = None
        # generate data
        records = \
            data_generator.make_decisions_for_epoch(
                epoch_index=epoch_index, reward_tracker=synthetic_data_tracker, scorer=scorer)

        # dump currently generated data to decision_{epoch_index}
        # data: list, path: str, mode='w'
        current_epoch_data_path = \
            '{}_{}'.format(DATA_DIR, case_id) + os.sep + '{}_{}.parquet'.format(INPUT_CHANNEL_NAME, epoch_index)
        data_generator.dump_data(data=records, path=current_epoch_data_path)

        # dump all data (along with previous records) to decision_0
        if epoch_index != 0:
            decision_channel_path = \
                '{}_{}'.format(DATA_DIR, case_id) + os.sep + '{}_0.parquet'.format(INPUT_CHANNEL_NAME)
            data_generator.dump_data(data=records, path=decision_channel_path)
        # train model -  run container
        train_model(epoch=epoch_index, dataset_name=data_set_name, case_id=case_id)

    copy_final_model(
        last_epoch_index=epoch_index, data_set_name=data_set_name,
        target_model_directory=target_model_directory, case_id=case_id)

    create_synthetic_model_test_json(
        data_generator=data_generator, model_directory=target_model_directory, )


def convert_xgb_to_mlmodel(source_models_directory):
    # TODO delete once conversion is back online in the trainer
    MODEL_NAME_METADATA_KEY = 'ai.improve.model'
    FEATURE_NAMES_METADATA_KEY = 'ai.improve.features'
    STRING_TABLES_METADATA_KEY = 'ai.improve.string_tables'
    MODEL_SEED_METADATA_KEY = 'ai.improve.seed'
    CREATED_AT_METADATA_KEY = 'ai.improve.created_at'
    VERSION_METADATA_KEY = 'ai.improve.version'

    USER_DEFINED_METADATA_KEY = 'user_defined_metadata'
    MLMODEL_REGRESSOR_MODE = 'regressor'
    VERSION = '8.0.0'

    # load model from FS
    booster = xgb.Booster()
    booster.load_model(source_models_directory + os.sep + 'model.xgb')
    xgb_booster_metadata = orjson.loads(booster.attr('user_defined_metadata'))

    model_name = xgb_booster_metadata[MODEL_NAME_METADATA_KEY]
    model_seed = xgb_booster_metadata[MODEL_SEED_METADATA_KEY]
    created_at = xgb_booster_metadata[CREATED_AT_METADATA_KEY]
    string_tables = xgb_booster_metadata[STRING_TABLES_METADATA_KEY]
    feature_names = xgb_booster_metadata[FEATURE_NAMES_METADATA_KEY]

    # fill metadata values
    mlmodel = ct.converters.xgboost.convert(booster, mode=MLMODEL_REGRESSOR_MODE, feature_names=feature_names, force_32bit_float=True)
    mlmodel.user_defined_metadata[MODEL_NAME_METADATA_KEY] = model_name
    mlmodel.user_defined_metadata[STRING_TABLES_METADATA_KEY] = json.dumps(string_tables)
    mlmodel.user_defined_metadata[MODEL_SEED_METADATA_KEY] = str(model_seed)
    mlmodel.user_defined_metadata[CREATED_AT_METADATA_KEY] = created_at
    mlmodel.user_defined_metadata[VERSION_METADATA_KEY] = VERSION

    return mlmodel


def copy_final_model(
        last_epoch_index: int, data_set_name: str, target_model_directory: str,
        case_id):
    # create dir for data set
    target_model_path = os.sep.join([target_model_directory, data_set_name])

    abs_target_model_path = os.sep.join([IMPROVE_ABS_PATH, target_model_path])

    if not os.path.isdir(abs_target_model_path):
        mkdir_sys_code = os.system('mkdir -p {}'.format(abs_target_model_path))
        assert mkdir_sys_code == 0
    else:
        # purge dir
        rm_sys_code = os.system('rm -rf {}/*'.format(abs_target_model_path))
        assert rm_sys_code == 0

    source_models_directory = \
        os.sep.join(['{}_{}'.format(MODELS_DIR, case_id), 'epoch_{}'.format(last_epoch_index)])

    # copy models from model/epoch_X -> <data set name>/
    # convert model to mlmodel
    mlmodel = convert_xgb_to_mlmodel(source_models_directory)
    mlmodel.save(source_models_directory + os.sep + 'model.mlmodel')

    cp_sys_code = \
        os.system('cp {}/* {}/'.format(source_models_directory, abs_target_model_path))
    assert cp_sys_code == 0
    xgb_gzipped = os.system('gzip {}'.format(os.sep.join([abs_target_model_path, 'model.xgb'])))
    assert xgb_gzipped == 0

    mlmodel_gzipped = os.system('gzip {}'.format(os.sep.join([abs_target_model_path, 'model.mlmodel'])))
    assert mlmodel_gzipped == 0


def create_synthetic_model_test_json(
        data_generator: BasicSemiRandomDataGenerator, model_directory: str):

    # load model to extract model seed and noise (?)
    abs_model_directory = os.sep.join([IMPROVE_ABS_PATH, model_directory])
    model_path = os.sep.join([abs_model_directory, data_generator.dataset_name, 'model.xgb.gz'])
    scorer = Scorer(model_url=model_path)
    # extract all variants, context and variants <-> context mapping from data def
    all_contexts = {} if data_generator.all_contexts is None else data_generator.all_contexts.copy()
    if data_generator.context_fraction < 1 and None not in all_contexts:
        all_contexts.append(None)
    noise_seed = data_generator.data_seed
    # mock up real life scenario(?):
    # -> fix model seed (comes from model) and encoding noise ()
    # -> for each context perform Decision(...)...get()
    # -> if not all decisions in synthetic data had context perform choosing on
    #    plain variants as well
    # -> dump results keeping specified synthetic model json conventino
    # append regret vs epoch to json (?)

    noise = None
    np.random.seed(data_generator.data_seed)
    if hasattr(data_generator, 'noise'):
        noise = data_generator.noise

    if noise is None:
        noise = np.random.rand()

    assert noise is not None

    test_case = {
        "model_url": os.sep.join([PYTHON_SDK_PACKAGE_NAME, model_path.split(PYTHON_SDK_PACKAGE_NAME)[-1][1:]]),
        "metadata": {
            "case_name": data_generator.dataset_name,
            "train_stats": data_generator.reward_cache,

            # "variants_vs_givens_stats": data_generator.variants_vs_givens_stats
        },
        "test_case": {
            "candidates": data_generator.candidates,
            "contexts": data_generator.all_contexts,
            "noise": noise,
            "python_seed": data_generator.data_seed
        },
        "expected_output": []
    }

    prev_encoding_noise = None
    i = 0
    if not all_contexts:
        all_contexts = [None]

    for context in all_contexts:
        scorer.chooser.imposed_noise = noise
        # fix seed so scores are reproducible
        np.random.seed(noise_seed)
        scores = scorer.score(data_generator.candidates, context=context)
        best_item = data_generator.candidates[np.argmax(scores)]
        # get best variant
        # encoding_noise = dm.chooser.current_noise

        decision_output = {
            "item": best_item,
            "scores": list(scores),
        }

        # append new values
        test_case['expected_output'].append(decision_output)

        i += 1

    # save test case
    natural_test_case_path = \
        os.sep.join([
            abs_model_directory, data_generator.dataset_name,
            '{}.json'.format(data_generator.dataset_name)])

    with open(natural_test_case_path, 'w') as ntcf:
        # ntcf.write(simplejson.dumps(test_case, indent=4))
        ntcf.write(json.dumps(test_case))

    test_dir_path = \
        os.sep.join([
            IMPROVE_ABS_PATH, SYNTHETIC_MODELS_TEST_CASES_DIR,
            '{}.json'.format(data_generator.dataset_name)])

    with open(test_dir_path, 'w') as ntcf1:
        # ntcf1.write(simplejson.dumps(test_case, indent=4))
        ntcf1.write(json.dumps(test_case))


if __name__ == '__main__':

    paths = get_test_data_definitions(SYNTHETIC_DATA_DEFINITIONS_DIRECTORY)

    first_batch = [p for p in paths if '/2_' in p or '/0_' in p or '/happy_sunday_' in p]
    second_batch = [p for p in paths if '/1000_' in p]

    paths = first_batch + second_batch

    trained_models_dir = os.sep.join([IMPROVE_ABS_PATH, SYNTHETIC_MODELS_TEST_CASES_DIR])
    already_trained_models = os.listdir(trained_models_dir)

    # paths = [SYNTHETIC_DATA_DEFINITIONS_DIRECTORY + '/happy_sunday.json']
    # paths = [SYNTHETIC_DATA_DEFINITIONS_DIRECTORY + '/a_z.json']
    paths = [
        SYNTHETIC_DATA_DEFINITIONS_DIRECTORY + os.sep + el for el in
        ['0_and_nan.json',
         '2_nested_dict_variants_20_random_nested_dict_givens_large_binary_reward.json',
         '2_numeric_variants_100_random_nested_dict_givens_binary_reward.json',
         '2_numeric_variants_100_random_nested_dict_givens_binary_reward.json',
         '2_numeric_variants_no_givens_large_binary_reward.json',
         '2_variants_20_huge_givens.json',
         '1000_list_of_numeric_variants_20_same_nested_givens_binary_reward.json',
         '1000_numeric_variants_20_random_nested_givens_small_binary_reward.json',
         '1000_numeric_variants_20_same_nested_givens_large_binary_reward.json',
         '1000_numeric_variants_no_givens_small_binary_reward.json',
         'happy_sunday.json',
         'primitive_variants_no_givens_binary_reward.json']]

    # recalc_paths = ['0_and_nan.json']
    # paths = [SYNTHETIC_DATA_DEFINITIONS_DIRECTORY + os.sep + path for path in recalc_paths]
    CONTINUE = False

    for case_id, data_definition_json_path in enumerate(paths):

        if CONTINUE and data_definition_json_path.split(os.sep)[-1] in already_trained_models:
            print('### MODEL ALREADY TRAINED: {} ###'.format(data_definition_json_path.split(os.sep)[-1]))
            continue

        print('### PROCESSING {} OUT OF {} FILES ###'.format(case_id + 1, len(paths)))

        if 'pattern_' in data_definition_json_path:
            print('### DETECTED PATTERN FILE - SKIPPING ###')
            continue

        run_single_synthetic_training(
            data_definition_json_path=data_definition_json_path,
            target_model_directory=SYNTHETIC_MODELS_TARGET_DIR, case_id=0)

        if CLEANUP_CONTAINERS:
            dc = docker.from_env()
            dc.containers.prune()

        os.system('rm -rf input/data_{}'.format(0))
