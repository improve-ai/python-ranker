import docker
import json
import numpy as np
import os
import requests_mock as rqm
import simplejson


from synthetic_model_training_config import SYNTHETIC_TRACKER_URL, CONFIG_DIR, \
    DATA_DIR, IMAGE_NAME, MODELS_DIR, MODEL_NAME_ENVVAR, INPUT_CHANNEL_NAME,\
    PYTHON_SDK_PACKAGE_NAME, SYNTHETIC_MODELS_TEST_CASES_DIR, \
    SYNTHETIC_MODELS_TARGET_DIR, CLEANUP_CONTAINERS, IMPROVE_ABS_PATH, \
    SYNTHETIC_DATA_DEFINITIONS_DIRECTORY, CONTINUE

from improveai import Decision, DecisionModel, DecisionTracker
from improveai.synthetic_models.data_generator import \
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


def get_decision_model(
        epoch_index: int, tracker: DecisionTracker, dataset_name: str, case_id):
    dm = DecisionModel(model_name='{}-{}'.format(dataset_name.replace('_', '-'), epoch_index - 1)[:64])

    if epoch_index != 0:
        model_path = \
            os.sep.join([MODELS_DIR + '_{}'.format(case_id), 'epoch_{}/model.xgb'.format(epoch_index - 1)])
        dm = DecisionModel(model_name=None).load(model_path)
        print(model_path)

    dm.track_with(tracker=tracker)

    return dm


def train_model(epoch: int, dataset_name: str, case_id):
    """
    Runs a single training iteration with python docker API

    Parameters
    ----------
    epoch: int
        current epoch inex

    Returns
    -------

    """
    # TODO add hyperparams
    dc = docker.from_env()
    volumes = {
        '{}_{}'.format(DATA_DIR, case_id): {
            "bind": '/opt/ml/input/data', "mode": 'rw'},
        '{}_{}'.format(MODELS_DIR, case_id) + os.sep + 'epoch_{}'.format(epoch): {
            "bind": '/opt/ml/model', "mode": 'rw'},
        CONFIG_DIR: {
            "bind": '/opt/ml/input/config', "mode": 'rw'}}
    environment = {
        MODEL_NAME_ENVVAR: '{}-epoch-{}'.format(dataset_name.replace('_', '-'), epoch)
    }

    container = \
        dc.containers.run(
            IMAGE_NAME, detach=True, volumes=volumes, environment=environment)
    print('### Waiting for the model to train ###')
    container.wait()


def run_single_synthetic_training(
        data_definition_json_path: str, target_model_directory: str, case_id):

    synthetic_data_tracker = \
        DecisionTracker(
            track_url=SYNTHETIC_TRACKER_URL)

    data_generator = \
        BasicSemiRandomDataGenerator(
            data_definition_json_path=data_definition_json_path,
            track_url=SYNTHETIC_TRACKER_URL)

    data_set_name = data_generator.get_dataset_name()
    prepare_synthetic_model_dirs(epochs=data_generator.epochs, case_id=case_id)

    print('### TRAINING {} MODEL ###'.format(data_generator.dataset_name))
    for epoch_index in range(data_generator.epochs):
        print('\n### PROCESSING EPOCH: {}/{} ###'.format(epoch_index + 1, data_generator.epochs))
        # create decision model
        dm = None
        dm = \
            get_decision_model(
                epoch_index=epoch_index, tracker=synthetic_data_tracker,
                dataset_name=data_set_name, case_id=case_id)

        # generate data
        records = \
            data_generator.make_decision_for_epoch(
                epoch_index=epoch_index, decision_model=dm)

        # TODO uncomment after validation
        # dump currently generated data to decision_{epoch_index}
        # data: list, path: str, mode='w'
        current_epoch_data_path = \
            '{}_{}'.format(DATA_DIR, case_id) + os.sep + '{}_{}'.format(INPUT_CHANNEL_NAME, epoch_index)
        data_generator.dump_data(
            data=records, path=current_epoch_data_path, mode='w')

        # dump all data (along with previous records) to decision_0
        if epoch_index != 0:
            decision_channel_path = \
                '{}_{}'.format(DATA_DIR, case_id) + os.sep + '{}_0'.format(INPUT_CHANNEL_NAME)
            data_generator.dump_data(
                data=records, path=decision_channel_path, mode='a')
        # train model -  run container
        train_model(epoch=epoch_index, dataset_name=data_set_name, case_id=case_id)

    copy_final_model(
        last_epoch_index=epoch_index, data_set_name=data_set_name,
        target_model_directory=target_model_directory, case_id=case_id)

    create_synthetic_model_test_json(
        data_generator=data_generator, model_directory=target_model_directory)


def copy_final_model(
        last_epoch_index: int, data_set_name: str, target_model_directory: str,
        case_id):
    # create dir for data set
    target_model_path = os.sep.join([target_model_directory, data_set_name])

    abs_target_model_path = os.sep.join([IMPROVE_ABS_PATH, target_model_path])
    if not os.path.isdir(abs_target_model_path):
        mkdir_sys_code = os.system('mkdir -p {}'.format(abs_target_model_path))
        assert mkdir_sys_code == 0

    source_models_directory = \
        os.sep.join(['{}_{}'.format(MODELS_DIR, case_id), 'epoch_{}'.format(last_epoch_index)])
    # copy models from model/epoch_X -> <data set name>/
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
    dm = DecisionModel(model_name=None).load(model_path)
    dm.track_with(DecisionTracker(data_generator.track_url))
    # extract all variants, givens and variants <-> givens mapping from data def
    all_givens = {} if data_generator.all_givens is None else data_generator.all_givens.copy()
    if data_generator.givens_fraction < 1 and None not in all_givens:
        all_givens.append(None)
    noise_seed = data_generator.data_seed
    # mock up real life scenario(?):
    # -> fix model seed (comes from model) and encoding noise ()
    # -> for each givens perform Decision(...)...get()
    # -> if not all decisions in synthetic data had givens perform choosing on
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
            "variants": data_generator.variants,
            "givens": data_generator.all_givens,
            "noise": noise,
            "python_seed": data_generator.data_seed
        },
        "expected_output": []
    }

    prev_encoding_noise = None
    i = 0
    if not all_givens:
        all_givens = [None]

    for givens in all_givens:
        with rqm.Mocker() as m:
            m.post(data_generator.track_url, text='success')
            # create a decision object
            decision = \
                Decision(decision_model=dm).choose_from(data_generator.variants)\
                .given(givens)

            # fix seed to get reproducible results
            np.random.seed(noise_seed)
            dm.chooser.imposed_noise = noise
            # get best variant
            best_variant = decision.get()
            # get encoding noise
            # encoding_noise = dm.chooser.current_noise

            decision_output = {
                "variant": best_variant,
                "scores": list(decision.scores)
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
