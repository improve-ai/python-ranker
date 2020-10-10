from argparse import ArgumentParser
import coremltools as ct
import json
import numpy as np
import os
import pickle as pkl
import tarfile
from xgboost import Booster, XGBClassifier, XGBRegressor


class BasicMLModelGenerator:

    @property
    def src_model(self) -> object:
        return self._src_model

    @src_model.setter
    def src_model(self, new_val: object):
        self._src_model = new_val

    @property
    def conv_model(self) -> ct.models.MLModel:
        return self._conv_model

    @conv_model.setter
    def conv_model(self, new_val: ct.models.MLModel):
        self._conv_model = new_val

    @property
    def metadata_json(self):
        return self._metadata_json

    @metadata_json.setter
    def metadata_json(self, new_val):
        self._metadata_json = new_val

    def __init__(self):
        # initialize
        self.src_model = None
        self.conv_model = None
        self.metadata_json = None

    def load_model(self, loader: callable, pth_to_model: str):
        self.src_model = loader(pth_to_model)

    def load_metadata_json(self, metadata_pth: str):
        with open(metadata_pth, 'r') as mj:
            json_str = mj.readline()
            read_json = json.loads(json_str)

        self.metadata_json = read_json

    def convert_src_model_to_mlmodel(
            self, converter: callable, converter_kwargs: dict = {}):
        self.conv_model = converter.convert(self.src_model, **converter_kwargs)

    def append_json_to_mlmodel(
            self, payload: str = None, payload_key: str = None):

        if not payload:
            payload = self.metadata_json
        if not payload_key or payload_key == 'coremltoolsVersion':
            payload_key = 'appended_payload'

        try:
            self.conv_model.user_defined_metadata[payload_key] = \
                json.dumps(payload)
        except Exception as exc:
            print('Appending mlmodel with desired JSON failed')
            print(exc)

    def save_mlmodel(
            self, conv_model_pth: str,
            save_callable_attr_n: str = 'save_model'):

        if not hasattr(self.conv_model, save_callable_attr_n):
            raise ValueError(
                'There was no attribute of name {} in converted model'
                .format(save_callable_attr_n))

        saver = getattr(self.conv_model, save_callable_attr_n)
        saver(conv_model_pth)


def load_xgb_model(model_pth: str):
    xgb_model = Booster()
    xgb_model.load_model(model_pth)
    return xgb_model


if __name__ == '__main__':

    ap = ArgumentParser()
    ap.add_argument(
        '--src_model_pth', default='../test_artifacts/model.xgb',
        help='Path to model to be converted into mlmodel')
    ap.add_argument(
        '--model_metadata_pth', default='../test_artifacts/model.json',
        help='Path to file with appended metadata')
    ap.add_argument(
        '--trgt_model_pth',
        default='../test_artifacts/conv_model.mlmodel',
        help='Path to saved model')

    ap.add_argument(
        '--check_conv_model',
        action='store_true',
        help='Predicts on converted model with 0s. Works only on macOS or higher')

    pa = ap.parse_args()

    bmlg = BasicMLModelGenerator()

    # src_model_fname = '../test_artifacts/model.xgb'
    # metadata_fname = '../test_artifacts/model.json'
    # conv_model_fname = '../test_artifacts/conv_model.mlmodel'

    run_params_vals = \
        [pa.src_model_pth, pa.model_metadata_pth, pa.trgt_model_pth]
    run_params_keys = ['src_model_fname', 'metadata_fname', 'conv_model_fname']

    run_params = {}

    for fname, fname_key in zip(run_params_vals, run_params_keys):
        run_params[fname_key] = fname
            # os.sep.join(str(__file__).split(os.sep)[:-1] + [fname])

    print(run_params)

    bmlg.load_model(
        loader=load_xgb_model, pth_to_model=run_params[run_params_keys[0]])

    bmlg.load_metadata_json(run_params[run_params_keys[1]])

    columns_count = len(bmlg.metadata_json['table'][1])
    feature_names = list(
        map(lambda i: 'f{}'.format(i), range(0, columns_count)))

#    for i in range(columns_count):
#        # print(bmlg.metadata_json['table'][1][i])
#        print('#####################################')
#        print('Printing info about the {}th feature'.format(i))
#        print(len(bmlg.metadata_json['table'][1][i][0]))
#        print(len(bmlg.metadata_json['table'][1][i][1]))

    converter = ct.converters.xgboost
    bmlg.convert_src_model_to_mlmodel(
        converter=converter,
        converter_kwargs={
            'mode': 'classifier', 'feature_names': feature_names,
            'force_32bit_float': False, 'class_labels': [0, 1]})

    bmlg.append_json_to_mlmodel(payload_key='json')

    # this is section where appending large JSON for tests takes place
    # print('\n\nGenerating large json list 0 - 14999999 -> about 140 MB')
    # with open('../test_artifacts/model_large.json', 'w') as mockup_big_json:
    #     mockup_big_json.writelines(json.dumps([el for el in range(15000000)]))
    #
    # print('Done!')
    #
    # with open('../test_artifacts/model_large.json', 'r') as jsonf:
    #     large_payload = jsonf.read()
    #
    # bmlg.append_json_to_mlmodel(payload=large_payload, payload_key='json')

    bmlg.save_mlmodel(
        run_params[run_params_keys[2]], save_callable_attr_n='save')

    if pa.check_conv_model:
        #### Loading saved model ####
        cached_model = ct.models.MLModel(run_params[run_params_keys[2]])
        preds = \
            cached_model.predict(
                dict(zip(feature_names, [0 for el in range(len(feature_names))])))
        print('Predictions print')
        print(preds)
        print('Asserting if original json is eequal to imputed one')
        assert json.dumps(bmlg.metadata_json) == \
               cached_model.user_defined_metadata['appended_payload']
        # assert int(
        #     json.loads(cached_model.user_defined_metadata['json'])[1:-1]
        #     .split(',')[-1]) == 15000000 - 1
        print('Assertion OK')
