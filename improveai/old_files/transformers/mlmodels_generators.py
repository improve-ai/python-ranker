from argparse import ArgumentParser
from collections.abc import Iterable
import coremltools as ct
import json
from xgboost import Booster


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

    def get_metadata_from_source_model(
            self, metadata_key: str = 'user_defined_metadata') -> str:
        metadata = json.loads(self.src_model.attr(metadata_key)).get('json', None)
        if not metadata:
            return {}
        return metadata

    def get_feature_names_from_model_metadata(
            self, model_metadata_string: str,
            feature_names_key: str = 'feature_names') -> Iterable:

        loaded_metadata = json.loads(model_metadata_string).get('json', None)
        if not loaded_metadata:
            raise ValueError('No metadata stored under `json` key!')

        feature_names = loaded_metadata.get(feature_names_key, None)
        if not feature_names:
            raise ValueError('Provided model has no feature names info!')
        return feature_names

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
        '--src_model_pth', default='../artifacts/models/dummy_v6.xgb',
        help='Path to model to be converted into mlmodel')
    ap.add_argument(
        '--model_metadata_pth', default='../artifacts/test_artifacts/model_metadata.json',
        help='Path to file with appended metadata')
    ap.add_argument(
        '--trgt_model_pth',
        default='../artifacts/models/v6_conv_model.mlmodel',
        help='Path to saved model')

    ap.add_argument(
        '--api_version', default='v6', help='Which API version should be used')

    ap.add_argument(
        '--feature_names_key', default='feature_names',
        help='Metadata`s key which stores feature names')

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

    if pa.api_version == 'v5':
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
    elif pa.api_version == 'v6':

        # extract JSON metadata from xgb model
        model_metadata = \
            bmlg.get_metadata_from_source_model(
                metadata_key='user_defined_metadata')
        # get feature names
        feature_names = model_metadata.get(pa.feature_names_key, None)

        if not feature_names:
            raise ValueError('No feature names in provided xgb model.')

        # convert xgb -> mlmodel
        converter = ct.converters.xgboost
        bmlg.convert_src_model_to_mlmodel(
            converter=converter,
            converter_kwargs={
                'mode': 'classifier', 'feature_names': feature_names,
                'force_32bit_float': False, 'class_labels': [0, 1]})
        bmlg.append_json_to_mlmodel(
            payload=model_metadata, payload_key='json')
        # append payload to mlmodel instance

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
               cached_model.user_defined_metadata['json']
        # assert int(
        #     json.loads(cached_model.user_defined_metadata['json'])[1:-1]
        #     .split(',')[-1]) == 15000000 - 1
        print('Assertion OK')
