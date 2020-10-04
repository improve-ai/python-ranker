from argparse import ArgumentParser
import json
import numpy as np

from choosers.basic_choosers import BasicChooser
from choosers.mlmodel_chooser import BasicMLModelChooser
from choosers.xgb_chooser import BasicNativeXGBChooser
from utils.gen_purp_utils import constant, read_jsonstring_frm_file


class ImproveModel:

    @property
    def chooser(self) -> BasicChooser:
        return self._chooser

    @chooser.setter
    def chooser(self, new_val: BasicChooser):
        self._chooser = new_val

    @property
    def model_kind(self) -> str:
        return self._model_kind

    @model_kind.setter
    def model_kind(self, new_val: str):
        self._model_kind = new_val

    @property
    def pth_to_model(self) -> str:
        return self._pth_to_model

    @pth_to_model.setter
    def pth_to_model(self, new_val: str):
        self._pth_to_model = new_val

    @constant
    def SUPPORTED_MODEL_KINDS():
        return ['mlmodel', 'xgb_native']

    @constant
    def SUPPORTED_CALLS():
        return ['score', 'sort', 'choose']

    def __init__(self, model_kind: str, model_pth: str):
        self.chooser = None
        self.model_kind = model_kind
        self.pth_to_model = model_pth
        self._set_chooser()
        self._load_choosers_model()

    def _set_chooser(self):
        """
        Sets desired chooser

        Returns
        -------
        None

        """

        if self.model_kind == 'mlmodel':
            self.chooser = BasicMLModelChooser()
        elif self.model_kind == 'xgb_native':
            self.chooser = BasicNativeXGBChooser()

    def _load_choosers_model(self):
        """
        Loads desired model using chooser API call (load_model method)

        Returns
        -------
        None

        """
        self.chooser.load_model(pth_to_model=self.pth_to_model)

    def _get_json_frm_str(self, json_str) -> list or dict or None:
        """
        Attempts to convert input json string into json

        Parameters
        ----------
        json_str: str
            JSON string to be converted into list or dict

        Returns
        -------
        list or dict or None
            JSON laoded from string or None if loading fails

        """
        try:
            loaded_json = json.loads(json_str)
        except Exception as exc:
            print(
                'Failed to load json from provided JSON string: \n {}'
                .format(json_str))
            loaded_json = None
        return loaded_json

    def _check_if_single_variant(self, variants_json) -> bool:
        """
        Determines if provided variant(s) JSON is a single variant or list of
        variants

        Parameters
        ----------
        variants_json: list or dict
            either single variant as dict or multiple variants as list of dicts

        Returns
        -------
        bool
            True if input json is a single variant False if multiple variants
            were provided

        """
        if isinstance(variants_json, list):
            return False
        elif isinstance(variants_json, dict):
            return True
        else:
            raise TypeError('Unsupported type of provided variants_json!')

    def _get_as_is_or_json_str(
            self, input_val: object, cli_call: bool) -> object:
        """
        If cli_call is True atempts to return json otherwise returns original
        input

        Parameters
        ----------
        input_val: object
            value to be returned or serialized to JSON
        cli_call: bool
            if True attempts JSON serialization of input

        Returns
        -------
        object
            plain input or JSON string

        """
        if not cli_call:
            return input_val
        else:
            dumped_val = input_val
            if isinstance(dumped_val, np.ndarray):
                dumped_val = input_val.tolist()

            return json.dumps(dumped_val)

    def score(
            self, variants: str, context: str, model_metadata: str,
            cli_call: bool = False) -> np.ndarray or str:
        """
        Scores provided variants with provided context

        Parameters
        ----------
        variants: list or dict
            variant(s) to score
        context: dict
            dict with lookup table

        Returns
        -------
        np.ndarray or str
            np.ndarray if this is not cli call else results as json string

        """
        variants_json = self._get_json_frm_str(json_str=variants)
        context_json = self._get_json_frm_str(json_str=context)
        model_metadata_json = self._get_json_frm_str(json_str=model_metadata)

        is_single_variant = \
            self._check_if_single_variant(variants_json=variants_json)
        if is_single_variant:
            variants_w_scores = \
                self.chooser.score(
                    variant=variants_json, context=context_json,
                    model_metadata=model_metadata_json)
        else:
            variants_w_scores = \
                self.chooser.score_all(
                    variants=variants_json, context=context_json,
                    model_metadata=model_metadata_json)

        ret_variants_w_scores = variants_w_scores
        if isinstance(variants_w_scores, float):
            ret_variants_w_scores = \
                np.array([[variants_json, variants_w_scores]]).reshape((1, 2))

        return self._get_as_is_or_json_str(
            input_val=ret_variants_w_scores, cli_call=cli_call)

    def sort(
            self, variants: str, context: str, model_metadata: str,
            cli_call: bool = False) -> np.ndarray or str:
        """
        Scores and sorts provided variants and context

        Parameters
        ----------
        variants: str
            json string with single variant or list of variants
        context: str
            json string with lookup table
        cli_call: bool
            is this exact cli_call or from inside code call

        Returns
        -------
        np.ndarray or str
            array with results or json string

        """

        variants_w_scores = \
            self.score(
                variants=variants, context=context,
                model_metadata=model_metadata, cli_call=False)
        srtd_variants_w_scores = \
            self.chooser.sort(variants_w_scores=variants_w_scores)

        return self._get_as_is_or_json_str(
            input_val=srtd_variants_w_scores, cli_call=cli_call)

    def choose(
            self, variants: str, context: str, model_metadata: str,
            cli_call: bool = False):

        variants_w_scores = \
            self.score(
                variants=variants, context=context,
                model_metadata=model_metadata, cli_call=False)
        chosen_variant = \
            self.chooser.choose(variants_w_scores=variants_w_scores)

        return self._get_as_is_or_json_str(
            input_val=chosen_variant, cli_call=cli_call)


if __name__ == '__main__':

    DEFAULT_VARIANT = \
        json.dumps(
            [{"text": "Sanity check test 1"},
             {"text": "Sanity check test 2"},
             {"text": "Completely different sanity check"}])

    with open('test_artifacts/context.json', 'r') as mj:
        context_str = mj.read()

    DEFAULT_CONTEXT = context_str

    with open('test_artifacts/model.json', 'r') as mmd:
        metadata_str = mmd.read()
        # model_metadata = json.loads(metadata_str)

    DEFAULT_MODEL_METADATA = metadata_str

    ap = ArgumentParser()
    ap.add_argument(
        'method_call', default='score',
        help='Which action of 4 available should be performed '
             '(score, sort, choose)')
    ap.add_argument(
        'model_kind', default='mlmodel',
        help='String indicating which model type would be used: mlmodel, '
             'xgb_native or tf_lite model')
    ap.add_argument('model_pth', help='Path to model file')
    ap.add_argument('--variants', help='JSON with variants to be scored',
                    default=DEFAULT_VARIANT)
    ap.add_argument(
        '--variants_pth',
        help='Path to file with JSON with variants to be scored',
        default='')
    ap.add_argument(
        '--context', help='JSON with context', default=DEFAULT_CONTEXT)
    ap.add_argument(
        '--context_pth', help='Path to file with JSON with context',
        default='')
    ap.add_argument(
        '--model_metadata', help='JSON with lookup table and seed',
        default=DEFAULT_MODEL_METADATA)
    ap.add_argument(
        '--model_metadata_pth',
        help='Path to file with JSON with lookup table and seed',
        default='')
    ap.add_argument(
        '--results_pth',
        help='Path to file where JSON with prediction results will be dumped',
        default='')

    pa = ap.parse_args()

    im = ImproveModel(model_kind=pa.model_kind, model_pth=pa.model_pth)

    if pa.method_call not in im.SUPPORTED_CALLS:
        raise ValueError(
            'CLI supported method_calls are {}. \n Provided method_calls: {}'
            .format(','.join(im.SUPPORTED_CALLS), pa.method_call))

    if pa.model_kind not in im.SUPPORTED_MODEL_KINDS:
        raise ValueError(
            'Currently supported model kinds are: {} \n. '
            'You have provided following model kind: {}'
            .format(','.join(im.SUPPORTED_MODEL_KINDS), pa.model_kind))

    assert hasattr(im, pa.method_call)

    # loading files
    input_objects = {
        'variants': pa.variants,
        'context': pa.context,
        'model_metadata': pa.model_metadata}

    input_pths = [pa.variants_pth, pa.context_pth, pa.model_metadata_pth]

    for input_obj_key, input_pth in zip(input_objects.keys(), input_pths):
        if input_pth:
            input_objects[input_obj_key] = read_jsonstring_frm_file(input_pth)

    input_objects['cli_call'] = True

    des_method = getattr(im, pa.method_call)

    res = \
        des_method(**input_objects)

    print('\n##################################################################\n\n'
          'Result of method call: {} \non variants: {}\nwith model: {} \n\n'
          '##################################################################\n\n'
          'is: \n'
          '{}'
          .format(pa.method_call, pa.variants, pa.model_pth, res))

    if pa.results_pth:
        with open(pa.results_pth, 'w') as resf:
            resf.writelines(res)
