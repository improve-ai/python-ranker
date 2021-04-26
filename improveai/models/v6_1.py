from copy import deepcopy
import coremltools as ct
import decisions.v6_1 as d
import json

import numpy as np
# TODO change to proper types once known
from multiprocessing import Process, Manager
from typing import Dict, List, Union

from choosers.basic_choosers import BasicChooser
from choosers.mlmodel_chooser import BasicMLModelChooser
from choosers.xgb_chooser import BasicNativeXGBChooser
from utils.general_purpose_utils import constant


class DecisionModel:

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, value):
        self._model_name = value

    # TODO determine desired namespace - chooser or model
    #  left chooser for now
    # @property
    # def model(self):
    #     return self._model
    #
    # @model.setter
    # def model(self, value):
    #     self._model = value

    @property
    def tracker(self):
        return self._tracker

    @tracker.setter
    def tracker(self, value):
        self._tracker = value

    @property
    def chooser(self):
        return self._chooser

    @chooser.setter
    def chooser(self, value):
        self._chooser = value

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tracker = None
        # self.model = None
        self.chooser = None

    def _set_chooser(self, chooser: BasicChooser):
        """
        Sets chooser

        Parameters
        ----------
        chooser: BasicChooser
            chooser object to be used from within DecisionModel

        Returns
        -------

        """
        self.chooser = chooser
        # At this point mode_name is set otherwise error would be thrown
        self.model_name = self.chooser.model_name

    @staticmethod
    def _get_chooser_in_subprocess(
            chooser_constructor: callable, model_src: str or bytes,
            chooser_container: object, chooser_kwargs: dict = None):
        """
        Calls load model on provided chooser. This is designed for subprocess
        model reading and due to xgboost's ability to crash python while reading
        non xgboost model
        
        Parameters
        ----------
        chooser_constructor: callable
            chooser constructor
        model_src: str or bytes
            path or bytes from which model will be loaded
        chooser_container: object
            multiprocessing shared dict
        chooser_kwargs: dict or None
            kwargs for chooser constructor

        Returns
        -------
        None
            None

        """

        used_chooser_kwargs = chooser_kwargs if chooser_kwargs else {}

        chooser = chooser_constructor(**used_chooser_kwargs)
        chooser.load_model(input_model_src=model_src)

        if chooser.__class__.__name__ == 'BasicMLModelChooser':
            # dump mlmodel to string - otherwise impossible to pickle
            spec_string = chooser.model._spec.SerializeToString()
            chooser.model = None
            chooser_container['serialized_model'] = spec_string

        chooser_container['chooser'] = deepcopy(chooser)
        chooser_container['model_kind'] = \
            'native_xgb' if chooser.__class__.__name__ == 'BasicNativeXGBChooser' \
            else 'mlmodel'

    @staticmethod
    def load_model(model_url: str, load_model_timeout: int = 30):
        """
        Synchronously loads XGBoost model from provided path, creates instance
        of DecisionModel and returns it

        Parameters
        ----------
        model_url: str
            path to desired model (FS path or url)
        load_model_timeout: int
            amount of time (seconds) given to chooser loader to finish

        Returns
        -------
        DecisionModel
            new instance of decision model

        """

        # get model source
        try:
            model_src = BasicChooser.get_model_src(model_src=model_url)
        except Exception as exc:
            print('On attempt to read {} model the following error ocurred:')
            print(exc)
            print('Returning empty DecisionModel')
            return DecisionModel(model_name='empty-model')

        man = Manager()
        chooser_container = man.dict()
        chooser_container['chooser'] = None
        chooser_container['model_kind'] = None
        chooser_container['serialized_model'] = None

        # chooser_constructor: callable, model_src: str or bytes,
        # chooser_container: object, chooser_kwargs: dict = None

        chooser_constructors = [BasicMLModelChooser, BasicNativeXGBChooser]
        # left for future use (maybe)
        chooser_constructor_kwargs = [None, None]

        for chooser_constructor, chooser_kwargs in \
                zip(chooser_constructors, chooser_constructor_kwargs):

            chooser_process = \
                Process(
                    target=DecisionModel._get_chooser_in_subprocess,
                    args=(chooser_constructor, model_src, chooser_container,
                          chooser_kwargs))
            chooser_process.start()
            chooser_process.join(load_model_timeout)

            if chooser_container['chooser'] is not None:
                break
        if chooser_container['chooser'] is None:
            print(
                'Failed to load desired model - returning empty DecisionModel')
            return DecisionModel(model_name='empty-model')

        loaded_chooser = chooser_container['chooser']

        if chooser_container['model_kind'] == 'mlmodel':
            # load model from cache
            spec_from_string = ct.proto.Model_pb2.Model()

            serialized_spec = chooser_container.get('serialized_model', None)
            if serialized_spec is None:
                raise ValueError('The serialized mlmodel spec is None')

            spec_from_string.ParseFromString(serialized_spec)
            loaded_chooser.model = ct.models.MLModel(spec_from_string)

        decision_model = DecisionModel(model_name=None)
        decision_model._set_chooser(chooser=loaded_chooser)

        return decision_model

    @staticmethod
    def load_model_async(model_url: str, model_type: str = 'native_xgb'):
        pass

    def score(
            self, variants: list or np.ndarray,
            givens: dict, **kwargs) -> list or np.ndarray:

        # TODO should chooser be settable from the outside ?
        if not self.chooser:  # add async support
            return DecisionModel.generate_descending_gaussians(
                count=len(variants))

        # self, variants: List[Dict[str, object]],
        # context: Dict[str, object],
        # mlmodel_score_res_key: str = 'target',
        # mlmodel_class_proba_key: str = 'classProbability',
        # target_class_label: int = 1, imputer_value: float = np.nan,
        # sigmoid_correction: bool = False, sigmoid_const: float = 0.5,
        # return_plain_results: bool = False,
        # ** kwargs
        # TODO ask about the sigmoid correction
        scores = \
            self.chooser.score_all(
                variants=variants, context=givens, return_plain_results=True)
        return scores

    @staticmethod
    def _validate_variants_and_scores(
            variants: list or np.ndarray, scores: list or np.ndarray,
            **kwargs) -> bool:
        """
        Check if variants and scores are not malformed

        Parameters
        ----------
        variants: np.ndarray
            array of variants
        scores: np.ndarray
            array of scores

        Returns
        -------
        bool
            Flag indicating whether variants and scores are valid

        """

        if variants is None or scores is None:
            raise ValueError('`variants` and `scores` can`t be None')

        if len(variants) != len(scores):
            raise ValueError('Lengths of `variants` and `scores` mismatch!')

        if len(variants) == 0:
            # TODO is this desired ?
            return False

        return True

    @staticmethod
    def top_scoring_variant(
            variants: list or np.ndarray, scores: list or np.ndarray,
            **kwargs) -> dict:
        """
        Gets best variant considering provided scores

        Parameters
        ----------
        variants: np.ndarray
            collection of variants to be ranked
        scores: np.ndarray
            collection of scores used for ranking

        Returns
        -------
        dict
            Returns best variant

        """

        if not DecisionModel._validate_variants_and_scores(
                variants=variants, scores=scores):
            return None

        return variants[np.argmax(scores)]

    @staticmethod
    def rank(
            variants: list or np.ndarray, scores: list or np.ndarray,
            **kwargs) -> list or np.ndarray:
        """
        Return a list of the variants ranked from best to worst

        Parameters
        ----------
        variants: np.ndarray
            collection of variants to be ranked
        scores: np.ndarray
            collection of scores used for ranking

        Returns
        -------
        np.ndarray
            sorted variants

        """

        if not DecisionModel._validate_variants_and_scores(
                variants=variants, scores=scores):
            return None

        variants_w_scores = np.array([variants, scores]).T

        sorted_variants_w_scores = \
            variants_w_scores[(variants_w_scores[:, 1] * -1).argsort()]
        return sorted_variants_w_scores[:, 0]

    @staticmethod
    def generate_descending_gaussians(
            count: int, **kwargs) -> list or np.ndarray:
        """
        Generates random floats and sorts in a descending fashion

        Parameters
        ----------
        count: int
            number of floats to generate

        Returns
        -------
        np.ndarray
            array of sorted floats

        """

        random_scores = np.random.normal(size=count)
        random_scores[::-1].sort()
        return random_scores

    def given(self, givens: dict, **kwargs) -> object:  # returns Decision
        return d.Decision(decision_model=self).given(givens=givens)

    def choose_from(self, variants: list or np.ndarray, **kwargs) -> object:
        return d.Decision(decision_model=self).choose_from(variants=variants)


# if __name__ == '__main__':
#     import json
#     import simplejson
#
#     # url = "https://improve-v5-resources-prod-models-117097735164.s3-us-west-2.amazonaws.com/models/bible/latest/improve-messages-2.0.xgb.gz"
#     # url = '../artifacts/models/dummy_v6_w_name.xgb'
#     url = '../artifacts/models/dummy_v6.xgb'
#
#     with open('../artifacts/data/test/v6_tests/feature_encoder/python_specific/test_batch_encoding_jsonlies.json', 'r') as rj:
#         test_data = json.loads(rj.read())
#
#     records = test_data.get("test_case", None)
#
#     input_variants = [r['variant'] for r in records]
#     context = records[0]['context']
#     print('context')
#     print(context)
#
#     # url = ''
#
#     np.random.seed(1)
#     scores = \
#         DecisionModel.load_model(model_url=url)\
#         .score(variants=input_variants, givens=context)
#
#     ranked = \
#         DecisionModel.rank(variants=np.array(input_variants), scores=scores)
#
#     print(scores)
#     print(ranked)
#
#     # print('[{}]'.format(', '.join([str(el) for el in scores])))
#
#     test_data = {
#         "test_case": {
#             "variants": input_variants,
#             "givens": context
#         },
#         "test_output": ranked.tolist(),  # scores.tolist(),
#         "model_seed": 1,
#         "scores_seed": 1
#
#     }
#     #
#     # # # test_data = {
#     # # #     "test_case": {
#     # # #         "count": len(input_variants),
#     # # #     },
#     # # #     "test_output": [1.74481176421648, 1.6243453636632417, 0.8654076293246785, 0.31903909605709857, -0.2493703754774101, -0.5281717522634557, -0.6117564136500754, -0.7612069008951028, -1.0729686221561705, -2.3015386968802827],
#     # # #     "model_seed": 1,
#     # # #     "scores_seed": 1
#     # # #
#     # # # }
#
#     written_str = \
#         simplejson.dumps(test_data, indent=4)
#
#     test_case_path = \
#         '../artifacts/data/test/v6_tests/decision_model/ranked_native.json'
#
#     with open(test_case_path, 'w') as wj:
#         wj.write(written_str)
