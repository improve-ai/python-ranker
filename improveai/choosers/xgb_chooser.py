from collections.abc import Iterable
from copy import deepcopy
import json
from numbers import Number
import numpy as np
import pickle
from time import time
from typing import Dict, List
from traceback import print_exc
from xgboost import Booster, DMatrix
from xgboost.core import XGBoostError

from improveai.choosers.basic_choosers import BasicChooser
from improveai.encoder_cython_utils import cfe
from improveai.feature_encoder import FeatureEncoder
import improveai.settings as improve_settings
from improveai.utils.general_purpose_tools import constant, sigmoid
from improveai.utils.choosers_feature_encoding_tools import encoded_variants_to_np


class BasicNativeXGBChooser(BasicChooser):

    @property
    def model(self) -> Booster:
        return self._model

    @model.setter
    def model(self, new_val: Booster):
        self._model = new_val

    @property
    def model_metadata(self) -> Dict[str, object]:
        return self._model_metadata

    @model_metadata.setter
    def model_metadata(self, new_val: Dict[str, object]):
        self._model_metadata = new_val

    @property
    def feature_encoder(self) -> FeatureEncoder:
        return self._feature_encoder

    @feature_encoder.setter
    def feature_encoder(self, new_val: FeatureEncoder):
        self._feature_encoder = new_val

    @property
    def model_metadata_key(self):
        return self._mlmodel_metadata_key

    @model_metadata_key.setter
    def model_metadata_key(self, new_val: str):
        self._mlmodel_metadata_key = new_val

    # @property
    # def lookup_table_key(self) -> str:
    #     return self._lookup_table_key
    #
    # @lookup_table_key.setter
    # def lookup_table_key(self, new_val: str):
    #     self._lookup_table_key = new_val

    @property
    def model_seed_key(self) -> str:
        return self._model_seed_key

    @model_seed_key.setter
    def model_seed_key(self, new_val: str):
        self._model_seed_key = new_val

    @property
    def model_seed(self):
        return self._model_seed

    @model_seed.setter
    def model_seed(self, value):
        self._model_seed = value

    @property
    def model_name_key(self):
        return self._model_name_key

    @model_name_key.setter
    def model_name_key(self, value):
        self._model_name_key = value

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, value):
        self._model_name = value

    @property
    def model_feature_names_key(self) -> str:
        return self._model_feature_names_key

    @model_feature_names_key.setter
    def model_feature_names_key(self, new_val: str):
        self._model_feature_names_key = new_val

    @property
    def model_feature_names(self) -> np.ndarray:
        return self._model_feature_names

    @model_feature_names.setter
    def model_feature_names(self, new_val: np.ndarray):
        self._model_feature_names = new_val

    @property
    def model_objective(self) -> str:
        return self._model_objective

    @model_objective.setter
    def model_objective(self, new_val: str):
        self._model_objective = new_val

    @property
    def feature_encoder_extras(self):
        return self._feature_encoder_extras

    @feature_encoder_extras.setter
    def feature_encoder_extras(self, value):
        self._feature_encoder_extras = value

    @constant
    def SUPPORTED_OBJECTIVES() -> list:
        return ['reg', 'binary', 'multi']

    @constant
    def TIEBREAKER_MULTIPLIER() -> float:
        return 1e-6

    def __init__(
            self, mlmodel_metadata_key: str = 'json',
            model_feature_names_key: str = 'feature_names',
            model_seed_key: str = 'model_seed',
            model_name_key: str = 'model_name'):

        self.model = None
        self.model_metadata_key = mlmodel_metadata_key
        self.model_metadata = None

        self.feature_encoder = None
        # self.lookup_table_key = lookup_table_key
        self.model_feature_names_key = model_feature_names_key
        self.model_feature_names = np.empty(shape=(1,))

        self.model_seed_key = model_seed_key
        self.model_seed = None

        self.model_name_key = model_name_key
        self.model_name = None

        self.model_objective = None

    def load_model(self, input_model_src: str, verbose: bool = False):
        """
        Loads desired model from input path.

        Parameters
        ----------
        input_model_src: str
            path to desired model
        verbose: bool
            should I print msgs
        Returns
        -------
        None
            None
        """

        try:
            if verbose:
                print('Attempting to load: {} model'.format(
                    input_model_src if len(input_model_src) < 100 else
                    str(input_model_src[:10]) + ' ... ' + str(
                        input_model_src[-10:])))

            raw_model_src = self.get_model_src(model_src=input_model_src)

            model_src = \
                raw_model_src if isinstance(raw_model_src, str) \
                else bytearray(raw_model_src)

            self.model = Booster()
            self.model.load_model(model_src)
            if verbose:
                print('Model: {} successfully loaded'.format(
                    input_model_src if len(input_model_src) < 100 else
                    str(input_model_src[:10]) + ' ... ' + str(
                        input_model_src[-10:])))
        except XGBoostError as xgbe:
            if verbose:
                print('Attempting to read via pickle interface')
            with open(input_model_src, 'rb') as xgbl:
                self.model = pickle.load(xgbl)
            print_exc()
        except Exception as exc:
            if verbose:
                print(
                    'When attempting to load the model: {} the following error '
                    'occured: {}'.format(input_model_src, exc))
            print_exc()

        self.model_metadata = self._get_model_metadata()
        self.model_seed = self._get_model_seed()
        self.model_name = self._get_model_name()

        self.model_feature_names = self._get_model_feature_names()
        self.feature_encoder = self._get_feature_encoder()
        self.model_objective = self._get_model_objective()

    def _get_model_metadata(self) -> dict:
        """
        Model metadata (hash table, etc. getter)

        Returns
        -------
        Dict[str, object]
            dict with model metadata
        """

        assert 'user_defined_metadata' in self.model.attributes().keys()
        user_defined_metadata_str = self.model.attr('user_defined_metadata')
        user_defined_metadata = json.loads(user_defined_metadata_str)
        assert self.model_metadata_key in user_defined_metadata.keys()

        return user_defined_metadata[self.model_metadata_key]

    def score(
            self, variants: List[Dict[str, object]],
            givens: Dict[str, object],
            mlmodel_score_res_key: str = 'target',
            mlmodel_class_proba_key: str = 'classProbability',
            target_class_label: int = 1, imputer_value: float = np.nan,
            sigmoid_correction: bool = False, sigmoid_const: float = 0.5,
            return_plain_results: bool = False,
            **kwargs) -> np.ndarray:

        """
        Scores all provided variants

        Parameters
        ----------
        variants: list
            list of variants to scores
        givens: dict
            context dict needed for encoding
                mlmodel_score_res_key
        mlmodel_class_proba_key: str
            param added for mlmodel api consistency
        target_class_label: int
            label of the target class
        imputer_value: float
            value with which missing valuse will be imputed
        sigmoid_correction: bool
            should sigmoid correction be applied (sigmoid function be applied
            to model`s scores)
        sigmoid_const: float
            intercept of sigmoid
        return_plain_results: bool
            should raw results (with sigmoid correction) be returned from predict
            added for speed optimization
        kwargs

        Returns
        -------
        np.ndarray
            2D numpy array which contains (variant, score) pair in each row
        """

        encoded_variants = \
            self._encode_variants_single_givens(
                variants=variants, givens=givens, noise=np.random.rand())

        encoded_variants_to_np_method = \
            cfe.encoded_variants_to_np if improve_settings.USE_CYTHON_BACKEND \
            else encoded_variants_to_np

        missings_filled_v = \
            encoded_variants_to_np_method(
                encoded_variants=encoded_variants,
                feature_names=self.model_feature_names)

        scores = \
            self.model.predict(
                DMatrix(
                    missings_filled_v, feature_names=self.model_feature_names))\
            .astype('float64')

        if sigmoid_correction:
            scores = sigmoid(scores, logit_const=sigmoid_const)

        scores += \
            np.array(
                np.random.rand(len(encoded_variants)), dtype='float64') * \
            self.TIEBREAKER_MULTIPLIER

        # TODO left for debugging sake - delete when no more neede
        # assert any([fel == sel for sel in scores for fel in scores])

        if return_plain_results:
            return scores

        variants_w_scores_list = \
            np.array(
                [self._get_processed_score(score=score, variant=variant)
                 for score, variant in zip(scores, variants)])

        return variants_w_scores_list  # np.column_stack([variants, scores])

    def _get_model_objective(self) -> str:
        """
        Helper method for native xgboost -> retrieves objective from booster
        object

        Returns
        -------
        str
            string representing task (check SUPPORTED_OBJECTIVES for currently
            supported tasks)

        """
        model_objective = ''
        # print(json.loads(self.model.save_config()))
        try:
            model_cfg = json.loads(self.model.save_config())
            model_objective = \
                model_cfg['learner']['learner_train_param']['objective'] \
                .split(':')[0]
        except Exception as exc:
            print('Cannot extract objective info from booster`s config')

        if model_objective not in self.SUPPORTED_OBJECTIVES:
            raise ValueError(
                'Unsupported booster objective: {}'.format(model_objective))
        return model_objective

    def _get_processed_score(self, score, variant):
        """
        Prepares 'result row' based on booster`s objective

        Parameters
        ----------
        score: float or np.ndarray
            single observation`s score
        variant: Dict[str, object]
            single obervation`s input

        Returns
        -------
        List[Number]
            list representing single row:
            [<variant>, <variant`s score>, <class>] (0 for regression task)

        """

        if self.model_objective == 'reg':
            return [variant, float(score), int(0)]
        elif self.model_objective == 'binary':
            return [variant, float(score), int(1)]
        elif self.model_objective == 'multi':
            return [variant, float(np.array(score).max()),
                    int(np.array(score).argmax())]

    def _encode_variants_single_givens(
            self, variants: Iterable, givens: dict or None,
            noise: float) -> Iterable:
        """
        Implemented as a XGBChooser helper method
        Cythonized loop over provided variants and a single givens dict.
        Returns array of encoded dicts.
        Parameters
        ----------
        variants: Iterable
            collection of input variants to be encoded
        givens: dict or None
            context to be encoded with variants
        noise: float
            noise param from 0-1 uniform distribution

        Returns
        -------
        np.ndarray
            array of encoded dicts
        """

        if not (isinstance(givens, dict) or givens is None or givens is {}):
            raise TypeError(
                'Unsupported givens` type: {}'.format(type(givens)))
            # process with single context

        # if improve_settings.USE_CYTHON_BACKEND:
        if improve_settings.USE_CYTHON_BACKEND:
            if isinstance(variants, list):
                used_variants = variants
            elif isinstance(variants, tuple) or isinstance(variants, np.ndarray):
                used_variants = list(variants)
            else:
                raise TypeError(
                    'Variants are of a wrong type: {}'.format(type(variants)))

            return cfe.encode_variants_single_givens(
                variants=used_variants, givens=givens, noise=noise,
                variants_encoder=self.feature_encoder.encode_variant,
                givens_encoder=self.feature_encoder.encode_givens)
        else:
            encoded_variants = np.empty(len(variants), dtype=object)

            encoded_givens = self.feature_encoder.encode_givens(givens=givens, noise=noise)

            encoded_variants[:] = [
                self.feature_encoder.encode_variant(
                    variant=variant, noise=noise, into=deepcopy(encoded_givens))
                for variant in variants]

            return encoded_variants


if __name__ == '__main__':

    mlmc = BasicNativeXGBChooser()

    # test_model_pth = '../artifacts/models/12_11_2020_verses_conv.xgb'
    # test_model_pth = "https://improve-v5-resources-prod-models-117097735164.s3-us-west-2.amazonaws.com/models/mindful/latest/improve-stories-2.0.xgb.gz"
    test_model_pth = "/Users/os/Downloads/model.gz"
    test_model_pth = \
        '/home/kw/Projects/upwork/python-sdk/improveai/artifacts/models/dummy_v6.xgb'
    mlmc.load_model(input_model_src=test_model_pth)

    # with open('../artifacts/test_artifacts/model.json', 'r') as mj:
    #     json_str = mj.readline()
    #     model_metadata = json.loads(json_str)

    with open('../artifacts/test_artifacts/context.json', 'r') as mj:
        json_str = mj.readline()
        context = json.loads(json_str)

    with open('../artifacts/test_artifacts/meditations.json', 'r') as vj:
        json_str = vj.readlines()
        variants = json.loads(''.join(json_str))

    # features_count = \
    #     len(model_metadata["table"][1])
    # feature_names = list(
    #     map(lambda i: 'f{}'.format(i), range(0, features_count)))
    #
    # sample_variants = [{"arrays": [1 for el in range(0, features_count + 10)]},
    #                    {"arrays": [el + 2 for el in range(0, features_count + 10)]},
    #                    {"arrays": [el for el in range(0, features_count + 10)]}]
    # 0.3775409052734039

    # single_score = mlmc.score(
    #     variant=variants[0], givens=context, sigmoid_correction=True)
    # print('single score')
    # print(single_score)

    st = time()
    batch_size = 100
    for _ in range(batch_size):
        score_all = mlmc.score(
            variants=variants, givens=context, sigmoid_correction=True,
            return_plain_results=True)
        score_all[-1] = 100
        score_all[::-1].sort()
        best = score_all[0]
    et = time()
    print(score_all[:10])
    print((et - st) / batch_size)
    input('score_all')

    score_all = mlmc.score(
        variants=variants, givens=context, sigmoid_correction=False,
        return_plain_results=False)

    sorted = mlmc.sort(variants_w_scores=score_all)
    print('sorted')
    print(sorted)

    choose = mlmc.choose(variants_w_scores=score_all)
    print('choose')
    print(choose)
