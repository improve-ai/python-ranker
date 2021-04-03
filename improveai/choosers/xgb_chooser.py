from Cython.Build import cythonize
import json
from numbers import Number
import numpy as np
import os
from setuptools import Extension
import pickle
import pyximport
from time import time
from typing import Dict, List
from traceback import print_exc
from xgboost import Booster, DMatrix
from xgboost.core import XGBoostError

try:
    # This is done for backward compatibilty
    from coremltools.models.utils import macos_version
except Exception as exc:
    from coremltools.models.utils import _macos_version as macos_version

if not macos_version():

    rel_pth_prfx = \
        os.sep.join(str(os.path.relpath(__file__)).split(os.sep)[:-1])

    pth_str = \
        '{}{}choosers_cython_utils/fast_feat_enc.pyx'\
        .format(
            os.sep.join(str(os.path.relpath(__file__)).split(os.sep)[:-1]),
            '' if not rel_pth_prfx else os.sep)

    print(pth_str)

    fast_feat_enc_ext = \
        Extension(
            'fast_feat_enc',
            sources=[pth_str],
            define_macros=[
                ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            include_dirs=[np.get_include()])

    pyximport.install(
        setup_args={
            'install_requires': ["numpy"],
            'ext_modules': cythonize(
                fast_feat_enc_ext,
                language_level="3")})

    import choosers.choosers_cython_utils.fast_feat_enc as ffe

from choosers.basic_choosers import BasicChooser
# from feature_encoders.v5 import FeatureEncoder
from feature_encoders.v6 import FeatureEncoder
from utils.general_purpose_utils import constant, sigmoid


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
    def seed_key(self) -> str:
        return self._seed_key

    @seed_key.setter
    def seed_key(self, new_val: str):
        self._seed_key = new_val

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

    @constant
    def SUPPORTED_OBJECTIVES() -> list:
        return ['reg', 'binary', 'multi']

    @constant
    def TIEBREAKER_MULTIPLIER() -> float:
        return 1e-6

    def __init__(
            self, mlmodel_metadata_key: str = 'json',
            model_feature_names_key: str = 'feature_names',
            seed_key: str = 'model_seed'):
        self.model = None
        self.model_metadata_key = mlmodel_metadata_key
        self.feature_encoder = None
        self.model_metadata = None
        # self.lookup_table_key = lookup_table_key
        self.model_feature_names_key = model_feature_names_key
        self.model_feature_names = np.empty(shape=(1,))
        self.seed_key = seed_key
        self.model_objective = None

    def load_model(self, inupt_model_src: str, verbose: bool = True):
        """
        Loads desired model from input path.

        Parameters
        ----------
        inupt_model_src: str
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
                print('Attempting to load: {} model'.format(inupt_model_src))

            raw_model_src = self._get_model_src(model_src=inupt_model_src)

            model_src = \
                raw_model_src if isinstance(raw_model_src, str) \
                else bytearray(raw_model_src)

            self.model = Booster()
            self.model.load_model(model_src)
            if verbose:
                print('Model: {} successfully loaded'.format(inupt_model_src))
        except XGBoostError as xgbe:
            if verbose:
                print('Attempting to read via pickle interface')
            with open(inupt_model_src, 'rb') as xgbl:
                self.model = pickle.load(xgbl)
            print('### TRACEBACK ###')
            print_exc()
        except Exception as exc:
            if verbose:
                print(
                    'When attempting to load the model: {} the following error '
                    'occured: {}'.format(inupt_model_src, exc))
            print_exc()

        self.model_metadata = self._get_model_metadata()
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

    def score_all(
            self, variants: List[Dict[str, object]],
            context: Dict[str, object],
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
        context: dict
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

        # encoded_context = \
        #     self.feature_encoder.encode_features({'context': context})
        #
        # all_feats_count = self._get_features_count()
        if macos_version():
            raise NotImplementedError(
                'Running on macOS - make sure cython works!')

            # TODO old version
            # all_feat_names = \
            #     np.array(['f{}'.format(el) for el in range(all_feats_count)])
            # encoded_variants = \
            #     np.array([self._get_nan_filled_encoded_variant(
            #         variant=v, context=encoded_context,
            #         all_feats_count=all_feats_count, missing_filler=imputer_value)
            #         for v in variants]) \
            #     .reshape((len(variants), len(all_feat_names)))
        else:
            # TODO old version
            # all_feat_names = np.asarray(ffe.get_all_feat_names(all_feats_count))
            # # st1 = time()
            # # TODO check if passing encode_features() method makes this faster !!!
            # encoded_variants = \
            #     np.asarray(ffe.get_nan_filled_encoded_variants(
            #         np.array(variants, dtype=dict), encoded_context, all_feats_count,
            #         self.feature_encoder, imputer_value))
            encoded_variants = \
                self.feature_encoder.encode_variants(
                    variants=np.array(variants, dtype=dict), contexts=context,
                    noise=np.random.rand())

            missings_filled_v = \
                self.feature_encoder.fill_missing_features(
                    encoded_variants=encoded_variants,
                    feature_names=self.model_feature_names)

        # et1 = time()
        # print('Encoding took: {}'.format(et1 - st1))

        # print(encoded_variants[0])
        # print(all_feat_names)

        # st1 = time()
        scores = \
            self.model.predict(
                DMatrix(
                    missings_filled_v, feature_names=self.model_feature_names))\
            .astype('float64')
        # et1 = time()
        # print('Predicting took: {}'.format(et1 - st1))

        if sigmoid_correction:
            scores = sigmoid(scores, logit_const=sigmoid_const)

        # st2 = time()
        scores += \
            np.array(
                np.random.rand(len(encoded_variants)), dtype='float64') * \
            self.TIEBREAKER_MULTIPLIER
        # et2 = time()
        # print('Breaking ties took {}'.format(et2 - st2))
        # input('sanity check')

        # TODO left for debugging sake - delete when no more neede
        # assert any([fel == sel for sel in scores for fel in scores])
        # print(scores)
        # input('sanity check')

        if return_plain_results:
            return scores

        variants_w_scores_list = \
            np.array(
                [self._get_processed_score(score=score, variant=variant)
                 for score, variant in zip(scores, variants)])

        return variants_w_scores_list  # np.column_stack([variants, scores])

    def score(
            self, variant: Dict[str, object],
            context: Dict[str, object],
            imputer_value: float = np.nan,
            sigmoid_correction: bool = False, sigmoid_const: float = 0.5,
            **kwargs) -> list:

        """
        Scores single variant

        Parameters
        ----------
        variant: Dict[str, object]
            single observation to be scored
        context: Dict[str, object]
            scoring context dict
        imputer_value: float
            value to impute missings with
        sigmoid_correction: bool
            should sigmoid function be applied to the predict`s result
        sigmoid_const: float
            intercept term of sigmoid
        kwargs

        Returns
        -------

        """

        # encoded_context = \
        #     self.feature_encoder.encode_context({'context': context})
        # # encoded_jsonlines = \
        # #     self.feature_encoder.encode_features({'variant': variant})
        # #
        # # all_encoded_features = deepcopy(encoded_context)
        # # all_encoded_features.update(encoded_jsonlines)
        #
        # all_feats_count = self._get_features_count()
        # all_feat_names = \
        #     np.array(['f{}'.format(el) for el in range(all_feats_count)])
        # # feat_names = np.array(['f{}'.format(el) for el in all_encoded_features.keys()])
        # # values = np.array([el for el in all_encoded_features.values()])
        # # vals_count = len(values)
        #
        # # st = time()
        # # missings_filled_v = \
        # #     self._get_missings_filled_variants(
        # #         input_dict=encoded_jsonlines, all_feats_count=all_feats_count,
        # #         missing_filler=imputer_value)
        #
        # missings_filled_v = \
        #     self._get_nan_filled_encoded_variant(
        #         variant=variant, context=encoded_context,
        #         all_feats_count=all_feats_count, missing_filler=imputer_value)
        # # et = time()
        # # print('Encoding feature_names took: {}'.format(et - st))

        noise = np.random.rand()

        encoded_variant = \
            self.feature_encoder.encode_variants(
                variants=np.array([variant]), contexts=context, noise=noise)

        missings_filled_v = \
            self.feature_encoder.fill_missing_features(
                encoded_variants=encoded_variant,
                feature_names=self.model_feature_names)

        # st1 = time()
        single_score = \
            self.model \
                .predict(
                    DMatrix(
                        missings_filled_v.reshape(1, -1),
                        # if missings_filled_v.shape == (1, all_feats_count)
                        # else missings_filled_v.reshape((1, all_feats_count)),
                        feature_names=self.model_feature_names, missing=np.nan
                    )
            ).astype('float64')
        # et1 = time()
        # print('Predicting took: {}'.format(et1 - st1))
        # input('sanity check')

        if sigmoid_correction:
            single_score = sigmoid(single_score, logit_const=sigmoid_const)

        single_score += \
            np.array(
                np.random.rand(), dtype='float64') * self.TIEBREAKER_MULTIPLIER

        score = \
            self._get_processed_score(score=single_score, variant=variant)
        return score

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


if __name__ == '__main__':

    mlmc = BasicNativeXGBChooser()

    # test_model_pth = '../artifacts/models/12_11_2020_verses_conv.xgb'
    # test_model_pth = "https://improve-v5-resources-prod-models-117097735164.s3-us-west-2.amazonaws.com/models/mindful/latest/improve-stories-2.0.xgb.gz"
    test_model_pth = "/Users/os/Downloads/model.gz"
    test_model_pth = \
        '/home/os/Projects/upwork/python-sdk/improveai/artifacts/models/dummy_v6.xgb'
    mlmc.load_model(inupt_model_src=test_model_pth)

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

    single_score = mlmc.score(
        variant=variants[0], context=context, sigmoid_correction=True)
    print('single score')
    print(single_score)

    st = time()
    batch_size = 100
    for _ in range(batch_size):
        score_all = mlmc.score_all(
            variants=variants, context=context, sigmoid_correction=True,
            return_plain_results=True)
        score_all[-1] = 100
        score_all[::-1].sort()
        best = score_all[0]
    et = time()
    print(score_all[:10])
    print((et - st) / batch_size)
    input('score_all')

    score_all = mlmc.score_all(
        variants=variants, context=context, sigmoid_correction=False,
        return_plain_results=False)

    sorted = mlmc.sort(variants_w_scores=score_all)
    print('sorted')
    print(sorted)

    choose = mlmc.choose(variants_w_scores=score_all)
    print('choose')
    print(choose)
