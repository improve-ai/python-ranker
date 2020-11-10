from copy import deepcopy
import json
import numpy as np
import pickle
from time import time
from typing import Dict, List
from xgboost import Booster, DMatrix
from xgboost.core import XGBoostError

from choosers.basic_choosers import BasicChooser
from encoders.feature_encoder import FeatureEncoder
from utils.gen_purp_utils import constant, append_prfx_to_dict_keys, \
    impute_missing_dict_keys


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

    @property
    def lookup_table_key(self) -> str:
        return self._lookup_table_key

    @lookup_table_key.setter
    def lookup_table_key(self, new_val: str):
        self._lookup_table_key = new_val

    @property
    def seed_key(self) -> str:
        return self._seed_key

    @seed_key.setter
    def seed_key(self, new_val: str):
        self._seed_key = new_val

    @property
    def model_objective(self) -> str:
        return self._model_objective

    @model_objective.setter
    def model_objective(self, new_val: str):
        self._model_objective = new_val

    @constant
    def SUPPORTED_OBJECTIVES() -> list:
        return ['reg', 'binary', 'multi']

    def __init__(
            self, mlmodel_metadata_key: str = 'json',
            lookup_table_key: str = 'table', seed_key: str = 'model_seed'):
        self.model = None
        self.model_metadata_key = mlmodel_metadata_key
        self.feature_encoder = None
        self.model_metadata = None
        self.lookup_table_key = lookup_table_key
        self.seed_key = seed_key
        self.model_objective = None

    def load_model(self, pth_to_model: str, verbose: bool = True):
        """
        Loads desired model from input path.

        Parameters
        ----------
        pth_to_model: str
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
                print('Attempting to load: {} model'.format(pth_to_model))
            self.model = Booster()
            self.model.load_model(pth_to_model)
            if verbose:
                print('Model: {} successfully loaded'.format(pth_to_model))
        except XGBoostError as xgbe:
            if verbose:
                print('Attempting to read via pickle interface')
            with open(pth_to_model, 'rb') as xgbl:
                self.model = pickle.load(xgbl)
        except Exception as exc:
            if verbose:
                print(
                    'When attempting to load the mode: {} the following error '
                    'occured: {}'.format(pth_to_model, exc))

        self.model_metadata = self._get_model_metadata()
        self.feature_encoder = self._get_feature_encoder()
        self.model_objective = self._get_model_objective()

    def _get_model_metadata(self) -> dict:

        assert 'user_defined_metadata' in self.model.attributes().keys()
        user_defined_metadata_str = self.model.attr('user_defined_metadata')
        user_defined_metadata = json.loads(user_defined_metadata_str)
        assert self.model_metadata_key in user_defined_metadata.keys()

        return user_defined_metadata[self.model_metadata_key]

    def _get_missings_filled_variants(
            self, context: Dict[str, object], all_feats_count: int):
        return np.array([
            context[el] if context.get(el, None) is not None
            else np.nan for el in np.arange(0, all_feats_count, 1)])\
            .reshape(1, all_feats_count)

    def _get_nan_filled_encoded_variants(
            self, variant: Dict[str, object], context: Dict[str, object],
            all_feats_count: int):

        context_copy = deepcopy(context)
        enc_variant = self.feature_encoder.encode_features({'variant': variant})
        context_copy.update(enc_variant)

        missings_filled_v = \
            self._get_missings_filled_variants(
                context=context_copy, all_feats_count=all_feats_count)

        return missings_filled_v

    def score_all(
            self, variants: List[Dict[str, object]],
            context: Dict[str, object],
            model_metadata: Dict[str, object] = None,
            lookup_table_key: str = "table",
            lookup_table_features_idx: int = 1, seed_key: str = "model_seed",
            mlmodel_score_res_key: str = 'target',
            mlmodel_class_proba_key: str = 'classProbability',
            target_class_label: int = 1, imputer_value: float = np.nan,
            **kwargs) -> np.ndarray:
        """
        Scores all provided variants
        Parameters
        ----------
        variants: list
            list of variants to scores
        context: dict
            context dict needed for encoding
        context_table_key: str
            context key storing lookup table
        context_table_features_idx: int
            index of a lookup table storing features encoding info (?)
        seed_key: str
            context key storing model seed
        Returns
        -------
        np.ndarray
            2D numpy array which contains (variant, score) pair in each row
        """

        encoded_context = \
            self.feature_encoder.encode_features({'context': context})

        all_feats_count = self._get_features_count()
        all_feat_names = \
            np.array(['f{}'.format(el) for el in range(all_feats_count)])

        encoded_variants = \
            np.array([self._get_nan_filled_encoded_variants(
                variant=v, context=encoded_context,
                all_feats_count=all_feats_count)
                for v in variants]) \
            .reshape((len(variants), len(all_feat_names)))

        scores = \
            self.model.predict(
                DMatrix(encoded_variants, feature_names=all_feat_names))
        # return scores

        # TODO this needs either to be flagged or sped up
        variants_w_scores_list = \
            np.array(
                [self._get_processed_score(score=score, variant=variant)
                 for score, variant in zip(scores, variants)])

        return variants_w_scores_list  # np.column_stack([variants, scores])

    def score(
            self, variant: Dict[str, object],
            context: Dict[str, object],
            model_metadata: Dict[str, object] = None,
            lookup_table_key: str = "table",
            lookup_table_features_idx: int = 1,
            seed_key: str = "model_seed",
            imputer_value: float = np.nan, **kwargs) -> list:

        encoded_context = \
            self.feature_encoder.encode_features({'context': context})
        encoded_features = \
            self.feature_encoder.encode_features({'variant': variant})

        all_encoded_features = deepcopy(encoded_context)
        all_encoded_features.update(encoded_features)

        all_feats_count = self._get_features_count()
        all_feat_names = \
            np.array(['f{}'.format(el) for el in range(all_feats_count)])

        missings_filled_v = \
            self._get_missings_filled_variants(
                context=encoded_features, all_feats_count=all_feats_count)

        single_score = \
            self.model \
                .predict(
                    DMatrix(missings_filled_v, feature_names=all_feat_names))

        # print(single_score_list)

        score = \
            self._get_processed_score(score=single_score, variant=variant)
        return score

    def _get_model_objective(self) -> str:
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

        if self.model_objective == 'reg':
            return [variant, float(score), int(0)]
        elif self.model_objective == 'binary':
            return [variant, float(score), int(1)]
        elif self.model_objective == 'multi':
            return [variant, float(np.array(score).max()), int(np.array(score).argmax())]


if __name__ == '__main__':

    mlmc = BasicNativeXGBChooser()

    test_model_pth = '../artifacts/test_artifacts/model_w_metadata.xgb'
    mlmc.load_model(pth_to_model=test_model_pth)

    with open('../artifacts/test_artifacts/model.json', 'r') as mj:
        json_str = mj.readline()
        model_metadata = json.loads(json_str)

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

    single_score = mlmc.score(
        variant=variants[0], context=context)
    print('single score')
    print(single_score)

    st = time()
    score_all = mlmc.score_all(
        variants=variants, context=context, model_metadata=model_metadata)
    et = time()
    print(et - st)
    print('score_all')
    print(score_all)

    sorted = mlmc.sort(variants_w_scores=score_all)
    print('sorted')
    print(sorted)

    choose = mlmc.choose(variants_w_scores=score_all)
    print('choose')
    print(choose)
