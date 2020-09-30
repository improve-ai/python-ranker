from copy import deepcopy
import json
import numpy as np
from typing import Dict, List
from xgboost import Booster, DMatrix

from choosers.basic_choosers import BasicChooser
from utils.gen_purp_utils import constant, append_prfx_to_dict_keys, \
    impute_missing_dict_keys


class BasicNativeXGBChooser(BasicChooser):

    @property
    def model(self) -> Booster:
        return self._model

    @model.setter
    def model(self, new_val: Booster):
        self._model = new_val

    @constant
    def SUPPORTED_OBJECTIVES() -> list:
        return ['reg', 'binary', 'multi']

    def __init__(self):
        self.model = None

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
        except Exception as exc:
            if verbose:
                print(
                    'When attempting to load the mode: {} the following error '
                    'occured: {}'.format(pth_to_model, exc))

    def _get_model_metadata(
            self, model_metadata: Dict[str, object] = None) -> dict:

        ml_meta = model_metadata

        if not ml_meta:
            raise ValueError('Model metadata is empty!')

        ret_ml_meta = \
            json.loads(ml_meta) if isinstance(ml_meta, str) else ml_meta

        return ret_ml_meta

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

        encoded_features = []

        for variant in variants:

            rnmd_all_encoded_features = \
                self._get_encoded_context_w_variant(
                    variant=variant, context=context,
                    model_metadata=model_metadata,
                    lookup_table_key=lookup_table_key, seed_key=seed_key)
            encoded_features.append(rnmd_all_encoded_features)

        # all_present_feature_names = list(np.unique(encoded_features_names))

        all_feature_names = \
            self._get_feature_names(
                model_metadata=model_metadata, lookup_table_key=lookup_table_key,
                lookup_table_features_idx=lookup_table_features_idx)

        imputed_encoded_features = []
        for single_variant_features in encoded_features:
            imputed_single_variant_features = \
                impute_missing_dict_keys(
                    all_des_keys=all_feature_names,
                    imputed_dict=single_variant_features,
                    imputer_value=imputer_value)
            imputed_encoded_features.append(
                [imputed_single_variant_features[key]
                 for key in all_feature_names])

        scores = \
            self.model.predict(
                DMatrix(
                    np.array(imputed_encoded_features)
                        .reshape(len(imputed_encoded_features),
                                 len(imputed_encoded_features[0])),
                    feature_names=all_feature_names))

        assert len(scores) == len(variants)

        variants_w_scores_list = []

        for score, variant in zip(scores, variants):
            single_score_list = score if isinstance(score, list) else [score]
            variants_w_scores_list.append(
                self._get_processed_score(
                    scores=single_score_list, variant=variant))

        return np.array(variants_w_scores_list)  # np.column_stack([variants, scores])

    def _get_encoded_context_w_variant(
            self, variant: Dict[str, object],
            context: Dict[str, object],
            model_metadata: Dict[str, object] = None,
            lookup_table_key: str = "table",
            seed_key: str = "model_seed") -> dict:

        encoded_context = \
            self._get_encoded_features(
                encoded_dict={'context': context},
                model_metadata=model_metadata,
                lookup_table_key=lookup_table_key, seed_key=seed_key)

        encoded_features = \
            self._get_encoded_features(
                encoded_dict={'variant': variant},
                model_metadata=model_metadata,
                lookup_table_key=lookup_table_key, seed_key=seed_key)

        all_encoded_features = deepcopy(encoded_context)
        all_encoded_features.update(encoded_features)

        rnmd_all_encoded_features = \
            append_prfx_to_dict_keys(input_dict=all_encoded_features, prfx='f')

        return rnmd_all_encoded_features

    def score(
            self, variant: Dict[str, object],
            context: Dict[str, object],
            model_metadata: Dict[str, object] = None,
            lookup_table_key: str = "table",
            lookup_table_features_idx: int = 1,
            seed_key: str = "model_seed",
            imputer_value: float = np.nan) -> list:

        rnmd_all_encoded_features = \
            self._get_encoded_context_w_variant(
                variant=variant, context=context,
                model_metadata=model_metadata,
                lookup_table_key=lookup_table_key, seed_key=seed_key)

        all_feature_names = \
            self._get_feature_names(
                model_metadata=model_metadata, lookup_table_key=lookup_table_key,
                lookup_table_features_idx=lookup_table_features_idx)

        imptd_missing_encoded_features = \
            impute_missing_dict_keys(
                    all_des_keys=all_feature_names,
                    imputed_dict=rnmd_all_encoded_features,
                    imputer_value=imputer_value)

        input_list = \
            [imptd_missing_encoded_features[key] for key in all_feature_names]

        single_score_list = \
            self.model\
                .predict(DMatrix(
                    data=np.array(input_list).reshape(1,len(input_list)),
                    feature_names=all_feature_names))

        score = \
            self._get_processed_score(scores=single_score_list, variant=variant)

        return score

    def _get_model_objective(self) -> str:
        model_objective = ''
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

    def _get_processed_score(self, scores, variant):

        model_objective = self._get_model_objective()

        if model_objective == 'reg':
            return [variant, float(scores[0]), int(0)]
        elif model_objective == 'binary':
            return [variant, float(scores[0]), int(1)]
        elif model_objective == 'multi':
            return [variant, float(np.array(scores).max()), int(np.array(scores).argmax())]


if __name__ == '__main__':

    mlmc = BasicNativeXGBChooser()

    test_model_pth = '../test_artifacts/model.xgb'
    mlmc.load_model(pth_to_model=test_model_pth)

    with open('../test_artifacts/model.json', 'r') as mj:
        json_str = mj.readline()
        model_metadata = json.loads(json_str)

    with open('../test_artifacts/context.json', 'r') as mj:
        json_str = mj.readline()
        context = json.loads(json_str)

    with open('../test_artifacts/meditations.json', 'r') as vj:
        json_str = vj.readlines()
        variants = json.loads(''.join(json_str))

    features_count = \
        len(model_metadata["table"][1])
    feature_names = list(
        map(lambda i: 'f{}'.format(i), range(0, features_count)))

    sample_variants = [{"arrays": [1 for el in range(0, features_count + 10)]},
                       {"arrays": [el + 2 for el in range(0, features_count + 10)]},
                       {"arrays": [el for el in range(0, features_count + 10)]}]

    single_score = mlmc.score(
        variant=sample_variants[0], context=context, model_metadata=model_metadata)
    print('single score')
    print(single_score)

    score_all = mlmc.score_all(
        variants=sample_variants, context=context, model_metadata=model_metadata)
    print('score_all')
    print(score_all)

    sorted = mlmc.sort(variants_w_scores=score_all)
    print('sorted')
    print(sorted)

    choose = mlmc.choose(variants_w_scores=score_all)
    print('choose')
    print(choose)
