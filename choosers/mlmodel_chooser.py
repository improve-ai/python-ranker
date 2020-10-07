import coremltools as ct
from copy import deepcopy
import json
import numpy as np
from typing import Dict, List

from choosers.basic_choosers import BasicChooser
from utils.gen_purp_utils import append_prfx_to_dict_keys, \
    impute_missing_dict_keys, sigmoid


class BasicMLModelChooser(BasicChooser):

    @property
    def model(self) -> ct.models.MLModel:
        return self._model

    @model.setter
    def model(self, new_val: ct.models.MLModel):
        self._model = new_val

    @property
    def mlmodel_metadata_key(self) -> str:
        return self._mlmodel_metadata_key

    @mlmodel_metadata_key.setter
    def model_metadata_key(self, new_val: str):
        self._mlmodel_metadata_key = new_val

    def __init__(self, mlmodel_metadata_key: str = 'json'):
        # initialize
        self.model = None
        self.model_metadata_key = mlmodel_metadata_key

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
            self.model = ct.models.MLModel(pth_to_model)
            if verbose:
                print('Model: {} successfully loaded'.format(pth_to_model))
        except Exception as exc:
            if verbose:
                print(
                    'When attempting to load the mode: {} the following error '
                    'occured: {}'.format(pth_to_model, exc))

    def _get_model_metadata(
            self, model_metadata: Dict[str, object] = None) -> dict:
        """
        Gets model metadata either from dict or from model attributed

        Parameters
        ----------
        model_metadata: dict
            user provided dict with metadata

        Returns
        -------
        dict
            metadata dict

        """

        ml_meta = model_metadata

        if not ml_meta:
            assert hasattr(self.model, 'user_defined_metadata')
            model_metadata = self.model.user_defined_metadata
            assert self.mlmodel_metadata_key in model_metadata.keys()
            ml_meta = model_metadata[self.mlmodel_metadata_key]

        ret_ml_meta = \
            json.loads(ml_meta) if isinstance(ml_meta, str) else ml_meta

        return ret_ml_meta

    def score(
            self, variant: Dict[str, object],
            context: Dict[str, object] = None,
            encoded_context: Dict[str, object] = None,
            model_metadata: Dict[str, object] = None,
            lookup_table_key: str = "table",
            lookup_table_features_idx: int = 1, seed_key: str = "model_seed",
            mlmodel_score_res_key: str = 'target',
            mlmodel_class_proba_key: str = 'classProbability',
            target_class_label: int = 1, imputer_value: float = np.nan,
            sigmoid_correction: bool = True,
            sigmoid_const: float = 0.5, **kwargs) -> list:
        """
        Performs scoring of a single variant using provided context and loaded
        model

        Parameters
        ----------
        variant: dict
            scored case / row as a dict
        context: dict
            dict with lookup table and seed
        lookup_table_key: str
            context key storing lookup table
        lookup_table_features_idx: int
            index of a lookup table storing features encoding info (?)
        seed_key: str
            context key storing model seed
        mlmodel_score_res_key: str
            key in mlmodel results dict under which result float is stored
        mlmodel_class_proba_key: str
            key storing dict with probas in mlmodel results dict
        target_class_label: str
            class label which is the <class 1>
        imputer_value: float
            value with which nans will be imputed
        sigmoid_correction
        kwargs

        Returns
        -------
        np.ndarray
            score of provided variant

        """

        if not encoded_context:
            encoded_context = \
                self._get_encoded_features(
                    encoded_dict={'context': context},
                    model_metadata=model_metadata,
                    lookup_table_key=lookup_table_key, seed_key=seed_key)

        # print(encoded_context)
        # # 64: 0.025116502088750277, 46: 0.0255248728305653, 120: 0.025116502088750277, 237: 0.05000000074505806, 180: 80.0, 104: 1.0, 244: 1.0, 69: 1.0, 175: 1.0, 117: 0.025827039478405665, 136: 80.0, 98: 0.02553597409873252, 111: 0.027643948483414886, 211: 0.033118124819099, 220: 0.028558938988829326

        encoded_features = \
            self._get_encoded_features(
                encoded_dict={'variant': variant},
                model_metadata=model_metadata,
                lookup_table_key=lookup_table_key, seed_key=seed_key)

        all_encoded_features = deepcopy(encoded_context)
        all_encoded_features.update(encoded_features)
        rnmd_all_encoded_features = \
            append_prfx_to_dict_keys(input_dict=all_encoded_features, prfx='f')

        # rename features
        feature_names = \
            self._get_feature_names(
                model_metadata=model_metadata,
                lookup_table_key=lookup_table_key,
                lookup_table_features_idx=lookup_table_features_idx)

        imputed_encoded_features = \
            impute_missing_dict_keys(
                all_des_keys=feature_names,
                imputed_dict=rnmd_all_encoded_features,
                imputer_value=imputer_value)

        assert len(feature_names) == len(imputed_encoded_features.keys())

        score_dict = \
            self.model.predict(imputed_encoded_features)

        best_score = \
            self._get_processed_score(
                variant=variant, score_dict=score_dict,
                mlmodel_class_proba_key=mlmodel_class_proba_key,
                mlmodel_score_res_key=mlmodel_score_res_key,
                target_class_label=target_class_label,
                sigmoid_correction=sigmoid_correction,
                sigmoid_const=sigmoid_const)

        return best_score

    def _get_processed_score(
            self, variant, score_dict, mlmodel_class_proba_key,
            mlmodel_score_res_key, target_class_label,
            sigmoid_correction: bool = True,
            sigmoid_const: float = 0.5) -> list:
        """
        Getter for object which would be returned with scores

        Parameters
        ----------
        variant: dict
            dict with scored variatn
        score_dict: dict
            dict with results
        mlmodel_class_proba_key: str
            string with mlmodel probability key in results dict
        mlmodel_score_res_key: str
            string with score key in results dict
        target_class_label: object
            desired class label of a target class
        sigmoid_correction: bool
            should sigmoid correction be applied
        sigmoid_const: float
            sigmoids intercept

        Returns
        -------
        list
            list with processed results

        """

        if mlmodel_class_proba_key in score_dict.keys():

            if len(score_dict[mlmodel_class_proba_key].keys()) == 2:

                class_1_proba = \
                    score_dict[mlmodel_class_proba_key][target_class_label]
                if sigmoid_correction:
                    class_1_proba = \
                        sigmoid(x=class_1_proba, logit_const=sigmoid_const)

                return [variant, class_1_proba, target_class_label]

            class_1_probas = score_dict[mlmodel_class_proba_key].values()

            all_scores = \
                np.array([
                    [sigmoid(x=val, logit_const=sigmoid_const)
                     for val in class_1_probas] if sigmoid_correction
                    else list(score_dict[mlmodel_class_proba_key].values()),
                    list(score_dict[mlmodel_class_proba_key].keys())]).T
            return \
                [variant] + \
                all_scores[all_scores[:, 0] == all_scores[:, 0].max()] \
                .flatten().tolist()
        else:
            score = score_dict.get(mlmodel_score_res_key, None)
            if not score:
                raise KeyError(
                    'There was no key named: {} in result of predict() method'
                    .format(mlmodel_score_res_key))
            return [variant,
                    sigmoid(x=score, logit_const=sigmoid_const)
                    if sigmoid_correction else score, 0]

    def score_all(
            self, variants: List[Dict[str, object]],
            context: Dict[str, object],
            model_metadata: Dict[str, object] = None,
            lookup_table_key: str = "table",
            lookup_table_features_idx: int = 1, seed_key: str = "model_seed",
            mlmodel_score_res_key: str = 'target',
            mlmodel_class_proba_key: str = 'classProbability',
            target_class_label: int = 1, imputer_value: float = np.nan,
            sigmoid_const: float = 0.5,
            sigmoid_correction: bool = True,
            **kwargs) -> np.ndarray:
        """
        Scores all provided variants

        Parameters
        ----------
        variants: list
            list of variants to scores
        context: dic        variant: dict
            scored case / row as a dict
        context: dict
            dict with lookup table and seed
        lookup_table_key: str
            context key storing lookup table
        lookup_table_features_idx: int
            index of a lookup table storing features encoding info (?)
        seed_key: str
            context key storing model seed
        mlmodel_score_res_key: str
            key in mlmodel results dict under which result float is stored
        mlmodel_class_proba_key: str
            key storing dict with probas in mlmodel results dict
        target_class_label: str
            class label which is the <class 1>
        imputer_value: float
            value with which nans will be imputed
        sigmoid_correction
        kwargs
            context dict needed for encoding

        Returns
        -------
        np.ndarray
            2D numpy array which contains (variant, score) pair in each row

        """

        scores = []

        encoded_context = \
            self._get_encoded_features(
                encoded_dict={'context': context},
                model_metadata=model_metadata,
                lookup_table_key=lookup_table_key, seed_key=seed_key)

        for variant in variants:
            scores.append(
                self.score(
                    variant=variant, context=context,
                    encoded_context=encoded_context,
                    model_metadata=model_metadata,
                    lookup_table_key=lookup_table_key,
                    lookup_table_features_idx=lookup_table_features_idx,
                    seed_key=seed_key,
                    mlmodel_score_res_key=mlmodel_score_res_key,
                    mlmodel_class_proba_key=mlmodel_class_proba_key,
                    target_class_label=target_class_label,
                    imputer_value=imputer_value,
                    sigmoid_correction=sigmoid_correction,
                    sigmoid_const=sigmoid_const))

        ret_scores = np.array(scores)

        return ret_scores


if __name__ == '__main__':

    mlmc = BasicMLModelChooser()

    test_model_pth = '../test_artifacts/improve-messages-2.0-3.mlmodel'
    mlmc.load_model(pth_to_model=test_model_pth)

    with open('../test_artifacts/context.json', 'r') as mj:
        json_str = mj.readline()
        context = json.loads(json_str)

    with open('../test_artifacts/meditations.json', 'r') as vj:
        json_str = vj.readlines()
        variants = json.loads(''.join(json_str))

    res = mlmc.score(variant=variants[0], context=context)
    print('res')
    print(res)
    # input('check')

    res_all = \
        mlmc.score_all(variants=variants, context=context)
    print('res_all')

    srtd_variants_w_scores = mlmc.sort(variants_w_scores=res_all)
    print('srtd_variants_w_scores')
    for row in srtd_variants_w_scores.tolist():
        print('{} -> sigmoid({}) = {}'.format(row, row[1], round(1 / (1 + np.exp(0.5 - row[1])), 4)))

    best_choice = mlmc.choose(variants_w_scores=srtd_variants_w_scores)
    print('best_choice')
    print(best_choice)


# {28: 1.0, 33: 1.0, 17: 1.0, 39: 1.0, 42: 1.0, 11: 1.0, 37: 1.0, 9: 1.0, 31: 1.0, 38: 1.0, 41: 1.0, 21: 1.0, 3: 1.0, 43: 1.0, 22: 1.0, 2: 1.0, 8: 1.0, 14: 1.0, 40: 1.0, 25: 1.0, 20: 1.0, 30: 1.0, 16: 1.0, 4: 1.0, 19: 1.0, 6: 1.0, 23: 1.0, 26: 1.0, 24: 1.0, 27: 1.0, 18: 1.0, 29: 1.0, 10: 1.0, 1: 1.0, 32: 1.0, 13: 1.0, 36: 1.0, 34: 1.0, 7: 1.0, 0: 1.0, 12: 1.0, 35: 1.0, 15: 1.0, 5: 1.0}
# {28: 1.0, 33: 1.0, 17: 1.0, 39: 1.0, 42: 1.0, 11: 1.0, 37: 1.0, 9: 1.0, 31: 1.0, 38: 1.0, 41: 1.0, 21: 1.0, 3: 1.0, 43: 1.0, 22: 1.0, 2: 1.0, 8: 1.0, 14: 1.0, 40: 1.0, 25: 1.0, 20: 1.0, 30: 1.0, 16: 1.0, 4: 1.0, 19: 1.0, 6: 1.0, 23: 1.0, 26: 1.0, 24: 1.0, 27: 1.0, 18: 1.0, 29: 1.0, 10: 1.0, 1: 1.0, 32: 1.0, 13: 1.0, 36: 1.0, 34: 1.0, 7: 1.0, 0: 1.0, 12: 1.0, 35: 1.0, 15: 1.0, 5: 1.0}