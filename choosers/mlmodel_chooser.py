import coremltools as ct
from copy import deepcopy
import json
import numpy as np
from time import time
from typing import Dict, List

from choosers.basic_choosers import BasicChooser
from encoders.feature_encoder import FeatureEncoder
from utils.gen_purp_utils import constant, sigmoid


class BasicMLModelChooser(BasicChooser):

    @property
    def model(self) -> ct.models.MLModel:
        return self._model

    @model.setter
    def model(self, new_val: ct.models.MLModel):
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
    def model_metadata_key(self) -> str:
        return self._model_metadata_key

    @model_metadata_key.setter
    def model_metadata_key(self, new_val: str):
        self._model_metadata_key = new_val

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

    @constant
    def TIEBREAKER_MULTIPLIER() -> float:
        return 1e-7

    def __init__(
            self, mlmodel_metadata_key: str = 'json',
            lookup_table_key: str = 'table', seed_key: str = 'model_seed'):
        # initialize
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
            self.model = ct.models.MLModel(pth_to_model)
            if verbose:
                print('Model: {} successfully loaded'.format(pth_to_model))
        except Exception as exc:
            if verbose:
                print(
                    'When attempting to load the mode: {} the following error '
                    'occured: {}'.format(pth_to_model, exc))

        self.model_metadata = self._get_model_metadata()
        self.feature_encoder = self._get_feature_encoder()

    def _get_model_metadata(self) -> dict:
        """
        Gets model metadata either from dict or from model attributes

        Returns
        -------
        dict
            metadata dict

        """

        assert hasattr(self.model, 'user_defined_metadata')
        assert self.model_metadata_key in self.model.user_defined_metadata.keys()
        return json.loads(
            self.model.user_defined_metadata[self.model_metadata_key])

    def score(
            self, variant: Dict[str, object],
            context: Dict[str, object] = None,
            encoded_context: Dict[str, object] = None,
            mlmodel_score_res_key: str = 'target',
            mlmodel_class_proba_key: str = 'classProbability',
            target_class_label: int = 1, imputer_value: float = np.nan,
            sigmoid_correction: bool = True,
            sigmoid_const: float = 0.5, return_plain_results: bool = False,
            **kwargs) -> list:
        """
        Performs scoring of a single variant using provided context and loaded
        model

        Parameters
        ----------
        variant: dict
            scored case / row as a dict
        context: dict
            dict with lookup table and seed
        encoded_context: Dict[str, float]
            already encoded context dict
        mlmodel_score_res_key: str
            key in mlmodel results dict under which result float is stored
        mlmodel_class_proba_key: str
            key storing dict with probas in mlmodel results dict
        target_class_label: str
            class label which is the <class 1>
        imputer_value: float
            value with which nans will be imputed
        sigmoid_correction: bool
            should sigmoid correction be applied to results of predict
        sigmoid_const: float
            intercept term of sigmoid
        kwargs

        Returns
        -------
        np.ndarray
            score of provided variant

        """

        if not encoded_context:
            encoded_context = \
                self.feature_encoder.encode_features({'context': context})
        # encoded_features = \
        #     self.feature_encoder.encode_features({'variant': variant})
        #
        # all_encoded_features = deepcopy(encoded_context)
        # all_encoded_features.update(encoded_features)

        all_feats_count = self._get_features_count()
        all_feat_names = \
            np.array(['f{}'.format(el) for el in range(all_feats_count)])

        # rename features
        # missings_filled_v = \
        #     self._get_missings_filled_variants(
        #         input_dict=encoded_features, all_feats_count=all_feats_count,
        #         missing_filler=imputer_value)

        # _get_nan_filled_encoded_variant

        missings_filled_v = \
            self._get_nan_filled_encoded_variant(
                variant=variant, context=encoded_context,
                all_feats_count=all_feats_count, missing_filler=imputer_value)\
            .reshape((all_feats_count, ))

        assert len(all_feat_names) == len(missings_filled_v)

        score_dict = \
            self.model.predict(dict(zip(all_feat_names, missings_filled_v)))

        best_score = \
            self._get_processed_score(
                variant=variant, score_dict=score_dict,
                mlmodel_class_proba_key=mlmodel_class_proba_key,
                mlmodel_score_res_key=mlmodel_score_res_key,
                target_class_label=target_class_label,
                sigmoid_correction=sigmoid_correction,
                sigmoid_const=sigmoid_const)

        best_score[1] = \
            float(np.array(best_score[1], dtype='float64') +
                  np.array(np.random.rand(), dtype='float64') *
                  self.TIEBREAKER_MULTIPLIER)

        if return_plain_results:
            return best_score[1]

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
            mlmodel_score_res_key: str = 'target',
            mlmodel_class_proba_key: str = 'classProbability',
            target_class_label: int = 1, imputer_value: float = np.nan,
            sigmoid_const: float = 0.5,
            sigmoid_correction: bool = True,
            return_plain_results: bool = False,
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
        mlmodel_score_res_key: str
            key in mlmodel results dict under which result float is stored
        mlmodel_class_proba_key: str
            key storing dict with probas in mlmodel results dict
        target_class_label: str
            class label which is the <class 1>
        imputer_value: float
            value with which nans will be imputed
        sigmoid_correction: bool
            should sigmoid correction be applied
        sigmoid_const: float
            sigmoids intercept
        return_plain_results: boold
            should results without 'post-processing' be returned (for speed`s sake)
        kwargs
            kwargs

        Returns
        -------
        np.ndarray
            2D numpy array which contains (variant, score) pair in each row

        """

        encoded_context = \
            self.feature_encoder.encode_features({'context': context})

        scores = \
            np.array([self.score(
                variant=variant, context=context,
                encoded_context=encoded_context,
                mlmodel_score_res_key=mlmodel_score_res_key,
                mlmodel_class_proba_key=mlmodel_class_proba_key,
                target_class_label=target_class_label,
                imputer_value=imputer_value,
                sigmoid_correction=sigmoid_correction,
                sigmoid_const=sigmoid_const,
                return_plain_results=return_plain_results) for variant in variants])

        return scores


if __name__ == '__main__':

    mlmc = BasicMLModelChooser()

    test_model_pth = '../artifacts/test_artifacts/improve-messages-2.0-3.mlmodel'
    mlmc.load_model(pth_to_model=test_model_pth)

    with open('../artifacts/test_artifacts/context.json', 'r') as mj:
        json_str = mj.readline()
        context = json.loads(json_str)

    with open('../artifacts/test_artifacts/meditations.json', 'r') as vj:
        json_str = vj.readlines()
        variants = json.loads(''.join(json_str))

    res = mlmc.score(variant=variants[0], context=context)
    print('res')
    print(res)
    # 0.0012913734900291936
    # input('check')

    batch_size = 10
    print(len(variants))
    st = time()

    for _ in range(batch_size):
        # res = mlmc.score(variant=variants[0], context=context)
        res_all = \
            mlmc.score_all(variants=variants, context=context)
    et = time()
    print((et - st) / batch_size)
    input('sanity check')
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