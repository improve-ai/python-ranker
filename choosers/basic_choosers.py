from abc import ABC, abstractmethod
import json
import numpy as np
from typing import List, Dict

from encoders.feature_encoder import FeatureEncoder


class BasicChooser(ABC):
    @property
    @abstractmethod
    def model(self) -> object:
        # return self._model
        pass

    @model.setter
    @abstractmethod
    def model(self, new_val: object):
        pass
        # self._model = new_val

    @property
    @abstractmethod
    def model_metadata_key(self):
        pass

    @model_metadata_key.setter
    @abstractmethod
    def model_metadata_key(self, new_val: str):
        pass

    @property
    @abstractmethod
    def feature_encoder(self) -> FeatureEncoder:
        pass

    @feature_encoder.setter
    @abstractmethod
    def feature_encoder(self, new_val: FeatureEncoder):
        pass

    @property
    @abstractmethod
    def model_metadata(self) -> Dict[str, object]:
        pass

    @model_metadata.setter
    @abstractmethod
    def model_metadata(self, new_val: Dict[str, object]):
        pass

    @property
    @abstractmethod
    def lookup_table_key(self) -> str:
        pass

    @lookup_table_key.setter
    @abstractmethod
    def lookup_table_key(self, new_val: str):
        pass

    @property
    @abstractmethod
    def seed_key(self) -> str:
        pass

    @seed_key.setter
    @property
    def seed_key(self, new_val: str):
        pass

    @abstractmethod
    def load_model(self, pth_to_model, **kwargs):
        pass

    @abstractmethod
    def score(self, variant, context, lookup_table, **kwargs):
        pass

    @abstractmethod
    def score_all(self, variants, context, lookup_table, **kwargs):
        pass

    # @abstractmethod
    # def sort(self, variants_w_scores, **kwargs):
    #     pass
    #
    # @abstractmethod
    # def choose(self, variants_w_scores, **kwargs):
    #     pass

    @abstractmethod
    def _get_model_metadata(self, **kwargs):
        pass

    # def _get_model_metadata(
    #         self, model_metadata: Dict[str, object] = None) -> dict:
    #
    #     ml_meta = model_metadata
    #
    #     if not ml_meta:
    #         assert hasattr(self.model, 'user_defined_metadata')
    #         model_metadata = self.model.user_defined_metadata
    #         assert self.mlmodel_metadata_key in model_metadata.keys()
    #         ml_meta = model_metadata[self.mlmodel_metadata_key]
    #
    #     ret_ml_meta = \
    #         json.loads(ml_meta) if isinstance(ml_meta, str) else ml_meta
    #
    #     return ret_ml_meta

    def _get_features_count(self) -> int:
        table = self.model_metadata.get(self.lookup_table_key, None)
        return len(table[1])

    def _get_feature_encoder(self) -> FeatureEncoder:

        # metadata_getter = getattr(self, '_get_model_metadata')
        #
        # usd_model_metadata = metadata_getter()
        #
        lookup_table = self.model_metadata.get(self.lookup_table_key, None)
        model_seed = self.model_metadata.get(self.seed_key, None)
        if not lookup_table or not model_seed:
            raise ValueError(
                'Lookup table or model seed not present in context!')
        # print('model_seed')
        # print(model_seed)
        return FeatureEncoder(table=lookup_table, model_seed=model_seed)

    def _get_encoded_features(
            self, encoded_dict: Dict[str, object],
            feature_encoder: FeatureEncoder = None,
            model_metadata: Dict[str, object] = None,
            lookup_table_key: str = "table",
            seed_key: str = "model_seed") -> Dict[str, float]:
        """
        Encodes features using existing FeatureEncoder class

        Parameters
        ----------
        encoded_dict: dict
            dict of features to be encoded
        context: dict
            dict with lookup table and seed
        lookup_table_key: str
            table key in context dict
        seed_key: str
            seed key in context dict

        Returns
        -------
        dict
            encoded features dict

        """

        # TODO this should be deleted after mlmodel chooser has been
        #  refactored

        return self.feature_encoder.encode_features(encoded_dict)

    def _get_feature_names(
            self, model_metadata: Dict[str, object],
            lookup_table_key: str = "table",
            lookup_table_features_idx: int = 1) -> List[str]:
        """
        Creates feature names from provided lookup table

        Parameters
        ----------
        context: dict
            context dict with lookup table and seed
        lookup_table_key: sre
            context lookup table key
        lookup_table_features_idx: int
            index in table storing features encoding representation (?)

        Returns
        -------
        list
            list of feature names extracted from context

        """

        metadata_getter = getattr(self, '_get_model_metadata')

        usd_model_metadata = metadata_getter(model_metadata=model_metadata)

        features_count = \
            len(usd_model_metadata[
                    lookup_table_key][lookup_table_features_idx])
        feature_names = list(
            map(lambda i: 'f{}'.format(i), range(0, features_count)))

        return feature_names

    def sort(
            self, variants_w_scores: np.ndarray,
            scores_col_idx: int = 1, class_cols_idx: int = 2) -> np.ndarray:
        """
        Performs sorting of provided variants with scores array

        Parameters
        ----------
        variants_w_scores: np.ndarray
            array with variant, scores rows
        scores_col_idx: int
            the index of column with scores

        Returns
        -------
        np.ndarray
            2D sorted array of rows (variant, score)

        """

        desc_scores_sorting_col = -1 * variants_w_scores[:, scores_col_idx]
        class_sorting_col = variants_w_scores[:, class_cols_idx]

        ind = np.lexsort((desc_scores_sorting_col, class_sorting_col))

        srtd_variants_w_scores = \
            variants_w_scores[ind]

        return srtd_variants_w_scores

    def choose(
            self, variants_w_scores: np.ndarray,
            scores_col_idx: int = 1, class_col_idx: int = 2) -> np.ndarray:
        """
        Chooses the variant with the highest score. Randomly breaks ties

        Parameters
        ----------
        variants_w_scores: np.ndarray
            2D array of (variant, score) rows
        scores_col_idx: int
            int indicating column with scores

        Returns
        -------
        np.ndarray
            1D array with <best variant, best_score>

        """
        # must break ties

        choices = []

        for curr_class_id in np.unique(variants_w_scores[:, class_col_idx]):
            curr_class_variants_w_scores = \
                variants_w_scores[
                    variants_w_scores[:, class_col_idx] == curr_class_id, :]

            scores = curr_class_variants_w_scores[:, scores_col_idx]

            top_variants_w_scores = \
                curr_class_variants_w_scores[scores == scores.max()]

            chosen_top_variant_idx = \
                np.random.randint(top_variants_w_scores.shape[0])

            choices.append(top_variants_w_scores[chosen_top_variant_idx])

        return np.array(choices)
