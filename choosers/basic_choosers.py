from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
from time import time
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
    def score_all(self, variants, context, **kwargs):
        pass

    @abstractmethod
    def _get_model_metadata(self, **kwargs):
        pass

    def _get_features_count(self) -> int:
        table = self.model_metadata.get(self.lookup_table_key, None)
        return len(table[1])

    def _get_feature_encoder(self) -> FeatureEncoder:
        """
        Getter for FeatureEncoder object

        Returns
        -------
        FeatureEncoder
            feature encoder for current model

        """

        lookup_table = self.model_metadata.get(self.lookup_table_key, None)
        model_seed = self.model_metadata.get(self.seed_key, None)
        if not lookup_table or not model_seed:
            raise ValueError(
                'Lookup table or model seed not present in context!')
        return FeatureEncoder(table=lookup_table, model_seed=model_seed)

    # def _get_missings_filled_variants(
    #         self, input_dict: Dict[str, object], all_feats_count: int,
    #         missing_filler: float = np.nan):
    #     """
    #     Fast wrapper around missing filling procedure
    #
    #     Parameters
    #     ----------
    #     input_dict: Dict[str, object]
    #         current scoring context
    #     all_feats_count: int
    #         definite number of features
    #
    #     Returns
    #     -------
    #     np.ndarray
    #         array with values of missing features filled
    #
    #     """
    #     return np.array([
    #         input_dict[el] if input_dict.get(el, None) is not None
    #         else missing_filler for el in np.arange(0, all_feats_count, 1)])  # \
    #         # .reshape(1, all_feats_count)

    def _get_nan_filled_encoded_variant(
            self, variant: Dict[str, object], context: Dict[str, object],
            all_feats_count: int, missing_filler: float = np.nan) -> np.ndarray:
        """
        Wrapper around fast filling missings in variant. This would be used in
        case many variants need to be scored at once and predict() accepts numpy
        array

        Parameters
        ----------
        variant: Dict[str, object]
            single observation
        context: Dict[str, object]
            scoring context
        all_feats_count: int
            desired number of features in the model
        missing_filler: float
            value ot impute missings with

        Returns
        -------
        np.ndarray
            np.ndarray of np.ndarrays of which each containes all features with
            missings filled

        """

        # st1 = time()
        context_copy = deepcopy(context)
        enc_variant = self.feature_encoder.encode_features({'variant': variant})
        context_copy.update(enc_variant)
        # et1 = time()
        # print('Encoding features took: {}'.format(et1 - st1))
        # print(context_copy)

        # st2 = time()
        missings_filled_v = np.empty((1, all_feats_count))
        # et2 = time()
        # print('Initializing empty took: {}'.format(et2 - st2))

        # st = time()
        missings_filled_v[0, :] = missing_filler
        missings_filled_v[0, list(context_copy.keys())] = \
            list(context_copy.values())
        # et = time()
        # print('Filling nans took: {}'.format(et - st))
        # input('sanity check')

        # missings_filled_v = \
        #     self._get_missings_filled_variants(
        #         input_dict=context_copy, all_feats_count=all_feats_count,
        #         missing_filler=missing_filler)

        return missings_filled_v

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
        class_cols_idx: int
            index of the class label in a single row


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
        class_cols_idx: int
            index of the class label in a single row

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
