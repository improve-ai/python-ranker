from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
from typing import Dict

# from feature_encoders.v5 import FeatureEncoder
from feature_encoders.v6 import FeatureEncoder
from utils.url_utils import is_path_http_addr, get_model_bytes_from_url
from utils.gz_utils import check_and_get_unzpd_model


class BasicChooser(ABC):
    @property
    @abstractmethod
    def model(self) -> object:
        pass

    @model.setter
    @abstractmethod
    def model(self, new_val: object):
        pass

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

    # @property
    # @abstractmethod
    # def lookup_table_key(self) -> str:
    #     pass
    #
    # @lookup_table_key.setter
    # @abstractmethod
    # def lookup_table_key(self, new_val: str):
    #     pass

    @property
    @abstractmethod
    def model_seed_key(self) -> str:
        pass

    @model_seed_key.setter
    @abstractmethod
    def model_seed_key(self, new_val: str):
        pass
    
    @property
    @abstractmethod
    def model_seed(self):
        pass 
    
    @model_seed.setter
    @abstractmethod
    def model_seed(self, value):
        pass
    
    @property
    @abstractmethod
    def model_name_key(self):
        pass
    
    @model_name_key.setter
    def model_name_key(self, value):
        pass

    @property
    @abstractmethod
    def model_name(self):
        pass

    @model_name.setter
    def model_name(self, value):
        pass

    @property
    @abstractmethod
    def model_feature_names_key(self):
        return

    @model_feature_names_key.setter
    @abstractmethod
    def model_feature_names_key(self, value):
        pass

    @property
    @abstractmethod
    def model_feature_names(self):
        return

    @model_feature_names.setter
    @abstractmethod
    def model_feature_names(self, value):
        pass

    @abstractmethod
    def load_model(self, inupt_model_src, **kwargs):
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

    @staticmethod
    def get_model_src(model_src: str or bytes) -> str or bytes:
        """
        Gets model src from provided input path, url or bytes
        
        Parameters
        ----------
        model_src: str or bytes
            pth to model, url or bytes

        Returns
        -------
        str or bytes
            path or downloaded model

        """
        raw_model_src = model_src
        if is_path_http_addr(pth_to_model=model_src):
            raw_model_src = get_model_bytes_from_url(model_url=model_src)

        unzpd_model_src = check_and_get_unzpd_model(model_src=raw_model_src)
        return unzpd_model_src

    def _get_features_count(self) -> int:
        table = self.model_metadata.get(self.lookup_table_key, None)
        return len(table[1])

    def _get_feature_encoder(self) -> FeatureEncoder:
        """
        Getter for FeatureEncoder object

        Returns
        -------
        FeatureEncoder
            feature feature_encoder for current model

        """

        # lookup_table = self.model_metadata.get(self.lookup_table_key, None)
        model_seed = self.model_metadata.get(self.model_seed_key, None)
        # if not lookup_table or not model_seed:
        #     raise ValueError(
        #         'Lookup table or model seed not present in context!')
        # return FeatureEncoder(table=lookup_table, model_seed=model_seed)
        return FeatureEncoder(model_seed=model_seed)

    def _get_model_feature_names(self):
        """
        Getter for model features

        Returns
        -------
        dict
            dict with feature names extracted from model metadata

        """

        if not self.model_metadata:
            raise ValueError('Model metadata empty or None!')

        feature_names = \
            self.model_metadata.get(self.model_feature_names_key, None)

        if not feature_names:
            raise ValueError('Feature names not in model metadata!')
        
        return np.array(feature_names)

    def _get_model_seed(self):
        if not self.model_metadata:
            raise ValueError('Model metadata empty or None!')

        model_seed = \
            self.model_metadata.get(self.model_seed_key, None)

        if not model_seed:
            raise ValueError('Feature names not in model metadata!')

        return model_seed

    def _get_model_name(self):
        if not self.model_metadata:
            raise ValueError('Model metadata empty or None!')

        model_name = \
            self.model_metadata.get(self.model_name_key, None)

        if not model_name:
            raise ValueError('Feature names not in model metadata!')

        return model_name

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
            desired number of feature_names in the model
        missing_filler: float
            value ot impute missings with

        Returns
        -------
        np.ndarray
            np.ndarray of np.ndarrays of which each containes all feature_names with
            missings filled

        """

        # st1 = time()
        context_copy = deepcopy(context)
        enc_variant = self.feature_encoder.encode_features({'variant': variant})
        context_copy.update(enc_variant)
        # et1 = time()
        # print('Encoding feature_names took: {}'.format(et1 - st1))
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
        class_col_idx: int
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
