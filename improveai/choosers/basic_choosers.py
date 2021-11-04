from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
from typing import Dict

from improveai.feature_encoder import FeatureEncoder
from improveai.utils.url_tools import is_path_http_addr, \
    get_model_bytes_from_url
from improveai.utils.gzip_tools import check_and_get_unzpd_model


class BasicChooser(ABC):
    MODEL_NAME_REGEXP = "^[a-zA-Z0-9][\w\-.]{0,63}$"

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
    def score(self, variants, givens, **kwargs):
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

    def _get_model_feature_names(self, model_metadata: dict):
        """
        Getter for model features

        Returns
        -------
        dict
            dict with feature names extracted from model metadata

        """

        if not model_metadata:
            raise ValueError('Model metadata empty or None!')

        feature_names = model_metadata.get(self.model_feature_names_key, None)

        if not feature_names:
            raise ValueError('Feature names not in model metadata!')
        
        return feature_names

    def _get_model_seed(self, model_metadata: dict):
        if not model_metadata:
            raise ValueError('Model metadata empty or None!')

        model_seed = model_metadata.get(self.model_seed_key, None)

        if not model_seed:
            raise ValueError('Feature names not in model metadata!')

        return model_seed

    def _get_model_name(self, model_metadata: dict):
        if not model_metadata:
            raise ValueError('Model metadata empty or None!')

        model_name = model_metadata.get(self.model_name_key, None)

        if not model_name:
            raise ValueError('Feature names not in model metadata!')

        return model_name

    # TODO sort and choose are deprecated (kept only for CLI compatibility)
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
