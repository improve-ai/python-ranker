from collections.abc import Iterable
from copy import deepcopy
import json
import numpy as np
from pathlib import Path
import pickle
import re
from traceback import print_exc
from xgboost import Booster, DMatrix
from xgboost.core import XGBoostError

from improveai.feature_encoder import FeatureEncoder
from improveai.settings import CYTHON_BACKEND_AVAILABLE
from improveai.utils.choosers_feature_encoding_tools import encoded_variants_to_np
from improveai.utils.gzip_tools import check_and_get_unzipped_model
from improveai.utils.url_tools import is_path_http_addr, get_model_bytes_from_url


if CYTHON_BACKEND_AVAILABLE:
    from improveai.cythonized_feature_encoding import cfe, cfeu
    FastFeatureEncoder = cfe.FeatureEncoder
    fast_encoded_variants_to_np = cfeu.encoded_variants_to_np
    fast_encode_variants_single_givens = cfeu.encode_variants_single_givens
else:
    FastFeatureEncoder = FeatureEncoder


class XGBChooser:
    MODEL_NAME_REGEXP = "^[a-zA-Z0-9][\w\-.]{0,63}$"
    """
    Model name regexp used to verify all model names (both user provided and cached in boosters)
    """

    @property
    def model(self) -> Booster:
        """
        xgboost's booster used by this chooser

        Returns
        -------
        Booster
            xgboost's booster used by this chooser

        """
        return self._model

    @model.setter
    def model(self, new_val: Booster):
        self._model = new_val

    @property
    def model_metadata(self) -> dict:
        """
        Improve AI model metadata dict

        Returns
        -------
        dict
            Improve AI model metadata dict

        """
        return self._model_metadata

    @model_metadata.setter
    def model_metadata(self, new_val: dict):
        self._model_metadata = new_val

    @property
    def feature_encoder(self) -> FeatureEncoder or FastFeatureEncoder:
        """
        FeatureEncoder of this chooser

        Returns
        -------
        FeatureEncoder
            FeatureEncoder of this chooser

        """
        return self._feature_encoder

    @feature_encoder.setter
    def feature_encoder(self, new_val: FeatureEncoder or FastFeatureEncoder):
        self._feature_encoder = new_val

    @property
    def model_metadata_key(self):
        """
        'User defined' metadata is stored inside Improve AI Booster. Inside a
        'user defined' metadata 'model metadata' is stored under `model_metadata_key`

        Returns
        -------
        str
            `model_metadata` key inside `user_defined_metadata`

        """
        return self._mlmodel_metadata_key

    @model_metadata_key.setter
    def model_metadata_key(self, new_val: str):
        self._mlmodel_metadata_key = new_val

    @property
    def model_seed_key(self) -> str:
        """
        `model_seed` key  in `model_metadata` dict

        Returns
        -------
        str
            `model_seed` key  in `model_metadata` dict

        """
        return self._model_seed_key

    @model_seed_key.setter
    def model_seed_key(self, new_val: str):
        self._model_seed_key = new_val

    @property
    def model_seed(self):
        """
        Model seed needed for FeatureEncoder constructor

        Returns
        -------
        int
            Model seed needed for FeatureEncoder constructor

        """
        return self._model_seed

    @model_seed.setter
    def model_seed(self, value):
        self._model_seed = value

    @property
    def model_name_key(self):
        """
        `model_name` key  in `model_metadata` dict

        Returns
        -------
        str
            `model_name` key  in `model_metadata` dict

        """
        return self._model_name_key

    @model_name_key.setter
    def model_name_key(self, value):
        self._model_name_key = value

    @property
    def model_name(self):
        """
        Model name of this Improve AI model

        Returns
        -------
        str
            Model name of this Improve AI model

        """
        return self._model_name

    @model_name.setter
    def model_name(self, value):
        assert value is not None
        assert isinstance(value, str)
        assert re.search(XGBChooser.MODEL_NAME_REGEXP, value) is not None

        self._model_name = value

    @property
    def model_feature_names_key(self) -> str:
        """
        `model_feature_names` key  in `model_metadata` dict

        Returns
        -------
        str
            `model_feature_names` key  in `model_metadata` dict

        """
        return self._model_feature_names_key

    @model_feature_names_key.setter
    def model_feature_names_key(self, new_val: str):
        self._model_feature_names_key = new_val

    @property
    def model_feature_names(self) -> list:
        """
        Feature names of this Improve AI model

        Returns
        -------
        list
            Feature names of this Improve AI model

        """
        return self._model_feature_names

    @model_feature_names.setter
    def model_feature_names(self, new_val: list):
        self._model_feature_names = new_val

    @property
    def current_noise(self):
        """
        Currently used noise value. Needed for SDK + synthetic model validation

        Returns
        -------
        float
            Currently used noise

        """
        return self._current_noise

    @current_noise.setter
    def current_noise(self, value):
        self._current_noise = value

    @property
    def imposed_noise(self):
        """
        Forced noise value. Needed for SDK + synthetic model validation

        Returns
        -------
        float
            Forced noise value

        """
        return self._imposed_noise

    @imposed_noise.setter
    def imposed_noise(self, value):
        # assert noise is valid
        assert not isinstance(value, bool)
        assert isinstance(value, float) or isinstance(value, int)
        assert 0 <= value <= 1
        self._imposed_noise = value

    @property
    def improveai_major_version_from_metadata(self) -> str or None:
        """
        Stores the Improve AI model version

        Returns
        -------
        str or None
            string version of the Improve AI model or None
        """
        return self._improveai_model_version

    @improveai_major_version_from_metadata.setter
    def improveai_major_version_from_metadata(self, value):
        self._improveai_model_version = value

    @property
    def MODEL_METADATA_KEY(self):
        """
        Key in booster.attr('user_defined_metadata') storing model metadata

        Returns
        -------
        str
            json string

        """
        return 'json'

    @property
    def MODEL_FEATURE_NAMES_KEY(self):
        """
        Key in model metadata storing feature names

        Returns
        -------
        str
            'feature_names'

        """
        return 'feature_names'

    @property
    def MODEL_SEED_KEY(self):
        """
        Key in model metadata storing model seed

        Returns
        -------
        str
            'model_seed'

        """
        return 'model_seed'

    @property
    def MODEL_NAME_KEY(self):
        """
        Key in model metadata storing model name

        Returns
        -------
        str
            'model_name'

        """
        return 'model_name'

    @property
    def IMPROVE_AI_ALLOWED_MAJOR_VERSION(self):
        """
        Latest supported major model version

        Returns
        -------
        int
            7

        """
        return 7

    @property
    def IMPROVEAI_VERSION_KEY(self):
        """
        model metadata key storing model version

        Returns
        -------
        str
            'ai.improve.version'

        """
        return 'ai.improve.version'

    def __init__(self):
        """
        Initialize chooser object
        """

        self.model = None
        self.model_metadata = None

        self.feature_encoder = None
        self.model_feature_names = np.empty(shape=(1,))

        self.model_seed = None
        self._model_name = None
        self.current_noise = None
        self._imposed_noise = None
        self._improveai_major_version_from_metadata = None

    def load_model(self, input_model_src: str, verbose: bool = False):
        """
        Loads desired model from input path.

        Parameters
        ----------
        input_model_src: str
            URL / path to desired model
        verbose: bool
            should I print debug messages
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

            model_src = \
                input_model_src if isinstance(input_model_src, str) or isinstance(input_model_src, Path) \
                else bytearray(input_model_src)

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

        model_metadata = self._get_model_metadata()
        self.model_seed = self._get_model_seed(model_metadata=model_metadata)
        self.model_name = self._get_model_name(model_metadata=model_metadata)
        self.model_feature_names = \
            self._get_model_feature_names(model_metadata=model_metadata)
        self.improveai_major_version_from_metadata = \
            self._get_improveai_major_version(model_metadata=model_metadata)

        if CYTHON_BACKEND_AVAILABLE:
            self.feature_encoder = FastFeatureEncoder(model_seed=self.model_seed)
        else:
            self.feature_encoder = FeatureEncoder(model_seed=self.model_seed)

    def _get_improveai_major_version(self, model_metadata: dict) -> str or None:
        """
        Extract Improve AI version from model metadata and return it if it is valid / allowed

        Parameters
        ----------
        model_metadata: dict
            a dictionary containing model metadata

        Returns
        -------
        str or None
            major Improve AI version extracted from loaded improve model
        """
        improveai_major_version = None
        if self.IMPROVEAI_VERSION_KEY in model_metadata.keys():
            improveai_version = model_metadata[self.IMPROVEAI_VERSION_KEY]

            print('### improveai_version ###')
            print(improveai_version)

            assert improveai_version is not None and isinstance(improveai_version, str)
            # major version is the first chunk of version string
            improveai_major_version = int(improveai_version.split('.')[0])
            assert improveai_major_version == self.IMPROVE_AI_ALLOWED_MAJOR_VERSION
        return improveai_major_version

    def _get_model_metadata(self) -> dict:
        """
        gets 'model metadata' from 'user defined metadata' of Improve AI model

        Returns
        -------
        dict
            dict with model metadata
        """

        assert 'user_defined_metadata' in self.model.attributes().keys()
        user_defined_metadata_str = self.model.attr('user_defined_metadata')
        user_defined_metadata = json.loads(user_defined_metadata_str)
        assert self.MODEL_METADATA_KEY in user_defined_metadata.keys()

        return user_defined_metadata[self.MODEL_METADATA_KEY]

    def score(self, variants: list or tuple or np.ndarray, givens: dict or None, **kwargs) -> np.ndarray:
        """
        Scores all provided variants

        Parameters
        ----------
        variants: list or tuple or np.ndarray
            list of variants to scores
        givens: dict or None
            context dict needed for encoding
        kwargs: dict
            kwargs

        Returns
        -------
        np.ndarray
            2D numpy array which contains (variant, score) pair in each row
        """

        encoded_variants = \
            self.encode_variants_single_givens(variants=variants, givens=givens)

        encoded_variants_to_np_method = \
            fast_encoded_variants_to_np if CYTHON_BACKEND_AVAILABLE \
            else encoded_variants_to_np

        missings_filled_v = \
            encoded_variants_to_np_method(
                encoded_variants=encoded_variants,
                feature_names=self.model_feature_names)

        if CYTHON_BACKEND_AVAILABLE:
            missings_filled_v = np.asarray(missings_filled_v)

        scores = \
            self.model.predict(
                DMatrix(
                    missings_filled_v, feature_names=self.model_feature_names))\
            .astype('float64')

        return scores

    def fill_missing_features(self, encoded_variants):
        """
        Fills missing features in encoded variants and packs them into 2D numpy array

        Parameters
        ----------
        encoded_variants: list
            a list of encoded variants (dicts)

        Returns
        -------
        np.ndarray
            2D numpy array with all features for xgb model

        """
        encoded_variants_to_np_method = \
            fast_encoded_variants_to_np if CYTHON_BACKEND_AVAILABLE else encoded_variants_to_np

        features_matrix = encoded_variants_to_np_method(
            encoded_variants=encoded_variants, feature_names=self.model_feature_names)

        if CYTHON_BACKEND_AVAILABLE:
            features_matrix = np.asarray(features_matrix)

        return features_matrix

    def calculate_predictions(self, features_matrix: np.ndarray):
        """
        Calculates predictions on provided matrix with loaded model

        Parameters
        ----------
        features_matrix: np.ndarray
            array to be a source for DMatrix

        Returns
        -------

        """
        # make sure input is a numpy array
        assert isinstance(features_matrix, np.ndarray)
        # make sure input for predictions is not empty
        assert features_matrix.size > 0
        # make sure it is 2D array
        assert len(features_matrix.shape) == 2
        # make sure all features are present
        assert len(self.model_feature_names) == features_matrix.shape[1]
        scores = \
            self.model.predict(
                DMatrix(features_matrix, feature_names=self.model_feature_names)) \
            .astype('float64')
        return scores

    def encode_variants_single_givens(
            self, variants: list or tuple or np.ndarray, givens: dict or None) -> Iterable:
        """
        Implemented as a XGBChooser helper method
        Cythonized loop over provided variants and a single givens dict.
        Returns array of encoded dicts.

        Parameters
        ----------
        variants: list or tuple or np.ndarray
            collection of input variants to be encoded
        givens: dict or None
            context to be encoded with variants

        Returns
        -------
        np.ndarray
            array of encoded dicts
        """

        if not (isinstance(givens, dict) or givens is None or givens is {}):
            raise TypeError(
                'Unsupported givens` type: {}'.format(type(givens)))

        if self.imposed_noise is None:
            noise = np.random.rand()
        else:
            noise = self.imposed_noise

        if CYTHON_BACKEND_AVAILABLE:
            if isinstance(variants, list):
                used_variants = variants
            elif isinstance(variants, tuple) or isinstance(variants, np.ndarray):
                used_variants = list(variants)
            else:
                raise TypeError(
                    'Variants are of a wrong type: {}'.format(type(variants)))

            return fast_encode_variants_single_givens(
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
        if not isinstance(model_src, Path) and is_path_http_addr(pth_to_model=model_src):
            raw_model_src = get_model_bytes_from_url(model_url=model_src)

        unzipped_model_src = check_and_get_unzipped_model(model_src=raw_model_src)
        return unzipped_model_src

    def _get_model_feature_names(self, model_metadata: dict) -> list:
        """
        Gets model feature names from model metadata

        Parameters
        ----------
        model_metadata: dict
            a dict containing model metadata

        Returns
        -------
        list
            list of feature names

        """

        if not model_metadata:
            raise ValueError('Model metadata empty or None!')

        feature_names = model_metadata.get(self.MODEL_FEATURE_NAMES_KEY, None)

        if not feature_names:
            raise ValueError('Feature names not in model metadata!')

        return feature_names

    def _get_model_seed(self, model_metadata: dict) -> int:
        """
        Gets model seed from model metadata

        Parameters
        ----------
        model_metadata: dict
            a dict containing model metadata


        Returns
        -------
        int
            model seed

        """
        if not model_metadata:
            raise ValueError('Model metadata empty or None!')

        model_seed = model_metadata.get(self.MODEL_SEED_KEY, None)

        if not model_seed:
            raise ValueError('Feature names not in model metadata!')

        return model_seed

    def _get_model_name(self, model_metadata: dict) -> str:
        """
        Gets model name from model metadata

        Parameters
        ----------
        model_metadata: dict
            a dict containing model metadata

        Returns
        -------
        str
            Improve AI model name

        """
        if not model_metadata:
            raise ValueError('Model metadata empty or None!')

        model_name = model_metadata.get(self.MODEL_NAME_KEY, None)

        if not model_name:
            raise ValueError('Feature names not in model metadata!')

        return model_name

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
