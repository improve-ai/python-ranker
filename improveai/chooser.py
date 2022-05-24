from collections.abc import Iterable
from copy import deepcopy
import json
import numpy as np
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

    def __init__(
            self, model_metadata_key: str = 'json',
            model_feature_names_key: str = 'feature_names',
            model_seed_key: str = 'model_seed',
            model_name_key: str = 'model_name'):
        """
        Init with params

        Parameters
        ----------
        model_metadata_key: str
            key storing 'model metadata' inside 'user defined metadata'
        model_feature_names_key: str
            key storing 'feature names' inside 'model metadata'
        model_seed_key: str
            key storing 'seed' inside 'model metadata'
        model_name_key: str
            key storing 'model name' inside 'model metadata'
        """

        self.model = None
        self.model_metadata_key = model_metadata_key
        self.model_metadata = None

        self.feature_encoder = None
        self.model_feature_names_key = model_feature_names_key
        self.model_feature_names = np.empty(shape=(1,))

        self.model_seed_key = model_seed_key
        self.model_seed = None

        self.model_name_key = model_name_key
        self._model_name = None

        self.current_noise = None
        self._imposed_noise = None

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

            raw_model_src = self.get_model_src(model_src=input_model_src)

            model_src = \
                raw_model_src if isinstance(raw_model_src, str) \
                else bytearray(raw_model_src)

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

        if CYTHON_BACKEND_AVAILABLE:
            self.feature_encoder = FastFeatureEncoder(model_seed=self.model_seed)
        else:
            self.feature_encoder = FeatureEncoder(model_seed=self.model_seed)

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
        assert self.model_metadata_key in user_defined_metadata.keys()

        return user_defined_metadata[self.model_metadata_key]

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
            self._encode_variants_single_givens(variants=variants, givens=givens)

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

    def _encode_variants_single_givens(
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
        if is_path_http_addr(pth_to_model=model_src):
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

        feature_names = model_metadata.get(self.model_feature_names_key, None)

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

        model_seed = model_metadata.get(self.model_seed_key, None)

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

        model_name = model_metadata.get(self.model_name_key, None)

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
