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
from improveai.utils.gzip_tools import check_and_get_unzipped_model
from improveai.utils.url_tools import is_path_http_addr, get_model_bytes_from_url


if CYTHON_BACKEND_AVAILABLE:
    from improveai.cythonized_feature_encoding import cfe
    FastFeatureEncoder = cfe.FeatureEncoder
    fast_encode_variants_to_matrix = cfe.encode_variants_to_matrix
else:
    FastFeatureEncoder = FeatureEncoder


USER_DEFINED_METADATA_KEY = 'user_defined_metadata'
FEATURE_NAMES_METADATA_KEY = 'ai.improve.features'


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
    def feature_names(self) -> list:
        """
        Feature names of this Improve AI model

        Returns
        -------
        list
            Feature names of this Improve AI model

        """
        return self._model_feature_names

    @feature_names.setter
    def feature_names(self, new_val: list):
        self._model_feature_names = new_val

    @property
    def string_tables(self):
        return self._string_tables

    @string_tables.setter
    def string_tables(self, value):
        # make sure that value is a dict
        assert isinstance(value, dict)
        # make sure that this dict contains all list as values
        assert all(isinstance(string_table, list) for string_table in value.values())
        self._string_tables = value

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
    def FEATURE_NAMES_METADATA_KEY(self):
        """
        Key in model metadata storing feature names

        Returns
        -------
        str
            'ai.improve.features'

        """
        return FEATURE_NAMES_METADATA_KEY

    @property
    def STRING_TABLES_METADATA_KEY(self):
        """
        Key in model metadata storing string tables

        Returns
        -------
        str
            'ai.improve.string_tables'

        """
        return 'ai.improve.string_tables'

    @property
    def MODEL_SEED_METADATA_KEY(self):
        """
        Key in model metadata storing model seed

        Returns
        -------
        str
            'ai.improve.seed'

        """
        return 'ai.improve.seed'

    @property
    def MODEL_NAME_METADATA_KEY(self):
        """
        Key in model metadata storing model name

        Returns
        -------
        str
            'ai.improve.model'

        """
        return 'ai.improve.model'

    @property
    def CREATED_AT_METADATA_KEY(self):
        """
        Key in model metadata storing model creation time

        Returns
        -------
        str
            'ai.improve.created_at'

        """
        return 'ai.improve.created_at'

    @property
    def IMPROVE_AI_ALLOWED_MAJOR_VERSION(self):
        """
        Latest supported major model version

        Returns
        -------
        int
            8

        """
        return 8

    @property
    def VERSION_METADATA_KEY(self):
        """
        model metadata key storing model version

        Returns
        -------
        str
            'ai.improve.version'

        """
        return 'ai.improve.version'

    @property
    def USER_DEFINED_METADATA_KEY(self):
        """
        booster attribute name storing an entire user defined metadata dict

        Returns
        -------
        str
            'user_defined_metadata'

        """
        return USER_DEFINED_METADATA_KEY

    @property
    def REQUIRED_METADATA_KEYS(self):
        """
        keys expected / required in model metadata

        Returns
        -------
        str
            list of required keys present in model metadata

        """

        return [
            self.MODEL_NAME_METADATA_KEY, self.FEATURE_NAMES_METADATA_KEY,
            self.STRING_TABLES_METADATA_KEY, self.MODEL_SEED_METADATA_KEY,
            self.CREATED_AT_METADATA_KEY, self.VERSION_METADATA_KEY]

    def __init__(self):
        """
        Initialize chooser object
        """

        self.model = None
        self.model_metadata = None

        self.feature_encoder = None
        self.feature_names = np.empty(shape=(1,))

        self.model_seed = None
        self._model_name = None
        self._string_tables = None
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
        self.feature_names = self._get_model_feature_names(model_metadata=model_metadata)
        self.string_tables = self._get_string_tables(model_metadata=model_metadata)
        self.improveai_major_version_from_metadata = \
            self._get_improveai_major_version(model_metadata=model_metadata)

        if CYTHON_BACKEND_AVAILABLE:
            self.feature_encoder = FastFeatureEncoder(
                feature_names=self.feature_names, string_tables=self.string_tables, model_seed=self.model_seed)
        else:
            self.feature_encoder = FeatureEncoder(
                feature_names=self.feature_names, string_tables=self.string_tables, model_seed=self.model_seed)

    def _get_noise(self):
        if self.imposed_noise is None:
            noise = np.random.rand()
        else:
            noise = self.imposed_noise

        return noise

    def encode_variants_to_matrix(
            self, variants: list or tuple or np.ndarray, givens: dict or None, noise: float = 0.0):
        into_matrix = np.full((len(variants), len(self.feature_encoder.feature_indexes)), np.nan)

        for variant, into_row in zip(variants, into_matrix):
            # variant: object, givens: object, into: np.ndarray, noise: float
            self.feature_encoder.encode_feature_vector(
                variant=variant, givens=givens, into=into_row, noise=noise)
        return into_matrix

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

        encoded_variants_matrix = \
            self.encode_variants_single_givens(variants=variants, givens=givens)

        scores = \
            self.model.predict(
                DMatrix(
                    encoded_variants_matrix, feature_names=self.feature_names))\
            .astype('float64')

        return scores

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
        assert len(self.feature_names) == features_matrix.shape[1]
        scores = \
            self.model.predict(
                DMatrix(features_matrix, feature_names=self.feature_names)) \
            .astype('float64')
        return scores

    def encode_variants_single_givens(
            self, variants: list or tuple or np.ndarray, givens: dict or None) -> np.ndarray:
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
            raise TypeError('Unsupported givens` type: {}'.format(type(givens)))

        if CYTHON_BACKEND_AVAILABLE:
            return \
                np.asarray(fast_encode_variants_to_matrix(
                    variants=variants, givens=givens, feature_encoder=self.feature_encoder, noise=self._get_noise()))
        else:
            return self.encode_variants_to_matrix(variants=variants, givens=givens, noise=self._get_noise())

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
        if self.VERSION_METADATA_KEY in model_metadata.keys():
            improveai_version = model_metadata[self.VERSION_METADATA_KEY]

            if improveai_version is None or not isinstance(improveai_version, str):
                raise IOError(f'Improve AI version stored in metadata ({improveai_version}) is either None or not a string')
            # major version is the first chunk of version string
            improveai_major_version = int(improveai_version.split('.')[0])

            if improveai_major_version != self.IMPROVE_AI_ALLOWED_MAJOR_VERSION:
                raise IOError(f'Attempting to load model from unsupported Improve AI version: {improveai_major_version}.'
                              f' Currently supported Improve AI major version is: {self.IMPROVE_AI_ALLOWED_MAJOR_VERSION}')
        return improveai_major_version

    def _get_model_metadata(self) -> dict:
        """
        gets 'model metadata' from 'user defined metadata' of Improve AI model

        Returns
        -------
        dict
            dict with model metadata
        """

        if self.USER_DEFINED_METADATA_KEY not in self.model.attributes().keys():
            raise IOError(f'Improve AI booster has no: {self.USER_DEFINED_METADATA_KEY} attribute')

        user_defined_metadata_str = self.model.attr(self.USER_DEFINED_METADATA_KEY)
        user_defined_metadata = json.loads(user_defined_metadata_str)

        if not user_defined_metadata:
            raise IOError('Model metadata is either None or empty')

        loaded_metadata_keys = set(user_defined_metadata.keys())

        for required_key in self.REQUIRED_METADATA_KEYS:
            if required_key not in loaded_metadata_keys:
                raise IOError(f'Improve AI booster`s metadata has no: {required_key} key')

        return user_defined_metadata

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

        feature_names = model_metadata.get(self.FEATURE_NAMES_METADATA_KEY, None)

        if not feature_names:
            raise IOError('Feature names can`t be None or empty collection')

        if not all(isinstance(fn, str) for fn in feature_names):
            raise IOError('All feature names must be strings')

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
        model_seed = model_metadata.get(self.MODEL_SEED_METADATA_KEY, None)

        if not model_seed or not isinstance(model_seed, int):
            raise IOError(f'Wrong {MODEL_SEED_METADATA_KEY}: {model_seed} (type: {type(model_seed)}).')

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
        model_name = model_metadata.get(self.MODEL_NAME_METADATA_KEY, None)

        if not model_name:
            raise IOError('Model name empty or None!')

        if not model_name or not isinstance(model_name, str):
            raise IOError(f'Wrong {MODEL_NAME_METADATA_KEY}: {model_name} (type: {type(model_name)}).')

        return model_name

    def _get_string_tables(self, model_metadata: dict):
        string_tables = model_metadata.get(self.STRING_TABLES_METADATA_KEY, None)
        if not string_tables or not isinstance(string_tables, dict):
            raise IOError('String tables can`t empty or None!')

        # make sure that all values of `string_tables` are
        if not all(isinstance(string_table, list) for string_table in string_tables.values()):
            raise IOError('At least one of string tables is not a list')
        return string_tables

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
