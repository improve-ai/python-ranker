import numpy as np
import xxhash


ITEM_FEATURE_KEY = 'item'
"""
Feature names prefix for features derived from candidates / items, e.g.:

- item == 1 -> feature name is "item"

- item == [1] -> feature names is "item.0"

- item == {"a": 1}} - feature name is "item.a"
"""

CONTEXT_FEATURE_KEY = 'context'
"""
Feature names prefix for features derived from context, e.g.:

- context == 1 -> feature name is "context"

- context == [1] -> feature names is "context.0"

- context == {"a": 1}} - feature name is "context.a"
"""

FIRST_LEVEL_FEATURES_CHUNKS = {ITEM_FEATURE_KEY, CONTEXT_FEATURE_KEY}

xxh3 = xxhash.xxh3_64_intdigest


class FeatureEncoder:
    """
    Encodes JSON encodable objects into float vectors
    """

    @property
    def feature_indexes(self) -> dict:
        """
        A map between feature names and feature indexes. Created by simple
        iteration with enumeration over feature names

        Returns
        -------
        dict
            a mapping between a string feature names and feature index

        """
        return self._feature_indexes

    @feature_indexes.setter
    def feature_indexes(self, value: dict):
        self._feature_indexes = value

    @property
    def string_tables(self) -> list:
        """
        List of StringTable objects. The order of elements follows constructor's
        `string_tables` parameter.

        Returns
        -------
        list
            list of StringTables

        """
        return self._string_tables

    @string_tables.setter
    def string_tables(self, value: list):
        self._string_tables = value

    def __init__(self, feature_names: list, string_tables: dict, model_seed: int):
        """
        Initialize the feature encoder for this model

        Parameters
        ----------
        feature_names: list
            the list of feature names. Order matters - first feature name should
            be the first feature in the model
        string_tables: dict
            a mapping from feature names to string hash tables
        model_seed: int
            model seed to be used during string encoding

        Raises
        ----------
        ValueError if feature names or tables are corrupt
        """

        self.feature_indexes = {}

        # index feature names for vectorization
        for i, feature_name in enumerate(feature_names):
            self.feature_indexes[feature_name] = i

        # initialize string encoding tables with a shared empty table
        self.string_tables = [StringTable([], model_seed)] * len(feature_names)

        try:
            for feature_name, table in string_tables.items():
                self.string_tables[self.feature_indexes[feature_name]] = StringTable(table, model_seed)
        except KeyError as exc:
            raise ValueError("Bad model metadata") from exc

    def _check_into(self, into: np.ndarray):
        """
        Checks if the provided `into` array is an array and has desired
        np.float64 dtype

        Parameters
        ----------
        into: np.ndarray
            array which will store feature values

        Raises
        -------
        ValueError if into is not a numpy array or not of a float64 dtype

        """

        if not isinstance(into, np.ndarray) or into.dtype != np.float64:
            raise ValueError("into must be a float64 array")

    def encode_item(self, item: object, into: np.ndarray, noise_shift: float = 0.0, noise_scale: float = 1.0):
        """
        Encodes provided item to `input` numpy array

        Parameters
        ----------
        item: object
            JSON encodable python object
        into: np.ndarray
            array storing results of encoding
        noise_shift: float
            value to be added to values of features
        noise_scale: float
            multiplier used to scale shifted feature values

        Returns
        -------

        """
        self._encode(item, path=ITEM_FEATURE_KEY, into=into, noise_shift=noise_shift, noise_scale=noise_scale)

    def encode_context(self, context: object, into: np.ndarray, noise_shift: float = 0.0, noise_scale: float = 1.0):
        """
        Encodes provided context to `input` numpy array

        Parameters
        ----------
        context: object
            JSON encodable python object
        into: np.ndarray
            array storing results of encoding
        noise_shift: float
            value to be added to values of features
        noise_scale: float
            multiplier used to scale shifted feature values

        Returns
        -------

        """
        self._encode(context, path=CONTEXT_FEATURE_KEY, into=into, noise_shift=noise_shift, noise_scale=noise_scale)

    def encode_feature_vector(
            self, item: object, context: object, into: np.ndarray, noise: float = 0.0):
        """
        Fully encodes provided variant and context into a np.ndarray provided as
        `into` parameter. `into` must not be None

        Parameters
        ----------
        item: object
            a JSON encodable item to be encoded
        context: object
            a JSON encodable context to be encoded
        into: np.ndarray
            an array into which feature values will be added
        noise: float
            value in [0, 1) which will be combined with the feature value

        Returns
        -------

        """

        noise_shift, noise_scale = get_noise_shift_scale(noise)

        if item is not None:
            self.encode_item(item, into, noise_shift, noise_scale)

        if context is not None:
            self.encode_context(context, into, noise_shift, noise_scale)

        # for 10k calls this takes only 4 ms
        # if both item and context are None into will not be checked until now.
        if item is None and context is None:
            self._check_into(into=into)

    def _encode(self, obj: object, path: str, into: np.ndarray, noise_shift: float = 0.0, noise_scale: float = 1.0):
        """
        Encodes a JSON serializable object to a float vector
        Rules of encoding go as follows:

        - None, json null, {}, [], and nan are treated as missing features and ignored.

        - numbers and booleans are encoded as-is.

        - strings are encoded using a lookup table

        Parameters
        ----------
        obj: object
            a JSON serializable object to be encoded to a flat key-value structure
        path: str
            the JSON-normalized path to the current object
        into: np.ndarray
            an array into which feature values will be encoded
        noise_shift: float
            small bias added to the feature value
        noise_scale: float
            small multiplier of the feature value

        Returns
        -------
        """
        if path in FIRST_LEVEL_FEATURES_CHUNKS:
            self._check_into(into)

        # TODO do we want to check if `obj` is JSON serializable?
        if isinstance(obj, (int, float)):  # bool is an instanceof int
            # TODO - do we want to differentiate between np.nan and None in python?
            if np.isnan(obj):  # nan is treated as missing feature, return
                return

            feature_index = self.feature_indexes.get(path)
            if feature_index is None:
                return

            into[feature_index] = sprinkle(obj, noise_shift, noise_scale)

        elif isinstance(obj, str):
            feature_index = self.feature_indexes.get(path)
            if feature_index is None:
                return

            string_table = self.string_tables[feature_index]

            into[feature_index] = sprinkle(string_table.encode(obj), noise_shift, noise_scale)

        elif isinstance(obj, dict):
            for key, value in obj.items():
                self._encode(obj=value, path=path + '.' + key, into=into, noise_shift=noise_shift, noise_scale=noise_scale)

        elif isinstance(obj, (list, tuple)):
            for index, item in enumerate(obj):
                self._encode(obj=item, path=path + '.' + str(index), into=into, noise_shift=noise_shift, noise_scale=noise_scale)

        elif obj is None:
            pass

        else:
            raise ValueError(f'{obj} not JSON encodable. Must be string, int, float, bool, list, tuple, dict, or None')


def get_noise_shift_scale(noise: float) -> tuple:
    """
    Returns noise shift (small value added to feature value) and noise scale
    (value by which shifted feature value is multiplied)

    Parameters
    ----------
    noise: float
        value in [0, 1) which will be combined with the feature value


    Returns
    -------
    tuple
        tuple of floats: (noise_shift, noise_scale)

    """
    assert noise >= 0.0 and noise < 1.0
    # x + noise * 2 ** -142 will round to x for most values of x. Used to create
    # distinct values when x is 0.0 since x * scale would be zero
    return (noise * 2 ** -142, 1 + noise * 2 ** -17)


def sprinkle(x: float, noise_shift: float, noise_scale: float) -> float:
    """
    Adds noise shift and scales shifted value

    Parameters
    ----------
    x: float
        value to be sprinkled
    noise_shift: float
        small bias added to the feature value
    noise_scale: float
        small multiplier of the shifted feature value

    Returns
    -------
    float
        sprinkled value

    """
    # x + noise_offset will round to x for most values of x
    # allows different values when x == 0.0
    return (x + noise_shift) * noise_scale


class StringTable:
    """
    A class responsible for target encoding of strings for a given feature
    """

    @property
    def model_seed(self) -> int:
        """
        32-bit random integer used to hash strings with xxhash

        Returns
        -------
        int
            model seed

        """
        return self._model_seed

    @model_seed.setter
    def model_seed(self, value: int):
        self._model_seed = value

    @property
    def mask(self) -> int:
        """
        At most 64 bit int representation of a string hash mask e.g., 000..00111

        Returns
        -------
        int
            mask used to 'decrease' hashed string value

        """
        return self._mask

    @mask.setter
    def mask(self, value: int):
        self._mask = value

    @property
    def miss_width(self) -> float:
        """
        Float value representing snap / width of the 'miss interval' - numeric
        interval into which all missing / unknown values are encoded. It is
        0-centered.

        Returns
        -------
        float
            miss width value

        """
        return self._miss_width

    @miss_width.setter
    def miss_width(self, value: float):
        self._miss_width = value

    @property
    def value_table(self) -> dict:
        """
        A mapping from masked string hash to target encoding's target value for a given feature

        Returns
        -------
        dict
            a dict with target value encoding

        """
        return self._value_table

    @value_table.setter
    def value_table(self, value: dict):
        self._value_table = value

    def __init__(self, string_table, model_seed):
        """
        Init StringTable with params

        Parameters
        ----------
        string_table: list
            a list of masked hashed for each string feature
        model_seed: int
            model seed value
        """

        if model_seed < 0:
            raise ValueError(
                "xxhash3_64 takes an unsigned 64-bit integer as seed. "
                "Seed must be greater than or equal to 0.")

        self.model_seed = model_seed
        self.mask = get_mask(string_table)
        max_position = len(string_table) - 1

        # empty and single entry tables will have a miss_width of 1 or range [-0.5, 0.5]
        # 2 / max_position keeps miss values from overlapping with nonzero table values
        self.miss_width = 1 if max_position < 1 else 2 / max_position

        self.value_table = {}

        for index, string_hash in enumerate(reversed(string_table)):
            # a single entry gets a value of 1.0
            self.value_table[string_hash] = 1.0 if max_position == 0 else scale(index / max_position)

    def encode(self, string: str) -> float:
        """
        Encode input string to a target value

        Parameters
        ----------
        string: str
            string to be encoded

        Returns
        -------
        float
            encoded value

        """
        string_hash = xxh3(string, seed=self.model_seed)
        value = self.value_table.get(string_hash & self.mask)
        if value is not None:
            return value

        return self.encode_miss(string_hash)

    def encode_miss(self, string_hash) -> float:
        """
        Encodes string hash as a miss

        Parameters
        ----------
        string_hash: int
            string hash to be encoded as a miss

        Returns
        -------
        float
            encoded miss value

        """
        # hash to float in range [-miss_width/2, miss_width/2]
        # 32 bit mask for JS portability
        return scale((string_hash & 0xFFFFFFFF) * 2 ** -32, self.miss_width)


def scale(val: float, width: float = 2) -> float:
    """
    Scales input miss value to [-width/2, width/2].
    Assumes input is within [0, 1] range.

    Parameters
    ----------
    val: float
        miss value to be scaled
    width: float
        miss range width

    Returns
    -------
    float
        scaled miss value

    """
    assert width >= 0
    # map value in [0, 1] to [-width/2, width/2]
    return val * width - 0.5 * width


def get_mask(string_table: list) -> int:
    """
    Returns an integer representation of a binary mask for a given string table

    Parameters
    ----------
    string_table: list
        list of hash string values for a given feature

    Returns
    -------
    int
        number of bytes needed to represent string hashed in the table

    """
    if len(string_table) == 0:
        return 0

    max_value = max(string_table)
    if max_value == 0:
        return 0

    # find the most significant bit in the table and create a mask
    return (1 << int(np.log2(max_value) + 1)) - 1
