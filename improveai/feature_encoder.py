import numpy as np
import xxhash


VARIANT_FEATURE_KEY = 'v'
GIVENS_FEATURE_KEY = 'g'

xxh3 = xxhash.xxh3_64_intdigest


class FeatureEncoder:
    """
    Encodes JSON encodable objects into float vectors
    """

    def __init__(self, feature_names: list, string_tables: dict, model_seed: int):
        """
        Initialize the feature encoder for this model

        Parameters
        ----------
        feature_names: list
            the feature names in order of how they are vectorized
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

    def encode_variant(self, variant, into: np.ndarray, noise_shift: float = 0.0, noise_scale: float = 1.0):
        self._encode(variant, path=VARIANT_FEATURE_KEY, into=into, noise_shift=noise_shift, noise_scale=noise_scale)

    def encode_givens(self, givens, into: np.ndarray, noise_shift: float = 0.0, noise_scale: float = 1.0):
        self._encode(givens, path=GIVENS_FEATURE_KEY, into=into, noise_shift=noise_shift, noise_scale=noise_scale)

    def encode_feature_vector(
            self, variant: object, givens: object, into: np.ndarray, noise: float = 0.0):
        """
        Fully encodes provided variant and givens into a np.ndarray provided as `into` parameter.
        `into` must not be None

        Parameters
        ----------
        variant: object
            a JSON encodable object to be encoded
        givens: object
            a JSON encodable object to be encoded
        into: np.ndarray
            an array into which feature values will be added
        noise: float
            value in [0, 1) which will be combined with the feature value

        Returns
        -------
        None
            None

        """
        noise_shift, noise_scale = get_noise_shift_scale(noise)

        if variant:
            self.encode_variant(variant, into, noise_shift, noise_scale)

        if givens:
            self.encode_givens(givens, into, noise_shift, noise_scale)

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
            JSON path of an encoded object (so far)
        into: np.ndarray
            a numpy array into which encoded value will be written
        noise_shift: float
            a noise shift
        noise_scale: float
            a noise scale

        Returns
        -------
        None
        """

        if isinstance(obj, (int, float)):  # bool is an instanceof int
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


def get_noise_shift_scale(noise):
    assert noise >= 0.0 and noise < 1.0
    # x + noise * 2 ** -142 will round to x for most values of x. Used to create
    # distinct values when x is 0.0 since x * scale would be zero
    return (noise * 2 ** -142, 1 + noise * 2 ** -17)


def sprinkle(x, noise_shift, noise_scale):
    # x + noise_offset will round to x for most values of x
    # allows different values when x == 0.0
    return (x + noise_shift) * noise_scale


class StringTable:
    
    def __init__(self, string_table, model_seed):
        
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

    def encode(self, string):
        string_hash = xxh3(string, seed=self.model_seed)
        value = self.value_table.get(string_hash & self.mask)
        if value is not None:
            return value

        print('This is a values absent in the string table -> returning miss encoding!')
        return self.encode_miss(string_hash)

    def encode_miss(self, string_hash):
        # hash to float in range [-miss_width/2, miss_width/2]
        # 32 bit mask for JS portability
        small_string = f'{(np.float64(2 ** -32)).view(np.uint64):0>64b}'
        print('### small_string ###')
        print(np.float64(2 ** -32))
        print(small_string)
        return scale((string_hash & 0xFFFFFFFF) * 2 ** -32, self.miss_width)


def scale(val, width=2):
    # map value in [0, 1] to [-width/2, width/2]
    return val * width - 0.5 * width


def get_mask(string_table):

    if len(string_table) == 0:
        return 0

    max_value = max(string_table)
    if max_value == 0:
        return 0

    # find the most significant bit in the table and create a mask
    return (1 << int(np.log2(max_value) + 1)) - 1
        