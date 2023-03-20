#!python3
#cython: language_level=3

cdef extern from "npy_no_deprecated_api.h": pass

import cython
from libc.math cimport isnan
import numpy as np
cimport numpy as np
import xxhash

from improveai.feature_encoder import ITEM_FEATURE_KEY, CONTEXT_FEATURE_KEY, \
    FIRST_LEVEL_FEATURES_CHUNKS

cdef object xxh3 = xxhash.xxh3_64_intdigest


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double scale(double val, double width=2.0) except *:
    """
    Scales input miss value to [-width/2, width/2]

    Parameters
    ----------
    val: double
        miss value to be scaled
    width: double
        miss range width

    Returns
    -------
    double
        scaled miss value

    """
    assert width >= 0.0
    # map value in [0, 1] to [-width/2, width/2]
    return val * width - 0.5 * width

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef unsigned long long get_mask(list string_table) except *:
    """
    Returns a hash string mask for a given string table

    Parameters
    ----------
    string_table: list
        list of hash string values for a given feature

    Returns
    -------
    unsigned long long
        number of bytes needed to represent string hashed in the table

    """
    if len(string_table) == 0:
        return 0

    cdef unsigned long long max_value = max(string_table)
    if max_value == 0:
        return 0

    # find the most significant bit in the table and create a mask
    return (1 << int(np.log2(max_value) + 1)) - 1


cdef class StringTable:
    """
    A class responsible for target encoding of strings
    """

    # int represents 32 bit int
    cdef public unsigned int model_seed
    """
    32-bit random integer used for string hashing with xxhash
    """

    # long represents 64 bit int
    cdef public unsigned long mask
    """
    At most 64 bit int representation of a string hash mask e.g., 000..00111
    """

    cdef public double miss_width
    """
    Float value representing width of the 0-centered miss numerical interval    
    """

    cdef public dict value_table
    """
    A mapping from masked string hash to target encoding's target value for a given feature
    """


    def __init__(self, list string_table, unsigned long long model_seed):
        """
        Init StringTable with params

        Parameters
        ----------
        string_table: dict
            a dict with list of masked hashed for each string feature
        model_seed: int
            model seed value
        """

        if model_seed < 0:
            raise ValueError(
                "xxhash3_64 takes an unsigned 64-bit integer as seed. "
                "Seed must be greater than or equal to 0.")

        self.model_seed = model_seed
        self.mask = get_mask(string_table)
        cdef long long max_position = len(string_table) - 1

        # empty and single entry tables will have a miss_width of 1 or range [-0.5, 0.5]
        # 2 / max_position keeps miss values from overlapping with nonzero table values
        self.miss_width = 1 if max_position < 1 else 2 / max_position

        self.value_table = {}

        for index, string_hash in enumerate(reversed(string_table)):
            # a single entry gets a value of 1.0
            self.value_table[string_hash] = 1.0 if max_position == 0 else scale(index / max_position)

    cpdef double encode(self, str string):
        """
        Encode input string to a target value

        Parameters
        ----------
        string: str
            string to be encoded

        Returns
        -------
        double
            encoded value

        """
        cdef unsigned long long string_hash = xxh3(string, seed=self.model_seed)

        # TODO validate against vanilla FE implementation
        cdef unsigned long long masked_hash = string_hash & self.mask
        if masked_hash in self.value_table:
            return self.value_table[masked_hash]

        return self.encode_miss(string_hash)

    cpdef double encode_miss(self, unsigned long string_hash):
        """
        Encodes string hash as a miss

        Parameters
        ----------
        string_hash: unsigned long
            string hash to be encoded as a miss

        Returns
        -------
        double
            encoded miss value

        """
        # TODO !! important note -> for negative exponents the base must be of
        #  a float type
        # hash to float in range [-miss_width/2, miss_width/2]
        # 32 bit mask for JS portability
        return scale((string_hash & 0xFFFFFFFF) * 2.0 ** -32, self.miss_width)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple get_noise_shift_scale(double noise):
    """
    Returns noise shift (small value added to feature value) and noise scale
    (value by which shifted feature value is multiplied)

    Parameters
    ----------
    noise: double
        value in [0, 1) which will be combined with the feature value


    Returns
    -------
    tuple
        tuple of double: (noise_shift, noise_scale)

    """
    assert noise >= 0.0 and noise < 1.0
    # x + noise * 2 ** -142 will round to x for most values of x. Used to create
    # distinct values when x is 0.0 since x * scale would be zero
    return (noise * 2.0 ** -142, 1 + noise * 2.0 ** -17)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double sprinkle(double x, double noise_shift, double noise_scale):
    """
    Adds noise shift and scales shifted value

    Parameters
    ----------
    x: double
        value to be sprinkled
    noise_shift: double
        small bias added to the feature value
    noise_scale: double
        small multiplier of the shifted feature value

    Returns
    -------
    double
        sprinkled value

    """
    # x + noise_offset will round to x for most values of x
    # allows different values when x == 0.0
    return (x + noise_shift) * noise_scale


cdef class FeatureEncoder:
    """
    Encodes JSON encodable objects into float vectors
    """

    cdef public dict feature_indexes
    """
    A map between feature names and feature indexes. Created by simple 
    iteration with enumeration over feature names 
    """

    cdef public list string_tables
    """
    List of StringTable objects indexed according to order present in the 
    constructor's string_tables param    
    """

    def __init__(self, list feature_names, dict string_tables, long long model_seed):
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
        if (model_seed < 0):
            raise TypeError(
                "xxhash3_64 takes an unsigned 64-bit integer as seed. "
                "Seed must be greater than or equal to 0.")

        self.feature_indexes = {}

        for i, feature_name in enumerate(feature_names):
            self.feature_indexes[feature_name] = i

        self.string_tables = [StringTable([], model_seed)] * len(feature_names)

        try:
            for feature_name, table in string_tables.items():
                self.string_tables[self.feature_indexes[feature_name]] = StringTable(table, model_seed)
        except KeyError as exc:
            raise ValueError("Bad model metadata") from exc

    cdef void _check_into(self, np.ndarray[double, ndim=1, mode='c'] into) except *:
        """
        Checks if the provided into array is an array and has desired dtype

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


    cpdef void encode_item(
            self, object item, np.ndarray[double, ndim=1, mode='c'] into,
            double noise_shift = 0.0, double noise_scale = 1.0) except *:
        """
        Encodes provided item to input numpy array

        Parameters
        ----------
        item: object
            JSON encodable python object
        into: np.ndarray[double, ndim=1, mode='c']
            array storing results of encoding
        noise_shift: double
            value to be added to features
        noise_scale: double
            multiplier used to scale shifted feature value

        Returns
        -------

        """
        self._encode(item, path=ITEM_FEATURE_KEY, into=into, noise_shift=noise_shift, noise_scale=noise_scale)

    cpdef void encode_context(
            self, object context, np.ndarray[double, ndim=1, mode='c'] into,
            double noise_shift = 0.0, double noise_scale = 1.0) except *:
        """
        Encodes provided context to input numpy array

        Parameters
        ----------
        context: object
            JSON encodable python object
        into: np.ndarray[double, ndim=1, mode='c']
            array storing results of encoding
        noise_shift: double
            value to be added to features
        noise_scale: double
            multiplier used to scale shifted feature value

        Returns
        -------

        """
        self._encode(context, path=CONTEXT_FEATURE_KEY, into=into, noise_shift=noise_shift, noise_scale=noise_scale)

    cpdef void encode_feature_vector(
            self, object item, object context, np.ndarray[double, ndim=1, mode='c'] into,
            double noise: float = 0.0) except *:
        """
        Fully encodes provided variant and givens into a np.ndarray provided as `into` parameter.
        `into` must not be None

        Parameters
        ----------
        item: object
            a JSON encodable object to be encoded
        context: object
            a JSON encodable object to be encoded
        into: np.ndarray[double, ndim=1, mode='c']
            an array into which feature values will be added
        noise: double
            value in [0, 1) which will be combined with the feature value

        Returns
        -------
        None

        """

        cdef double noise_shift
        cdef double noise_scale
        noise_shift, noise_scale = get_noise_shift_scale(noise)

        if item is not None:
            self.encode_item(item, into, noise_shift, noise_scale)

        if context is not None:
            self.encode_context(context, into, noise_shift, noise_scale)

        # if both item and context are None into will not be checked until now.
        if item is None and context is None:
            self._check_into(into=into)


    cpdef void _encode(
            self, object obj, str path, np.ndarray[double, ndim=1, mode='c'] into,
            double noise_shift = 0.0, double noise_scale = 1.0) except *:
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
            the path to the current object
        into: np.ndarray[double, ndim=1, mode='c']
            an array into which feature values will be encoded
        noise_shift: double
            small bias added to the feature value
        noise_scale: double
            small multiplier of the feature value

        Returns
        -------
        None
        """

        if path in FIRST_LEVEL_FEATURES_CHUNKS:
            self._check_into(into)

        cdef object feature_index
        cdef StringTable string_table

        if isinstance(obj, (int, float)):  # bool is an instanceof int
            if isnan(obj):  # nan is treated as missing feature, return
                return

            # TODO feature_index might be of an int type but .get() returns None
            #  if path is not found in self.feature_indexes
            feature_index = self.feature_indexes.get(path)
            if feature_index is None:
                return

            into[feature_index] = sprinkle(obj, noise_shift, noise_scale)

        elif isinstance(obj, str):
            feature_index = self.feature_indexes.get(path)
            if feature_index is None:
                return

            string_table = self.string_tables[feature_index]

            # TODO is that the best way to go? str(obj)
            into[feature_index] = sprinkle(string_table.encode(str(obj)), noise_shift, noise_scale)

        elif isinstance(obj, dict):
            for key, value in obj.items():
                self._encode(obj=value, path=path + '.' + key, into=into, noise_shift=noise_shift, noise_scale=noise_scale)

        elif isinstance(obj, (list, tuple)):
            for index, item in enumerate(obj):
                self._encode(obj=item, path=path + '.' + str(index), into=into, noise_shift=noise_shift, noise_scale=noise_scale)

        elif obj is None:
            pass

        else:
            raise ValueError(
                f'{obj} not JSON encodable. Must be string, int, float, bool, list, tuple, dict, or None')


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[double, ndim=2, mode='c'] encode_candidates_to_matrix(
        object candidates, object context, object feature_encoder, double noise=0.0):
    """
    Encodes list of candidates to 2D np.array for a given context with provided noise

    Parameters
    ----------
    candidates: object
        list or tuple or np.ndarray of JSON encodable candidates / items to encode
    context: object
        JSON encodable object
    noise: double
        noise to be used for sprinkling of encoded features

    Returns
    -------
    np.ndarray[double, ndim=2, mode='c']
        2D numpy array with encoded candidates

    """

    cdef np.ndarray[double, ndim=2, mode='c'] into_matrix = \
        np.full((len(candidates), len(feature_encoder.feature_indexes)), np.nan)

    for item, into_row in zip(candidates, into_matrix):
        feature_encoder.encode_feature_vector(item=item, context=context, into=into_row, noise=noise)

    return into_matrix
