#!python3
#cython: language_level=3

cdef extern from "npy_no_deprecated_api.h": pass

import cython
from libc.math cimport isnan
import numpy as np
cimport numpy as np
import xxhash

from improveai.feature_encoder import ITEM_FEATURE_KEY, CONTEXT_FEATURE_KEY

cdef object xxh3 = xxhash.xxh3_64_intdigest


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double scale(double val, double width=2):
    # map value in [0, 1] to [-width/2, width/2]
    return val * width - 0.5 * width

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef unsigned long long get_mask(list string_table):
    if len(string_table) == 0:
        return 0

    cdef unsigned long long max_value = max(string_table)
    if max_value == 0:
        return 0

    # find the most significant bit in the table and create a mask
    return (1 << int(np.log2(max_value) + 1)) - 1


cdef class StringTable:

    cdef public unsigned long long model_seed
    cdef public unsigned long mask
    cdef public double miss_width
    cdef public dict value_table


    def __init__(self, list string_table, unsigned long long model_seed):

        if model_seed < 0:
            raise ValueError(
                "xxhash3_64 takes an unsigned 64-bit integer as seed. "
                "Seed must be greater than or equal to 0.")

        self.model_seed = model_seed
        self.mask = get_mask(string_table)
        cdef unsigned long long max_position = len(string_table) - 1

        # empty and single entry tables will have a miss_width of 1 or range [-0.5, 0.5]
        # 2 / max_position keeps miss values from overlapping with nonzero table values
        self.miss_width = 1 if max_position < 1 else 2 / max_position

        self.value_table = {}

        for index, string_hash in enumerate(reversed(string_table)):
            # a single entry gets a value of 1.0
            self.value_table[string_hash] = 1.0 if max_position == 0 else scale(
                index / max_position)

    cpdef double encode(self, str string):
        cdef unsigned long long string_hash = xxh3(string, seed=self.model_seed)

        # TODO validate against vanilla FE implementation
        cdef unsigned long long masked_hash = string_hash & self.mask
        if masked_hash in self.value_table:
            return self.value_table[masked_hash]

        # TODO delete after tests!
        # print(f'\n`{string}` is a value absent in the string table -> returning miss encoding!')
        return self.encode_miss(string_hash)

    cpdef double encode_miss(self, string_hash):
        # TODO !! important note -> for negative exponents the base must be of
        #  a float type
        # hash to float in range [-miss_width/2, miss_width/2]
        # 32 bit mask for JS portability
        return scale((string_hash & 0xFFFFFFFF) * 2.0 ** -32, self.miss_width)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple get_noise_shift_scale(double noise):
    assert noise >= 0.0 and noise < 1.0
    # x + noise * 2 ** -142 will round to x for most values of x. Used to create
    # distinct values when x is 0.0 since x * scale would be zero
    return (noise * 2 ** -142, 1 + noise * 2 ** -17)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double sprinkle(double x, double noise_shift, double noise_scale):
    # x + noise_offset will round to x for most values of x
    # allows different values when x == 0.0
    return (x + noise_shift) * noise_scale


cdef class FeatureEncoder:
    """
    Encodes JSON encodable objects into float vectors
    """

    cdef public dict feature_indexes
    cdef public list string_tables

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

    cpdef void encode_item(
            self, object item, np.ndarray[double, ndim=1, mode='c'] into,
            double noise_shift = 0.0, double noise_scale = 1.0):
        self._encode(item, path=ITEM_FEATURE_KEY, into=into,
                     noise_shift=noise_shift, noise_scale=noise_scale)

    cpdef void encode_context(
            self, object context, np.ndarray[double, ndim=1, mode='c'] into,
            double noise_shift = 0.0, double noise_scale = 1.0):
        self._encode(context, path=CONTEXT_FEATURE_KEY, into=into,
                     noise_shift=noise_shift, noise_scale=noise_scale)

    cpdef void encode_feature_vector(
            self, object item, object context, np.ndarray[double, ndim=1, mode='c'] into,
            double noise: float = 0.0):
        """
        Fully encodes provided item and context into a np.ndarray provided as `into` parameter.
        `into` must not be None

        Parameters
        ----------
        item: object
            a JSON encodable object to be encoded
        context: object
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
        cdef float noise_shift
        cdef float noise_scale
        noise_shift, noise_scale = get_noise_shift_scale(noise)

        if item is not None:
            self.encode_item(item, into, noise_shift, noise_scale)

        if context is not None:
            self.encode_context(context, into, noise_shift, noise_scale)

    cdef void _encode(
            self, object obj, str path, np.ndarray[double, ndim=1, mode='c'] into,
            double noise_shift = 0.0, double noise_scale = 1.0):
        """
        Encodes a JSON serializable object to a float vector
        Rules of encoding go as follows:

        - None, json null, {}, [], and nan are treated as missing features and ignored.

        - numbers and booleans are encoded as-is.

        - strings are encoded using a lookup table

        Parameters
        ----------
        object_: object
            a JSON serializable object to be encoded to a flat key-value structure
        seed: int
            seed for xxhash3 to generate feature name
        features: dict
            a flat dict of {<feature name>: <feature value>, ...} pairs

        Returns
        -------
        None
        """

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
            raise ValueError(
                f'{obj} not JSON encodable. Must be string, int, float, bool, list, tuple, dict, or None')


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[double, ndim=2, mode='c'] encode_candidates_to_matrix(
        object candidates, object context, object feature_encoder, double noise=0.0):

    cdef np.ndarray[double, ndim=2, mode='c'] into_matrix = \
        np.full((len(candidates), len(feature_encoder.feature_indexes)), np.nan)

    for item, into_row in zip(candidates, into_matrix):
        feature_encoder.encode_feature_vector(item=item, context=context, into=into_row, noise=noise)

    return into_matrix
