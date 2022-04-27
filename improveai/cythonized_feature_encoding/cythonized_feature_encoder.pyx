#!python3
#cython: language_level=3

cdef extern from "npy_no_deprecated_api.h": pass

import cython
from libc.math cimport isnan
import numpy as np
cimport numpy as np
import warnings
import xxhash

cdef object xxhash3 = xxhash.xxh3_64_intdigest
cdef list JSON_SERIALIZABLE_TYPES = [int, float, str, bool, list, tuple, dict]

import improveai.cythonized_feature_encoding.cythonized_feature_encoding_utils as cfeu
import improveai.feature_encoder as fe


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _is_object_json_serializable(object object_):
    """
    Checks if input value is JSON serializable

    Parameters
    ----------
    object_: object
        object to be checked

    Returns
    -------
    bool
        True if input is JSON serializable False otherwise
    """
    return any([isinstance(object_, type_) for type_ in JSON_SERIALIZABLE_TYPES]) or object_ is None


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _has_top_level_string_keys(checked_dict):
    """
    Check if all top level keys are of a string type. This is a helper function 
    for encode() and since encode recurs in case of nested dicts only top level 
    keys need to be checked. Return False on first encountered non-string key

    Parameters
    ----------
    checked_dict: dict
        dict which top level keys will be checked

    Returns
    -------
    bool
        True if all keys are strings otherwise False

    """
    return all([isinstance(k, str) for k in checked_dict.keys()])


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef warn_about_array_encoding(object object_):
    """
    If object is an array warning will be printed (only once so the encoding speed does not suffer)

    Parameters
    ----------
    object_: object
        encoded object

    Returns
    -------
    None
        None

    """
    # check if variant is a list, tuple or array
    if not fe.WARNED_ABOUT_ARRAY_ENCODING and \
            (isinstance(object_, list) or isinstance(object_, tuple) or isinstance(object_, np.ndarray)):
        warnings.warn('Array encoding may change in the near future')
        fe.WARNED_ABOUT_ARRAY_ENCODING = True

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef encode(object object_, unsigned long long seed, double small_noise, dict features):

    assert _is_object_json_serializable(object_=object_)
    warn_about_array_encoding(object_=object_)

    cdef str feature_name = None
    cdef double previous_object_

    if isinstance(object_, (int, float)):  # bool is an instanceof int
        if isnan(object_):  # nan is treated as missing feature, return
            return

        feature_name = hash_to_feature_name(seed)

        previous_object_ = \
            _get_previous_value(
                feature_name=feature_name, into=features,
                small_noise=small_noise)

        features[feature_name] = \
            sprinkle(object_ + previous_object_, small_noise)

        return

    cdef unsigned long long hashed
    cdef double previous_hashed
    cdef str hashed_feature_name
    cdef double previous_hashed_for_feature_name

    if isinstance(object_, str):
        hashed = xxhash3(object_, seed=seed)

        feature_name = hash_to_feature_name(seed)

        previous_hashed = \
            _get_previous_value(
                feature_name=feature_name, into=features,
                small_noise=small_noise)

        features[feature_name] = \
            sprinkle(
                <double>(<long long>(((hashed & 0xffff0000) >> 16) - 0x8000)) + previous_hashed,
                small_noise)

        hashed_feature_name = hash_to_feature_name(hashed)

        previous_hashed_for_feature_name = \
            _get_previous_value(
                feature_name=hashed_feature_name, into=features,
                small_noise=small_noise)

        features[hashed_feature_name] = \
            sprinkle(
                <double>(<long long>((hashed & 0xffff) - 0x8000)) + previous_hashed_for_feature_name,
                small_noise)

        return

    if isinstance(object_, dict):
        assert _has_top_level_string_keys(object_)
        for key, value in object_.items():
            encode(
                value, xxhash3(key, seed=seed), small_noise, features)
        return

    if isinstance(object_, (list, tuple)):
        for index, item in enumerate(object_):
            encode(
                item, xxhash3(index.to_bytes(8, byteorder='big'), seed=seed),
                small_noise, features)

        return

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double _get_previous_value(str feature_name, dict into, double small_noise):
    """
    Gets previous, 'unsprinkled' value of <feature_name> feature
    from <features> dict

    Parameters
    ----------
    feature_name: str
        name of a feature which previous value is desired
    into: dict
        dict containing current results of encoding
    small_noise: float
        small noise used to sprinkle

    Returns
    -------
    float
        'unsprinkled' value of desired feature

    """

    cdef double  previous_sprinkled_object_

    if into.get(feature_name, None) is None:
        return 0.0
    else:
        previous_sprinkled_object_ = into.get(feature_name, None)
        return reverse_sprinkle(previous_sprinkled_object_, small_noise)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef str hash_to_feature_name(unsigned long long hash_):
    return '%0*x' % (8, (hash_ >> 32))


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double shrink(double noise):
    return noise * pow(2, -17)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double sprinkle(double x, double small_noise):
    return (x + small_noise) * (1 + small_noise)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double reverse_sprinkle(double sprinkled_x, double small_noise):
    """
    Retrieves true value from sprinkled one

    Parameters
    ----------
    sprinkled_x: float
        sprinkled value
    small_noise: float
        noise used to calculate sprinkled_Xx

    Returns
    -------
    float
        true value of x which was used to calculate sprinkled_x

    """
    return sprinkled_x / (1 + small_noise) - small_noise


cdef class FeatureEncoder:
    """
    This class serves as a preprocessor which allows to
    convert event data in form of a JSON line (variants, givens and
    extra features) to the tabular format expected on input by XGBoost.

    Short summary of methods:
     - encode_givens() -> encodes givens / variant metadata to dict of
       feature name -> float pairs

     - encode_variant() -> encodes variant to dict of feature name -> float pairs

     - encode_feature_vector() -> encodes provided variant and givens;
       appends extra features to the encoded variant dict;
       converts encoded variant to numpy array using provided feature names

     - add_noise() -> sprinkles each value of the input dict with small noise

     - encode_to_np_matrix()  -> encodes collection of variants with collection
       of givens and extra features to the numpy 2D array / matrix. If desired
       uses cython backed (works much faster)

     - add_extra_features() - helper method for adding extra features to already
       encoded variants

    """

    cdef unsigned long long variant_seed
    cdef unsigned long long value_seed
    cdef unsigned long long givens_seed

    def __init__(self, long long model_seed):
        if (model_seed < 0):
            raise TypeError(
                "xxhash3_64 takes an unsigned 64-bit integer as seed. "
                "Seed must be greater than or equal to 0.")

        # memoize commonly used seeds
        self.variant_seed = xxhash3("variant", seed=model_seed)
        self.value_seed = xxhash3("$value", seed=self.variant_seed)
        self.givens_seed = xxhash3("givens", seed=model_seed)

    cdef void _check_noise_value(self, double noise) except *:
        if noise > 1.0 or noise < 0.0:
            raise ValueError(
                'Provided `noise` is out of allowed bounds <0.0, 1.0>')

    cpdef dict encode_givens(self, dict givens, double noise=0.0, dict into=None):

        self._check_noise_value(noise=noise)

        if givens is not None and not isinstance(givens, dict):
            raise TypeError(
                "Only dict type is supported for context encoding. {} type was "
                "provided.".format(type(givens)))

        if into is None:
            into = {}

        encode(givens, self.givens_seed, shrink(noise), into)

        return into

    cpdef dict encode_variant(self, object variant, double noise=0.0, dict into=None):

        self._check_noise_value(noise=noise)

        if into is None:
            into = {}

        small_noise = shrink(noise)

        if isinstance(variant, dict):
            encode(variant, self.variant_seed, small_noise, into)
        else:
            encode(variant, self.value_seed, small_noise, into)

        return into


    cpdef encode_feature_vector(
            self, object variant = None, dict givens = None,
            dict extra_features = None, list feature_names = None,
            double noise = 0.0, np.ndarray into = None):

        if into is None:
            raise ValueError('`into` can`t be None')

        cdef dict encoded_givens = {}
        # encode givens
        encoded_givens = \
            self.encode_givens(givens=givens, noise=noise, into=dict())

        cdef dict encoded_variant = {}
        # encoded variant and givens
        encoded_variant = \
            self.encode_variant(
                variant=variant, noise=noise, into=encoded_givens)

        if extra_features:
            if not isinstance(extra_features, dict):
                raise TypeError(
                    'Provided `extra_features` is not a dict but: {}'
                    .format(extra_features))

            encoded_variant.update(extra_features)

        # n + nan = nan so you'll have to check for nan values on into
        cfeu.encoded_variant_into_np_row(
            encoded_variant=encoded_variant, feature_names=feature_names,
            into=into)

    cpdef double [:, :] encode_to_np_matrix(
            self, list variants, list multiple_givens,
            list multiple_extra_features, list feature_names, double noise):
        """
        Provided list of variants and corresponding lists of givens and extra
        features encodes variants completely and converts to numpy array

        Parameters
        ----------
        variants: list
            list of variants to be encoded
        multiple_givens: list
            list of givens - givens[i] will be used to encode variants[i]
        multiple_extra_features: list
            list of extra_features - multiple_extra_features[i] will be used
            to encode variants[i]
        feature_names: list
            names of features expected by the model
        noise: float
            noise to be used when encoding data

        Returns
        -------
        np.ndarray
            2D numpy array of encoded variants

        """

        encoded_variants = cfeu.encode_variants_multiple_givens(
            variants=variants, multiple_givens=multiple_givens,
            multiple_extra_features=multiple_extra_features, noise=noise,
            feature_encoder=self.encode_variant,
            givens_encoder=self.encode_givens)
        encoded_variants_array = cfeu.encoded_variants_to_np(
            encoded_variants=encoded_variants, feature_names=feature_names)

        return encoded_variants_array

    cpdef void add_extra_features(self, list encoded_variants, list extra_features):
        """
        Once variants are encoded this method can be used to quickly append
        extra features

        Parameters
        ----------
        encoded_variants: list
            collection of encoded variants to be updated with extra features
        extra_features: list
            payload to be appended to encoded variants

        Returns
        -------
        None
            None

        """

        if extra_features is None:
            return

        if not isinstance(encoded_variants, list):
            raise TypeError('`encoded_variants` should be of a list type')

        if not isinstance(extra_features, list):
            raise TypeError('`extra_features` should be of a list type')

        [encoded_variant.update(single_extra_features)
         if single_extra_features is not None else encoded_variant
         for encoded_variant, single_extra_features in
         zip(encoded_variants, extra_features)]
