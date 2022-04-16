#!python3
#cython: language_level=3

cdef extern from "npy_no_deprecated_api.h": pass

import cython
import numpy as np
cimport numpy as np


cpdef encoded_variant_into_np_row(
        dict encoded_variant, list feature_names, np.ndarray into):
    """
    Fills in `into` row using provided variants and `feature_names`

    Parameters
    ----------
    encoded_variant: dict
        dict containing fully encoded variant (variant + givens)
    feature_names: np.ndarray or list
        collection of desired feature names
    into: np.ndarray
        array to be filled with encoded

    Returns
    -------
    None
        None

    """

    cdef dict hash_index_map = \
        {feature_hash: index for index, feature_hash
         in enumerate(feature_names)}

    cdef np.ndarray filler = \
        np.array(
            [(hash_index_map.get(feature_name, None), value)
             for feature_name, value in encoded_variant.items()
             if hash_index_map.get(feature_name, None) is not None])

    # sum into with encoded variants treating nans in sums as zeros
    cdef np.ndarray subset_index = np.empty(len(filler))

    if len(filler) > 0:

        subset_index = filler[:, 0].astype(int)

        into[subset_index] = np.nansum(
            np.array([into[subset_index], filler[:, 1]]), axis=0)


@cython.boundscheck(False)
cpdef np.ndarray encode_variants_single_givens(
        list variants, dict givens, double noise,
        object variants_encoder, object givens_encoder):

    cdef dict into = {}

    cdef dict encoded_givens = \
        givens_encoder(givens, noise, into=dict(into))

    cdef int variants_count = len(variants)
    cdef np.ndarray res = np.empty((variants_count, ), dtype=object)

    cdef dict fully_encoded_variant = {}

    for variant_idx in range(variants_count):

        fully_encoded_variant = \
            variants_encoder(
                variants[variant_idx], noise, into=dict(encoded_givens))

        res[variant_idx] = fully_encoded_variant

    return res

@cython.boundscheck(False)
cpdef np.ndarray encode_variants_multiple_givens(
        list variants, list multiple_givens, list multiple_extra_features,
        double noise, object feature_encoder, object givens_encoder):

    assert len(variants) == len(multiple_givens)
    cdef int records_count = len(variants)
    cdef np.ndarray res = np.empty((records_count,), dtype=object)

    cdef dict empty_into = {}

    cdef dict givens = {}
    cdef dict extra_features = {}
    cdef dict encoded_givens = {}
    cdef dict fully_encoded_variant = {}

    for variant_idx in range(records_count):

        encoded_givens = {}
        givens = multiple_givens[variant_idx]
        if givens:
            encoded_givens = \
                givens_encoder(givens, noise, into=dict(empty_into))

        res[variant_idx] = \
            feature_encoder(variants[variant_idx], noise, into=encoded_givens)

        extra_features = multiple_extra_features[variant_idx]
        if extra_features:
            res[variant_idx].update(extra_features)

    return res


@cython.boundscheck(False)
cpdef double[:, :] encoded_variants_to_np(
        np.ndarray encoded_variants, list feature_names):

    cdef np.ndarray encoded_variants_array = \
        np.empty(shape=(len(encoded_variants), len(feature_names)), dtype=float)

    cdef dict name_to_index = \
        {feature_name: feature_index for feature_index, feature_name
         in enumerate(feature_names)}

    encoded_variants_array[:, :] = np.nan

    for encoded_variant_ixd, processed_variant in enumerate(encoded_variants):

        # this assumes that we have much less features per single variant that
        # total number of features
        filler = \
            np.array(
                [(name_to_index.get(feature_name, None), value)
                 for feature_name, value in processed_variant.items()
                 if name_to_index.get(feature_name, None) is not None])

        if len(filler) > 0:
            encoded_variants_array[encoded_variant_ixd, filler[:, 0]\
                .astype(int)] = filler[:, 1]

    return encoded_variants_array
