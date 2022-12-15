#!python3
#cython: language_level=3

cdef extern from "npy_no_deprecated_api.h": pass

import cython
import numpy as np
cimport numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[double, ndim=2, mode='c'] encode_variants_to_matrix(
        object variants, object givens, object feature_encoder, double noise=0.0):

    cdef np.ndarray[double, ndim=2, mode='c'] into_matrix = \
        np.full((len(variants), len(feature_encoder.feature_indexes)), np.nan)

    for variant, into_row in zip(variants, into_matrix):
        # variant: object, givens: object, extra_features: dict, into: np.ndarray, noise: float
        feature_encoder.encode_feature_vector(
            variant=variant, givens=givens, extra_features=None, into=into_row, noise=noise)

    return into_matrix


# @cython.boundscheck(False)
# @cython.wraparound(False)
# cpdef encoded_variant_into_np_row(
#         dict encoded_variant, list feature_names, np.ndarray into):
#     """
#     Fills in `into` row using provided variants and `feature_names`
#
#     Parameters
#     ----------
#     encoded_variant: dict
#         dict containing fully encoded variant (variant + givens)
#     feature_names: np.ndarray or list
#         collection of desired feature names
#     into: np.ndarray
#         array to be filled with encoded
#
#     Returns
#     -------
#     None
#         None
#
#     """
#
#     cdef dict hash_index_map = \
#         {feature_hash: index for index, feature_hash
#          in enumerate(feature_names)}
#
#     cdef np.ndarray filler = \
#         np.array(
#             [(hash_index_map.get(feature_name, None), value)
#              for feature_name, value in encoded_variant.items()
#              if hash_index_map.get(feature_name, None) is not None])
#
#     # sum into with encoded variants treating nans in sums as zeros
#     cdef np.ndarray subset_index = np.empty(len(filler))
#
#     if len(filler) > 0:
#
#         subset_index = filler[:, 0].astype(int)
#
#         into[subset_index] = np.nansum(
#             np.array([into[subset_index], filler[:, 1]]), axis=0)
#
#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cpdef np.ndarray encode_variants_single_givens(
#         list variants, dict givens, double noise,
#         object variants_encoder, object givens_encoder):
#     """
#     Encodes provided variants to a 2D array with single givens
#
#     Parameters
#     ----------
#     variants: list
#         list of encoded variants
#     givens: dict
#         givens to be encoded with each variant
#     noise: float
#         value within 0 - 1 range which will be 'shrunk' and added to feature value
#     variants_encoder: callable
#         a callable to be used as a variants encoder
#     givens_encoder: callable
#         a callable to be used as a givens encoder
#
#     Returns
#     -------
#     np.ndarray
#         2D array with encoded variants; missing values are filled with np.nan
#
#     """
#
#     cdef dict into = {}
#
#     cdef dict encoded_givens = \
#         givens_encoder(givens, noise, into=dict(into))
#
#     cdef int variants_count = len(variants)
#     cdef np.ndarray res = np.empty((variants_count, ), dtype=object)
#
#     cdef dict fully_encoded_variant = {}
#
#     for variant_idx in range(variants_count):
#
#         fully_encoded_variant = \
#             variants_encoder(
#                 variants[variant_idx], noise, into=dict(encoded_givens))
#
#         res[variant_idx] = fully_encoded_variant
#
#     return res
#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cpdef np.ndarray encode_variants_multiple_givens(
#         list variants, list multiple_givens, list multiple_extra_features,
#         double noise, object variants_encoder, object givens_encoder):
#     """
#     Encode each variant with corresponding givens
#
#     Parameters
#     ----------
#     variants: list
#         list of variants to be encoded
#     multiple_givens: list
#         list of givens - each element corresponds to a single variant which
#     multiple_extra_features: list
#         list of extra features  - each element is a set of extra features for a single variant - givens pair
#     noise: float
#         value within 0 - 1 range which will be 'shrunk' and added to feature value
#     variants_encoder: callable
#         a callable to be used as a variants encoder
#     givens_encoder: callable
#         a callable to be used as a givens encoder
#
#     Returns
#     -------
#     np.ndarray
#         2D array with encoded variants; missing values are filled with np.nan
#
#     """
#
#     assert len(variants) == len(multiple_givens)
#     cdef int records_count = len(variants)
#     cdef np.ndarray res = np.empty((records_count,), dtype=object)
#
#     cdef dict empty_into = {}
#
#     cdef dict givens = {}
#     cdef dict extra_features = {}
#     cdef dict encoded_givens = {}
#     cdef dict fully_encoded_variant = {}
#
#     for variant_idx in range(records_count):
#
#         encoded_givens = {}
#         givens = multiple_givens[variant_idx]
#         if givens:
#             encoded_givens = \
#                 givens_encoder(givens, noise, into=dict(empty_into))
#
#         res[variant_idx] = \
#             variants_encoder(variants[variant_idx], noise, into=encoded_givens)
#
#         extra_features = multiple_extra_features[variant_idx]
#         if extra_features:
#             res[variant_idx].update(extra_features)
#
#     return res
#
#
# @cython.boundscheck(False)
# @cython.wraparound(False)
# cpdef double[:, :] encoded_variants_to_np(
#         np.ndarray encoded_variants, list feature_names):
#     """
#     For each of `encoded_variants` selects features overlapping with `feature_names`,
#     fill missing features with np.nan and returns 2D array with all variants
#
#     Parameters
#     ----------
#     encoded_variants: np.ndarray
#         as array with all variants to be encoded
#     feature_names: list
#         list of desires feature names
#
#     Returns
#     -------
#     np.ndarray
#         a 2D array with all encoded variants
#
#     """
#
#     cdef np.ndarray encoded_variants_array = \
#         np.empty(shape=(len(encoded_variants), len(feature_names)), dtype=float)
#
#     cdef dict name_to_index = \
#         {feature_name: feature_index for feature_index, feature_name
#          in enumerate(feature_names)}
#
#     encoded_variants_array[:, :] = np.nan
#
#     for encoded_variant_ixd, processed_variant in enumerate(encoded_variants):
#
#         # this assumes that we have much less features per single variant that
#         # total number of features
#         filler = \
#             np.array(
#                 [(name_to_index.get(feature_name, None), value)
#                  for feature_name, value in processed_variant.items()
#                  if name_to_index.get(feature_name, None) is not None])
#
#         if len(filler) > 0:
#             encoded_variants_array[encoded_variant_ixd, filler[:, 0]\
#                 .astype(int)] = filler[:, 1]
#
#     return encoded_variants_array
