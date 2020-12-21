#!python3
#cython: language_level=3

import cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
cpdef str[:] get_all_feat_names(int feat_count):
    feats_idxs = np.arange(feat_count).astype(object)

    for el_ixd in range(feat_count):
        feats_idxs[el_ixd] = 'f' + str(feats_idxs[el_ixd])

    return feats_idxs

@cython.boundscheck(False)
cpdef double[:,:] fast_get_nan_filled_encoded_variant(
        dict variant, dict context, int all_feats_count, object feature_encoder,
        float missing_filler = np.nan):

    cdef dict context_copy = dict(context)
    cdef dict encoded_variant = \
        feature_encoder.encode_features({'variant': variant})
    context_copy.update(encoded_variant)
    # print(context_copy)

    cdef np.ndarray missings_filled_v = \
        np.asarray(np.empty((1, all_feats_count)))
    # cdef long[:] encoded_variant_keys = \
    #     np.array([el for el in context_copy.keys()])
    cdef long[:] all_indices = np.arange(all_feats_count)


    missings_filled_v[0, :] = missing_filler
    missings_filled_v[0, np.array(list(context_copy.keys())).astype(int)] = \
        np.array(list(context_copy.values()))

    return missings_filled_v


# TODO check if passing encode_features() method makes this faster !!!
@cython.boundscheck(False)
cpdef double[:,:] get_nan_filled_encoded_variants(
        dict[:] variants, dict context, int all_feats_count,
        object feature_encoder, float missing_filler = np.nan):

    cdef int variants_count = len(variants)
    cdef np.ndarray res = \
        np.asarray(np.empty((variants_count, all_feats_count)))

    res[:,:] = missing_filler

    cdef dict context_copy
    cdef dict encoded_variant

    for v_idx in range(variants_count):
        context_copy = dict(context)
        encoded_variant = \
            feature_encoder.encode_features({'variant': variants[v_idx]})
        context_copy.update(encoded_variant)

        res[np.array([v_idx]).astype(int),
            np.array(list(context_copy.keys())).astype(int)] = \
            np.array(list(context_copy.values()))

    return res
