#!python3
#cython: language_level=3

import cython
import numpy as np
cimport numpy as np


@cython.boundscheck(False)
cpdef np.ndarray encode_variants_single_context(
        np.ndarray variants, dict context, double noise, object feature_encoder,
        object context_encoder):

    cdef dict encoded_context = context_encoder(context, noise)
    cdef int variants_count = len(variants)
    cdef np.ndarray res = np.empty((variants_count, ), dtype=object)
    cdef dict encoded_variant = {}
    cdef dict fully_encoded_variant = {}

    for variant_idx in range(variants_count):

        encoded_variant = \
            feature_encoder(variants[variant_idx], noise)

        fully_encoded_variant = {}

        for k in set(encoded_context) | set(encoded_variant):
            fully_encoded_variant[k] = \
                encoded_context.get(k, 0) + encoded_variant.get(k, 0)

        # fully_encoded_variant = \
        #     {k: encoded_context.get(k, 0) + encoded_variant.get(k, 0)
        #       for k in set(encoded_context) | set(encoded_variant)}

        res[variant_idx] = fully_encoded_variant

    return res

@cython.boundscheck(False)
cpdef np.ndarray encode_variants_multiple_contexts(
        np.ndarray variants, np.ndarray contexts, double noise,
        object feature_encoder, object context_encoder):

    assert len(variants) == len(contexts)
    cdef int records_count = len(variants)
    cdef np.ndarray res = np.empty((records_count,), dtype=object)

    cdef dict context = {}

    cdef dict encoded_variant = {}
    cdef dict encoded_context = {}
    cdef dict fully_encoded_variant = {}

    for variant_idx in range(records_count):

        encoded_context = {}
        context = contexts[variant_idx]
        if context:
            encoded_context = \
                context_encoder(context, noise)

        encoded_variant = \
            feature_encoder(variants[variant_idx], noise)

        if not encoded_context:
            fully_encoded_variant = encoded_variant
        else:
            fully_encoded_variant = {}

            for k in set(encoded_context) | set(encoded_variant):
                fully_encoded_variant[k] = \
                    encoded_context.get(k, 0) + encoded_variant.get(k, 0)

        res[variant_idx] = fully_encoded_variant

    return res


@cython.boundscheck(False)
cpdef np.ndarray encode_jsonlines(
        np.ndarray jsonlines, double noise, str variant_key, str context_key,
        object feature_encoder, object context_encoder):

    cdef int records_count = len(jsonlines)
    cdef np.ndarray res = np.empty((records_count,), dtype=object)

    cdef dict encoded_variant = {}
    cdef dict encoded_context = {}
    cdef dict fully_encoded_variant = {}

    for jsonline_idx in range(records_count):

        encoded_context = {}
        context = jsonlines[jsonline_idx].get(context_key, {})
        if context:
            encoded_context = \
                context_encoder(context, noise)

        encoded_variant = \
            feature_encoder(
                jsonlines[jsonline_idx].get(variant_key, None), noise)

        if not encoded_context:
            fully_encoded_variant = encoded_variant
        else:
            fully_encoded_variant = {}

            for k in set(encoded_context) | set(encoded_variant):
                fully_encoded_variant[k] = \
                    encoded_context.get(k, 0) + encoded_variant.get(k, 0)

        res[jsonline_idx] = fully_encoded_variant

    return res

@cython.boundscheck(False)
cpdef np.ndarray fill_missing_features(
        np.ndarray encoded_variants, np.ndarray feature_names):

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
                 if name_to_index.get(feature_name, None)])

        # this might be faster if there are less feature names than keys in
        # processed variants
        # filler = \
        #     np.array(
        #         [(idx, processed_variant.get(val, None))
        #          for idx, val in enumerate(feature_names)
        #          if processed_variant.get(val, None)])

        encoded_variants_array[encoded_variant_ixd, filler[:, 0].astype(int)] =\
            filler[:, 1]

    return encoded_variants_array

