from collections.abc import Iterable
from Cython.Build import cythonize
import math
import numpy as np
import os
import pyximport
from setuptools import Extension
import xxhash


try:
    # This is done for backward compatibilty
    from coremltools.models.utils import macos_version
except Exception as exc:
    from coremltools.models.utils import _macos_version as macos_version


if not macos_version():

    rel_pth_prfx = \
        os.sep.join(str(os.path.relpath(__file__)).split(os.sep)[:-1])

    pth_str = \
        '{}{}encoder_cython_utils/cythonized_feature_encoding.pyx'\
        .format(
            os.sep.join(str(os.path.relpath(__file__)).split(os.sep)[:-1]),
            '' if not rel_pth_prfx else os.sep)

    fast_feat_enc_ext = \
        Extension(
            'cythonized_feature_encoding',
            sources=[pth_str],
            define_macros=[
                ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            include_dirs=[np.get_include()])

    pyximport.install(
        setup_args={
            'install_requires': ["numpy"],
            'ext_modules': cythonize(
                fast_feat_enc_ext,
                language_level="3")})

    import improveai.encoder_cython_utils.cythonized_feature_encoding as cfe


xxhash3 = xxhash.xxh3_64_intdigest


class FeatureEncoder:

    def __init__(self, model_seed):
        if (model_seed < 0):
            raise TypeError(
                "xxhash3_64 takes an unsigned 64-bit integer as seed. "
                "Seed must be greater than or equal to 0.")

        # memoize commonly used seeds
        self.variant_seed = xxhash3("variant", seed=model_seed)
        self.value_seed = xxhash3("$value", seed=self.variant_seed)
        self.context_seed = xxhash3("context", seed=model_seed)

    def _check_noise_value(self, noise: float):
        if noise > 1.0 or noise < 0.0:
            raise ValueError(
                'Provided `noise` is out of allowed bounds <0.0, 1.0>')

    def encode_context(self, context, noise):

        self._check_noise_value(noise=noise)

        features = {}

        if context is not None and not isinstance(context, dict):
            raise TypeError(
                "Only dict type is supported for context encoding. {} type was "
                "provided.".format(type(context)))

        encode(context, self.context_seed, shrink(noise), features)

        return features

    def encode_variant(self, variant, noise):

        self._check_noise_value(noise=noise)

        features = {}

        small_noise = shrink(noise)

        if isinstance(variant, dict):
            encode(variant, self.variant_seed, small_noise, features)
        else:
            encode(variant, self.value_seed, small_noise, features)

        return features

    def encode_variants(
            self, variants: Iterable, context: Iterable or dict or None,
            noise: float, verbose: bool = False) -> Iterable:
        """
        Cythonized loop over provided variants and context(s).
        Returns array of encoded dicts.
        Parameters
        ----------
        variants: Iterable
            collection of input variants to be encoded
        context: Iterable or dict or None
            collection or a single item of input contexts to be encoded with
            variants
        noise: float
            noise param from 0-1 uniform distribution
        verbose: bool
            debug prints toggle
        Returns
        -------
        np.ndarray
            array of encoded dicts
        """

        used_variants = variants
        if not isinstance(variants, np.ndarray):
            if verbose:
                print(
                    'Provided `variants` are not of np.ndarray type - '
                    'attempting to convert')
            used_variants = np.empty(len(variants), dtype=object)
            used_variants[:] = [el for el in variants]
            # used_variants = np.array(variants)

        encoder_method_kwargs = {
            'variants': used_variants,
            'noise': noise}

        if isinstance(context, dict) or context is None or context is {}:
            # process with single context
            encoder_method = self._encode_variants_single_context
            encoder_method_kwargs['context'] = context

        elif not isinstance(context, str) and isinstance(context, Iterable):
            used_contexts = context
            if not isinstance(variants, np.ndarray):
                if verbose:
                    print(
                        'Provided `contexts` is Iterable and not of np.ndarray '
                        'type - attempting to convert')
                used_contexts = np.array(context)

            encoder_method = self._encode_variants_multiple_contexts
            encoder_method_kwargs['contexts'] = used_contexts

        else:
            raise TypeError(
                'Unsupported contexts` type: {}'.format(type(context)))

        encoded_variants = encoder_method(**encoder_method_kwargs)

        return encoded_variants

    def encode_jsonlines(
            self, jsonlines: Iterable, noise: float, variant_key: str,
            context_key: str, verbose: bool = False,  **kwargs) -> np.ndarray:
        """
        Wrapper around cythonized jsonlines encoder
        Parameters
        ----------
        jsonlines: Iterable
            collection of jsonlines to be encoded
        variant_key: str
            key with which `variant` can be extracted from a jsonline
        context_key: str
            key with which `context` can be extracted from a jsonline
        verbose: bool
            debug prints toggle
        kwargs
        Returns
        -------
        np.ndarray
            array of encoded variants
        """

        if not variant_key or not context_key:
            raise ValueError(
                '`variant_key` and `context_key must both be provided and not '
                'empty / None')

        if not isinstance(jsonlines, np.ndarray):
            if verbose:
                print(
                    'Provided `jsonlines` are not of np.ndarray type - '
                    'attempting to convert')
            jsonlines = np.array(jsonlines)

        if not macos_version():
            return cfe.encode_jsonlines(
                jsonlines=jsonlines, noise=noise, variant_key=variant_key,
                context_key=context_key, feature_encoder=self.encode_variant,
                context_encoder=self.encode_context)
        else:
            return self._encode_jsonlines_numpy(
                jsonlines=jsonlines, noise=noise, variant_key=variant_key,
                context_key=context_key)

    def _encode_jsonlines_numpy(
            self, jsonlines: np.ndarray, noise: float, variant_key: str,
            context_key: str) -> np.ndarray:
        """
        Numpy based jsonlines batch encoder. Slower than cython but used for now
        due to macOS - numpy - cython problems

        Parameters
        ----------
        jsonlines: np.ndarray
            array of jsonlines to be encoded
        noise: float
            noise parameter for encoding
        variant_key: str
            key to extract variant from jsonline / record
        context_key: str
            key to extract context form jsonline / record

        Returns
        -------
        np.ndarray
            array of dicts with encoded records

        """

        encoded_variants = np.empty(len(jsonlines), dtype=object)
        encoded_variants[:] = [
            self.fully_encode_single_variant(
                variant=record.get(variant_key, None),
                context=record.get(context_key, None), noise=noise)
            for record in jsonlines]

        return encoded_variants

    def _encode_variants_single_context(
            self, variants: Iterable, context: dict, noise: float,
            **kwargs) -> np.ndarray:
        """
        Wrapper around cythonized single-context variant encoder
        Parameters
        ----------
        variants: Iterable
            collection of variants to be encoded
        context: dict
            context to be encoded
        kwargs: dict
            kwargs
        Returns
        -------
        np.ndarray
            array of encoded variants
        """

        if not macos_version():
            return cfe.encode_variants_single_context(
                variants=variants, context=context, noise=noise,
                feature_encoder=self.encode_variant,
                context_encoder=self.encode_context)
        else:
            return self._encode_variants_single_context_numpy(
                variants=variants, context=context, noise=noise)

    def _encode_variants_single_context_numpy(
            self, variants: np.ndarray, context: dict, noise: float):
        """
        Encodes variants with single context using numpy. Slower than cython
        implementation; Needed for now for macOS compat.

        Parameters
        ----------
        variants: np.ndarray
            array of variants to be encoded
        context: dict
            context dict
        noise: float
            noise parameter for the encoding

        Returns
        -------
        np.ndarray
            array of encoded variants

        """

        encoded_variants = np.empty(len(variants), dtype=object)
        encoded_variants[:] = [
            self.fully_encode_single_variant(
                variant=variant, context=context, noise=noise)
            for variant in variants]

        return encoded_variants

    def _encode_variants_multiple_contexts(
            self, variants: Iterable, contexts: Iterable, noise: float,
            **kwargs) -> np.ndarray:

        """
        Wrapper around cythonized multi-context variant encoder
        Parameters
        ----------
        variants: Iterable
            collection of variants to encode
        contexts: Iterable
            collection of contexts to encode
        kwargs: dict
            kwargs
        Returns
        -------
        np.ndarray
            array of encoded variants
        """

        if not macos_version():
            return cfe.encode_variants_multiple_contexts(
                variants=variants, contexts=contexts, noise=noise,
                feature_encoder=self.encode_variant,
                context_encoder=self.encode_context)
        else:
            return self._encode_variants_multiple_context_numpy(
                variants=variants, contexts=contexts, noise=noise)

    def _encode_variants_multiple_context_numpy(
            self, variants: np.ndarray, contexts: dict,
            noise: float) -> np.ndarray:
        """
        Encodes variants with multiple contexts using numpy. Slower than cython
        implementation; Needed for now for macOS compat.

        Parameters
        ----------
        variants: np.ndarray
            array of variants to be encoded
        contexts: np.ndarray
            array of context dicts
        noise: float
            noise parameter for the encoding

        Returns
        -------
        np.ndarray
            array of encoded variants

        """
        encoded_variants = np.empty(len(variants), dtype=object)
        encoded_variants[:] = [
            self.fully_encode_single_variant(
                variant=variant, context=context, noise=noise)
            for variant, context in zip(variants, contexts)]

        return encoded_variants

    def fill_missing_features(
            self, encoded_variants: np.ndarray,
            feature_names: np.ndarray) -> np.ndarray:
        """
        Fills missing features in provided encoded variants. Needs model feature
        names. Missings are filled with np.nan

        Parameters
        ----------
        encoded_variants: np.ndarray
            array of encoded vairants
        feature_names: np.ndarray
            array of feature names from model which will be used for predictions

        Returns
        -------
        np.ndarray
            array of (num. variants, num. features) shape

        """

        if not macos_version():
            return cfe.fill_missing_features(
                encoded_variants=encoded_variants, feature_names=feature_names)
        else:
            return self._fill_missing_features_numpy(
                encoded_variants=encoded_variants, feature_names=feature_names)

    def _fill_missing_features_numpy(
            self, encoded_variants: np.ndarray,
            feature_names: np.ndarray) -> np.ndarray:
        """
        Plain numpy missing features filler. Used for macOS compat

        Parameters
        ----------
        encoded_variants: np.ndarray
            array of encoded vairants
        feature_names: np.ndarray
            array of feature names from model which will be used for predictions

        Returns
        -------
        np.ndarray
            array of (num. variants, num. features) shape

        """
        no_missing_features_variants = \
            np.empty((len(encoded_variants), len(feature_names)))
        no_missing_features_variants[:] = [
            self.fill_missing_features_single_variant(
                encoded_variant=encoded_variant, feature_names=feature_names)
            for encoded_variant in encoded_variants]
        return no_missing_features_variants

    def fully_encode_single_variant(
            self, variant: dict, context: dict, noise: float,
            verbose: bool = False) -> dict:
        """
        Encodes variant and context and 'concatenates' results of encoding
        (sums values in case of key collision)

        Parameters
        ----------
        variant: dict
            variant to be encoded
        context: dict
            context to be encoded
        noise: float
            noise parameter for the encoding
        verbose: bool
            logging parameter

        Returns
        -------
        dict
            result of context and variant encoding

        """

        # encode context
        encoded_context = {}
        if context:
            encoded_context = self.encode_context(context=context, noise=noise)

        # encode variant
        encoded_variant = self.encode_variant(variant=variant, noise=noise)

        # add
        return \
            {k: encoded_context.get(k, 0) + encoded_variant.get(k, 0)
             for k in set(encoded_context) | set(encoded_variant)}

    def fill_missing_features_single_variant(
            self, encoded_variant: dict,
            feature_names: np.ndarray) -> np.ndarray:
        """
        Fills missing features in a single variant

        Parameters
        ----------
        encoded_variant: dict
            fully encoded single variant
        feature_names: np.ndarray
            array of feature names from model which will be used for predictions

        Returns
        -------
        np.ndarray
            single row of a shape (1, num. features) which contains fully
            encoded variant

        """

        result = np.empty(shape=(len(feature_names), ))
        result[:] = np.nan
        hash_index_map = \
            {feature_hash: index for index, feature_hash
             in enumerate(feature_names)}

        filler = \
            np.array(
                [(hash_index_map.get(feature_name, None), value)
                 for feature_name, value in encoded_variant.items()
                 if hash_index_map.get(feature_name, None)])
        if len(filler) > 0:
            result[filler[:, 0].astype(int)] = filler[:, 1]

        return result


# - None, json null, {}, [], nan, are treated as missing feature_names and ignored.
# NaN is technically not allowed in JSON but check anyway.
# - boolean true and false are encoded as 1.0 and 0.0 respectively.
# - strings are encoded into an additional one-hot feature.
# - numbers are encoded as-is with up to about 5 decimal places of precision
# - a small amount of noise is incorporated to avoid overfitting
# - feature names are 8 hexidecimal characters
def encode(object_, seed, small_noise, features):
    if isinstance(object_, (int, float)):  # bool is an instanceof int

        if math.isnan(object_):  # nan is treated as missing feature, return
            return

        feature_name = hash_to_feature_name(seed)

        features[feature_name] = features.get(feature_name, 0.0) + sprinkle(
            object_, small_noise)
        return

    if isinstance(object_, str):
        hashed = xxhash3(object_, seed=seed)

        feature_name = hash_to_feature_name(seed)

        features[feature_name] = features.get(feature_name, 0.0) + sprinkle(
            ((hashed & 0xffff0000) >> 16) - 0x8000, small_noise)

        hashed_feature_name = hash_to_feature_name(hashed)

        features[hashed_feature_name] = \
            features.get(hashed_feature_name, 0.0) + sprinkle(
            (hashed & 0xffff) - 0x8000, small_noise)
        return

    if isinstance(object_, dict):
        for key, value in object_.items():
            encode(value, xxhash3(key, seed=seed), small_noise, features)
        return

    if isinstance(object_, (list, tuple)):

        for index, item in enumerate(object_):
            encode(
                item, xxhash3(index.to_bytes(8, byteorder='big'), seed=seed),
                small_noise, features)
        return

    # None, json null, or unsupported type. Treat as missing feature, return


def hash_to_feature_name(hash_):
    return hex(hash_ >> 32)[2:]  # chop off '0x'. Could be further optimized


def shrink(noise):
    return noise * 2 ** -17


def sprinkle(x, small_noise):
    return (x + small_noise) * (1 + small_noise)

