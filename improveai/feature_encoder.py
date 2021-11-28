from collections.abc import Iterable
import math
import numpy as np
import xxhash

import improveai.settings as improve_settings
from improveai.cythonized_feature_encoding import cfeu


xxhash3 = xxhash.xxh3_64_intdigest


class FeatureEncoder:
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

    VARIANT_METADATA_PARAMETER_NAME = 'givens'

    def __init__(self, model_seed):
        if (model_seed < 0):
            raise TypeError(
                "xxhash3_64 takes an unsigned 64-bit integer as seed. "
                "Seed must be greater than or equal to 0.")

        # memoize commonly used seeds
        self.variant_seed = xxhash3("variant", seed=model_seed)
        self.value_seed = xxhash3("$value", seed=self.variant_seed)
        self.givens_seed = xxhash3("givens", seed=model_seed)

    def _check_noise_value(self, noise: float):
        if noise > 1.0 or noise < 0.0:
            raise ValueError(
                'Provided `noise` is out of allowed bounds <0.0, 1.0>')

    def encode_givens(self, givens, noise=0.0, into=None):

        self._check_noise_value(noise=noise)

        if givens is not None and not isinstance(givens, dict):
            raise TypeError(
                "Only dict type is supported for context encoding. {} type was "
                "provided.".format(type(givens)))

        if into is None:
            into = {}

        encode(givens, self.givens_seed, shrink(noise), into)

        # TODO wait until the conversion mechanism is determined
        # self._convert_values_to_float32(into=into)
        return into

    def encode_variant(self, variant, noise=0.0, into=None):

        # TODO maybe this check is only a waste of runtime
        self._check_noise_value(noise=noise)

        if into is None:
            into = {}

        small_noise = shrink(noise)

        if isinstance(variant, dict):
            encode(variant, self.variant_seed, small_noise, into)
        else:
            encode(variant, self.value_seed, small_noise, into)

        # TODO wait until the conversion mechanism is determined
        # self._convert_values_to_float32(into=into)
        return into

    def encode_feature_vector(
            self, variant: dict = None, givens: dict = None,
            extra_features: dict = None, feature_names: list or np.ndarray = None,
            noise: float = 0.0, into: np.ndarray = None):

        if into is None:
            raise ValueError('`into` can`t be None')

        # encode givens
        encoded_givens = \
            self.encode_givens(givens=givens, noise=noise, into=dict())
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
        # TODO once python-SDK is available from pip this should be changed
        if improve_settings.USE_CYTHON_BACKEND:
        # if USE_CYTHON_BACKEND:
            # i7 10th gen time per iter - 7.464-05 [s] / 0.96% of pure python time
            cfeu.encoded_variant_into_np_row(
                encoded_variant=encoded_variant, feature_names=feature_names, into=into)
        else:
            # i7 10th gen time per iter - 7.755e-05 [s]
            hash_index_map = \
                {feature_hash: index for index, feature_hash
                 in enumerate(feature_names)}

            filler = \
                np.array(
                    [(hash_index_map.get(feature_name, None), value)
                     for feature_name, value in encoded_variant.items()
                     if hash_index_map.get(feature_name, None) is not None])

            # sum into with encoded variants treating nans in sums as zeros
            if len(filler) > 0:
                subset_index = filler[:, 0].astype(int)

                into[subset_index] = np.nansum(
                    np.array([into[subset_index], filler[:, 1]]), axis=0)

        # TODO wait until the conversion mechanism is determined
        # np_to_float32_inplace(into)

    def encode_to_np_matrix(
            self, variants: Iterable, multiple_givens: Iterable,
            multiple_extra_features: Iterable, feature_names: Iterable,
            noise: float) -> np.ndarray:
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

        if not isinstance(variants, list):
            variants = list(variants)

        if not isinstance(multiple_givens, list):
            multiple_givens = list(multiple_givens)

        if not isinstance(multiple_extra_features, list):
            multiple_extra_features = list(multiple_extra_features)

        if not isinstance(feature_names, list):
            feature_names = list(feature_names)

        # TODO once python-SDK is available from pip this should be changed
        if improve_settings.USE_CYTHON_BACKEND:
        # if USE_CYTHON_BACKEND:
            # i7 10th gen time per variant - 2.778e-05 [s] |
            # 33% of pure python implementation runtime per iter
            encoded_variants = cfeu.encode_variants_multiple_givens(
                variants=variants, multiple_givens=multiple_givens,
                multiple_extra_features=multiple_extra_features, noise=noise,
                feature_encoder=self.encode_variant,
                givens_encoder=self.encode_givens)
            encoded_variants_array = cfeu.encoded_variants_to_np(
                encoded_variants=encoded_variants, feature_names=feature_names)
        else:
            # i7 10th gen time per variant - 7.995e-05 [s]
            encoded_variants_array = np.full((len(variants), len(feature_names)), np.nan)

            [self.encode_feature_vector(
                variant=variant, givens=givens, extra_features=extra_features,
                feature_names=feature_names, noise=noise, into=into)
             for variant, givens, extra_features, into
             in zip(variants, multiple_givens, multiple_extra_features,
                    encoded_variants_array)]

        # TODO wait until the conversion mechanism is determined
        # return np.float32(encoded_variants_array)

        return encoded_variants_array

    def add_extra_features(self, encoded_variants: list, extra_features: list):
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

    def _convert_values_to_float32(self, into: dict):
        """
        Converts all values in the input dict to float32

        Parameters
        ----------
        into: dict
            converted dict

        Returns
        -------

        """
        for k in into.keys():
            into[k] = np.float32(into[k])


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

        previous_object_ = \
            _get_previous_value(
                feature_name=feature_name, into=features,
                small_noise=small_noise)

        features[feature_name] = sprinkle(object_ + previous_object_, small_noise)

        return

    if isinstance(object_, str):
        hashed = xxhash3(object_, seed=seed)

        feature_name = hash_to_feature_name(seed)

        previous_hashed = \
            _get_previous_value(
                feature_name=feature_name, into=features,
                small_noise=small_noise)

        features[feature_name] = \
            sprinkle(
                (((hashed & 0xffff0000) >> 16) - 0x8000) + previous_hashed,
                small_noise)

        hashed_feature_name = hash_to_feature_name(hashed)

        previous_hashed_for_feature_name = \
            _get_previous_value(
                feature_name=hashed_feature_name, into=features,
                small_noise=small_noise)

        features[hashed_feature_name] = \
            sprinkle(
                ((hashed & 0xffff) - 0x8000) + previous_hashed_for_feature_name,
                small_noise)

        return

    if isinstance(object_, dict):
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
    # None, json null, or unsupported type. Treat as missing feature, return


def _get_previous_value(
        feature_name: str, into: dict, small_noise: float):
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

    previous_sprinkled_object_ = into.get(feature_name, None)

    if previous_sprinkled_object_ is None:
        return 0.0
    else:
        return reverse_sprinkle(previous_sprinkled_object_, small_noise)


def hash_to_feature_name(hash_):
    return '%0*x' % (8, (hash_ >> 32))


def shrink(noise):
    return noise * 2 ** -17


def sprinkle(x, small_noise):
    return (x + small_noise) * (1 + small_noise)


def reverse_sprinkle(sprinkled_x, small_noise):
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


# def add_noise(into, noise):
#     small_noise = shrink(noise)
#     for feature_name, value in into.items():
#         into[feature_name] = sprinkle(value, small_noise)


def np_to_float32_inplace(converted_array: np.ndarray):
    """
    Converts in-place numpy array to float32 dtype

    Parameters
    ----------
    converted_array: np.ndarray
        array to be converted to float32

    Returns
    -------

    """
    f32_input_copy = converted_array.copy().astype(np.float32)

    # change dtype
    converted_array.dtype = np.float32
    # resize back to the original shape
    converted_array.resize(f32_input_copy.shape, refcheck=False)
    # fill with original values
    converted_array[:] = f32_input_copy
