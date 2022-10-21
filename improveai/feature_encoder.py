import warnings
from collections.abc import Iterable
import math
import numpy as np
import xxhash


xxhash3 = xxhash.xxh3_64_intdigest
JSON_SERIALIZABLE_TYPES = {int, float, str, bool, list, tuple, dict, type(None)}
WARNED_ABOUT_ARRAY_ENCODING = False


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
        """
        Init with params

        Parameters
        ----------
        model_seed: int
            model seed to be used during feature encoding
        """

        if (model_seed < 0):
            raise TypeError(
                "xxhash3_64 takes an unsigned 64-bit integer as seed. "
                "Seed must be greater than or equal to 0.")

        # memoize commonly used seeds
        self.variant_seed = xxhash3("variant", seed=model_seed)
        self.value_seed = xxhash3("$value", seed=self.variant_seed)
        self.givens_seed = xxhash3("givens", seed=model_seed)
        self.__user_warned_about_array_encoding = False

    def _check_noise_value(self, noise: float):
        """
        Check whether noise value is valid; Raises if `noise` is invalid

        Parameters
        ----------
        noise: float
            checked value of no ise

        Returns
        -------
        None
            None
        """

        if noise > 1.0 or noise < 0.0:
            raise ValueError(
                'Provided `noise` is out of allowed bounds <0.0, 1.0>')

    def encode_givens(
            self, givens: dict or None, noise: float = 0.0, into: dict = None) -> dict:
        """
        Encodes provided givens of arbitrary complexity to a flat feature dict;
        Givens must be JSON encodable

        Parameters
        ----------
        givens: dict or None
            givens to be encoded
        noise: float
            value within 0 - 1 range which will be 'shrunk' and added to feature value
        into: dict
            a dict into which new features will be inserted; Can be None

        Returns
        -------
        dict
            A dict with results of givens encoding

        """

        self._check_noise_value(noise=noise)

        if givens is not None and not isinstance(givens, dict):
            raise TypeError(
                "Only dict type is supported for context encoding. {} type was "
                "provided.".format(type(givens)))

        if into is None:
            into = {}

        encode(givens, self.givens_seed, shrink(noise), into)
        return into

    def encode_variant(
            self, variant: object, noise: float = 0.0, into: dict = None) -> dict:
        """
        Encodes provided variant of arbitrary complexity to a flat feature dict;
        Variant must be JSON encodable

        Parameters
        ----------
        variant: object
            variant to be encoded
        noise: float
            value within 0 - 1 range which will be 'shrunk' and added to feature value
        into: dict
            a dict into which new features will be inserted; Can be None

        Returns
        -------
        dict
            A dict with results of variant encoding

        """

        self._check_noise_value(noise=noise)

        if into is None:
            into = {}

        small_noise = shrink(noise)

        if isinstance(variant, dict):
            encode(variant, self.variant_seed, small_noise, into)
        else:
            encode(variant, self.value_seed, small_noise, into)

        return into

    def encode_feature_vector(
            self, variant: object = None, givens: dict = None,
            extra_features: dict = None, feature_names: list or np.ndarray = None,
            noise: float = 0.0, into: np.ndarray = None):
        """
        Fully encodes provided variant and givens into a np.ndarray provided as `into` parameter.
        `into` must not be None

        Parameters
        ----------
        variant: object
            a JSON encodable object to be encoded to flat features' dict
        givens: dict
            a dict with givens to be enncoded to flat features' dict (all entries must be JSON encodable)
        extra_features: dict
            features to be added to encoded variant with givens
        feature_names: list or np.ndarray
            list of model's feature names (only overlapping features will be selected for resulting vector)
        noise: float
            value within 0 - 1 range which will be 'shrunk' and added to feature value
        into: np.ndarray
            an array into which feature values will be added / inserted

        Returns
        -------
        None
            None

        """

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
        hash_index_map = \
            {feature_hash: index for index, feature_hash in enumerate(feature_names)}

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

        encoded_variants_array = np.full((len(variants), len(feature_names)), np.nan)

        [self.encode_feature_vector(
            variant=variant, givens=givens, extra_features=extra_features,
            feature_names=feature_names, noise=noise, into=into)
         for variant, givens, extra_features, into
         in zip(variants, multiple_givens, multiple_extra_features,
                encoded_variants_array)]
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


def _is_object_json_serializable(object_) -> bool:
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
    # try - except was slower
    return type(object_) in JSON_SERIALIZABLE_TYPES


def _has_top_level_string_keys(checked_dict) -> bool:
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

    return all(isinstance(k, str) for k in checked_dict.keys())


def warn_about_array_encoding(object_):
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
    global WARNED_ABOUT_ARRAY_ENCODING

    if not WARNED_ABOUT_ARRAY_ENCODING and \
            (isinstance(object_, list) or isinstance(object_, tuple) or isinstance(object_, np.ndarray)):
        warnings.warn('Array encoding may change in the near future')
        WARNED_ABOUT_ARRAY_ENCODING = True


def encode(object_, seed, small_noise, features):
    """
    Encodes a JSON serializable object to  a flat key - value pair(s) structure / dict
    (sometimes a single `object_` will result in 2 features, e.g. strings, lists, etc...).
    Rules of encoding go as follows:

    - None, json null, {}, [], nan, are treated as missing feature_names and ignored. NaN is technically not allowed in JSON but check anyway.

    - boolean true and false are encoded as 1.0 and 0.0 respectively.

    - strings are encoded into an additional one-hot feature.

    - numbers are encoded as-is with up to about 5 decimal places of precision

    - a small amount of noise is incorporated to avoid overfitting

    - feature names are 8 hexadecimal characters


    Parameters
    ----------
    object_: object
        a JSON serializable object to be encoded to a flat key-value structure
    seed: int
        seed for xxhash3 to generate feature name
    small_noise: float
        a shrunk noise to be added to value of encoded feature
    features: dict
        a flat dict of {<feature name>: <feature value>, ...} pairs

    Returns
    -------
    None
        None

    """

    assert _is_object_json_serializable(object_)
    warn_about_array_encoding(object_=object_)

    if isinstance(object_, (int, float)):  # bool is an instanceof int
        if math.isnan(object_):  # nan is treated as missing feature, return
            return

        feature_name = hash_to_feature_name(seed)

        previous_object_ = \
            _get_previous_value(feature_name=feature_name, into=features, small_noise=small_noise)

        features[feature_name] = sprinkle(object_ + previous_object_, small_noise)

        return

    if isinstance(object_, str):
        hashed = xxhash3(object_, seed=seed)

        feature_name = hash_to_feature_name(seed)

        previous_hashed = \
            _get_previous_value(feature_name=feature_name, into=features, small_noise=small_noise)

        features[feature_name] = \
            sprinkle((((hashed & 0xffff0000) >> 16) - 0x8000) + previous_hashed, small_noise)

        hashed_feature_name = hash_to_feature_name(hashed)

        previous_hashed_for_feature_name = \
            _get_previous_value(feature_name=hashed_feature_name, into=features, small_noise=small_noise)

        features[hashed_feature_name] = \
            sprinkle(((hashed & 0xffff) - 0x8000) + previous_hashed_for_feature_name, small_noise)

        return

    if isinstance(object_, dict):
        assert _has_top_level_string_keys(object_)
        for key, value in object_.items():
            encode(value, xxhash3(key, seed=seed), small_noise, features)
        return

    if isinstance(object_, (list, tuple)):
        for index, item in enumerate(object_):
            encode(item, xxhash3(index.to_bytes(8, byteorder='big'), seed=seed), small_noise, features)
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

    if feature_name not in into:
        return 0.0
    else:
        return reverse_sprinkle(into[feature_name], small_noise)


def hash_to_feature_name(hash_: int):
    """
    Converts a hash to string which will become a feature name

    Parameters
    ----------
    hash_: int
        an integer output from xxhash3

    Returns
    -------
    str
        a string representation of a hex feature name created from int

    """
    return '%0*x' % (8, (hash_ >> 32))


def shrink(noise):
    """
    Shrinks noise value by a hardcoded factor 2 ** -17

    Parameters
    ----------
    noise: float
        value within 0 - 1 range which will be 'shrunk' and added to feature value


    Returns
    -------
    float
        a shrunk noise

    """
    return noise * 2 ** -17


def sprinkle(x, small_noise):
    """
    Slightly modified input valuse using `small_noise`: (x + small_noise) * (1 + small_noise)

    Parameters
    ----------
    x: float
        a number to be 'modified'
    small_noise: float
        a small number with which `x` will be modified

    Returns
    -------
    float
        x modified with `small_noise`

    """
    return (x + small_noise) * (1 + small_noise)

def get_noise_shift_scale(noise):
    assert noise >= 0.0 and noise < 1.0
    # x + noise * 2 ** -142 will round to x for most values of x. Used to create
    # distinct values when x is 0.0 since x * (1 + noise * 2 ** -17) will be zero
    return (noise * 2 ** -142, 1 + noise * 2 ** -17)

def v8_sprinkle(x, noise_shift, noise_scale):
    # x + noise_offset will round to x for most values of x
    # allows different values when x == 0.0
    return (x + noise_shift) * noise_scale


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
