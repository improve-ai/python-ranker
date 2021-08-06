import numpy as np


def encoded_variant_to_np(
        encoded_variant: dict, feature_names: np.ndarray or list) -> np.ndarray:
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
             if hash_index_map.get(feature_name, None) is not None])
    if len(filler) > 0:
        result[filler[:, 0].astype(int)] = filler[:, 1]

    return result


def encoded_variants_to_np(
        encoded_variants: np.ndarray, feature_names: np.ndarray) -> np.ndarray:
    """
    Fills missing features in provided encoded variants. Needs model feature
    names. Missing features are filled with np.nan

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

    if len(encoded_variants) == 0:
        return np.full(
            shape=(len(encoded_variants), len(feature_names)),
            fill_value=np.nan)

    encoded_variants_matrix = \
        np.empty((len(encoded_variants), len(feature_names)))

    encoded_variants_matrix[:] = [
        encoded_variant_to_np(
            encoded_variant=encoded_variant, feature_names=feature_names)
        for encoded_variant in encoded_variants]
    return encoded_variants_matrix
