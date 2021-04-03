from collections.abc import Iterable
from Cython.Build import cythonize
import math
import numpy as np
import os
import pyximport
from setuptools import Extension
# import struct
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
        '{}{}encoders_cython_utils/cythonized_feature_encoding.pyx'\
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

    import feature_encoders.encoders_cython_utils.cythonized_feature_encoding \
        as cfe


xxhash3 = xxhash.xxh3_64_intdigest


class FeatureEncoder:
    def __init__(self, model_seed):
        if (model_seed < 0):
            raise TypeError(
                "xxhash3_64 takes an unsigned 64-bit integer as seed. Seed must be greater than or equal to 0.")

        # memoize commonly used seeds
        self.variant_seed = xxhash3("variant", seed=model_seed)
        self.value_seed = xxhash3("$value", seed=self.variant_seed)
        self.context_seed = xxhash3("context", seed=model_seed)

    def encode_context(self, context, noise):
        features = {}

        encode(context, self.context_seed, shrink(noise), features)

        return features

    def encode_variant(self, variant, noise):
        features = {}

        small_noise = shrink(noise)

        if isinstance(variant, dict):
            encode(variant, self.variant_seed, small_noise, features)
        else:
            encode(variant, self.value_seed, small_noise, features)

        return features

    def encode_variants(
            self, variants: Iterable, contexts: Iterable or dict or None,
            noise: float, verbose: bool = False) -> Iterable:
        """
        Cythonized loop over provided variants and context(s).
        Returns array of encoded dicts.

        Parameters
        ----------
        variants: Iterable
            collection of input variants to be encoded
        contexts: Iterable or dict or None
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
            used_variants = np.array(variants)

        encoder_method_kwargs = {
            'variants': used_variants,
            'noise': noise}

        if isinstance(contexts, dict) or contexts is None or contexts is {}:
            # process with single context
            encoder_method = self._encode_variants_single_context
            encoder_method_kwargs['context'] = contexts

        elif not isinstance(contexts, str) and isinstance(contexts, Iterable):
            used_contexts = contexts
            if not isinstance(variants, np.ndarray):
                if verbose:
                    print(
                        'Provided `contexts` is Iterable and not of np.ndarray '
                        'type - attempting to convert')
                used_contexts = np.array(contexts)

            encoder_method = self._encode_variants_multiple_contexts
            encoder_method_kwargs['contexts'] = used_contexts

        else:
            raise TypeError(
                'Unsupported contexts` type: {}'.format(type(contexts)))

        return encoder_method(**encoder_method_kwargs)

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

        return cfe.encode_jsonlines(
            jsonlines=jsonlines, noise=noise, variant_key=variant_key,
            context_key=context_key, feature_encoder=self.encode_variant,
            context_encoder=self.encode_context)

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

        return cfe.encode_variants_single_context(
            variants=variants, context=context, noise=noise,
            feature_encoder=self.encode_variant,
            context_encoder=self.encode_context)

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

        return cfe.encode_variants_multiple_contexts(
            variants=variants, contexts=contexts, noise=noise,
            feature_encoder=self.encode_variant,
            context_encoder=self.encode_context)

    def fill_missing_features(
            self,encoded_variants: np.ndarray,
            feature_names: np.ndarray) -> np.ndarray:
        return cfe.fill_missing_features(
            encoded_variants=encoded_variants, feature_names=feature_names)

    def fully_encode_single_variant(
            self, variant: dict, context: dict, noise: float,
            verbose: bool = False) -> dict:

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
        features[hashed_feature_name] = features.get(hashed_feature_name,
                                                     0.0) + sprinkle(
            (hashed & 0xffff) - 0x8000, small_noise)
        return

    if isinstance(object_, dict):
        for key, value in object_.items():

            # # TODO make sure this checks out
            # # TODO numeric keys are illegal in json and hence untrackable
            # key = raw_key
            # if isinstance(raw_key, int):
            #     key = raw_key.to_bytes(8, byteorder='big')
            # elif isinstance(raw_key, float):
            #     key = struct.pack(">f", raw_key)

            encode(value, xxhash3(key, seed=seed), small_noise, features)
        return

    if isinstance(object_, (list, tuple)):
        for index, item in enumerate(object_):
            encode(item, xxhash3(index.to_bytes(8, byteorder='big'), seed=seed),
                   small_noise, features)
        return

    # None, json null, or unsupported type. Treat as missing feature, return


def hash_to_feature_name(hash_):
    return hex(hash_ >> 32)[2:]  # chop off '0x'. Could be further optimized


def shrink(noise):
    return noise * 2 ** -17


def sprinkle(x, small_noise):
    return (x + small_noise) * (1 + small_noise)


if __name__ == '__main__':
    from coolname import generate
    import json
    import xgboost as xgb
    import time

    encoder_seed = 1

    # load dummy model
    b = xgb.Booster()
    b.load_model('../artifacts/models/dummy_v6.xgb')

    user_defined_metadata = json.loads(b.attr('user_defined_metadata'))['json']

    # extract features map (?)
    feature_names = np.array(user_defined_metadata['feature_names'])
    model_seed = user_defined_metadata['model_seed']
    print('model_seed')
    print(model_seed)

    # test_jsonlines = []
    with open('../artifacts/data/dummy/test_jsonlines.json') as tjl:
        test_jsonlines = np.array([json.loads(jl) for jl in tjl])

    fe = FeatureEncoder(model_seed=model_seed)

    np.random.seed(6)
    noise = np.random.rand()
    # value = {"$value": 1.7976931348623157e+308}
    # TODO dict with integer  / float keys will fail
    # value = {"$value": {'0': 0, '1': {'0': 0, '1': 1, '2': 2}, '2': {'0': {'0': 0, '1': 1, '2': 2}, '1': {'0': 0, '1': 1, '2': 2}}}}
    # value = ['foo', 'bar']

    # value = {"\0\0\0\0\0\0\0\0": "foo", "\0\0\0\0\0\0\0\1": "bar"}
    value = {'a': 1, 'b': 1, 'c': 1, 'd': 1}

    from dask.distributed import Client, LocalCluster
    import dask.bag as db

    lc = LocalCluster(n_workers=7, threads_per_worker=2)
    c = Client(lc)

    attempts = int(5e9)
    parts = 1000  # int(attempts / 10000000)

    context = dict(
        ('_'.join(generate(2)), generate(4) if idx != 3 else [0, 1, 2, 3])
        for idx in range(4))

    encoded_context = fe.encode_context(context, noise=noise)

    def check_if_keys_overlap(value):
        v = fe.encode_variant(value, noise=noise)

        if set(v.keys()).intersection(set(encoded_context.keys())):
            return value
            # print(value)
        return None


    colliders = db.range(attempts, npartitions=parts).map(
        lambda x:
        dict(('_'.join(generate(2)),
              ' '.join(generate(np.random.randint(2, 5))))
             for _ in range(np.random.randint(2, 5))))\
        .map(check_if_keys_overlap)\
        .filter(lambda x: x is not None)\
        .compute()

    print(colliders)

    # for _ in range(int(1e6)):
    #     keys_count = np.random.randint(2, 5)
    #     value = \
    #         dict(('_'.join(generate(2)), ' '.join(generate(np.random.randint(2, 5))))
    #              for _ in range(np.random.randint(2, 5)))

    input('check after brute force')
    # print(v)
    # print(c)
    # input('sanity check')
    #
    # print(len(test_jsonlines))

    # time vectorization

    total_times = []
    times_per_iter = []

    from tqdm import tqdm

    single_context = \
        {"device_manufacturer": "Apple", "version_name": "4.3", "os_name": "ios",
         "day": 0, "language": "English", "page": 0, "os_version": "12.1.2",
         "country": "United States", "carrier": "AT&T",
         "device_model": "iPhone 7 Plus"}

    test_variants = np.array([jl['variant'] for jl in test_jsonlines])[:1000]
    test_contexts = \
        np.array([jl.get('context', None) for jl in test_jsonlines])[:1000]

    test_contexts = \
        np.array(
            [test_jsonlines[0]['context'] for _ in range(len(test_contexts))])

    encoded_context = fe.encode_context(single_context, np.random.rand())
    q = ['bac' for el in range(1000)]

    vhash = np.vectorize(xxhash3)

    def encode_variants_single_context(
            variants, context, noise, feature_encoder, context_encoder):

        variants_count = len(variants)
        res = np.empty((variants_count, ), dtype=object)

        encoded_context = context_encoder(context, noise)

        for variant_idx in range(variants):

            encoded_variant = feature_encoder(variant_idx, noise)

            fully_encoded_variant = {}

            for k in set(encoded_context) | set(encoded_variant):
                fully_encoded_variant[k] = \
                    encoded_context.get(k, 0) + encoded_variant.get(k, 0)

            # fully_encoded_variant = \
            #     {k: encoded_context.get(k, 0) + encoded_variant.get(k, 0)
            #       for k in set(encoded_context) | set(encoded_variant)}

            res[variant_idx] = fully_encoded_variant

        return res

    def encode_single_variant(
            variant, encoded_context, noise, feature_encoder):

        encoded_variant = feature_encoder(variant, noise)

        # for k in set(encoded_context) | set(encoded_variant):
        #     fully_encoded_variant[k] = \
        #         encoded_context.get(k, 0) + encoded_variant.get(k, 0)

        fully_encoded_variant = \
            {k: encoded_context.get(k, 0) + encoded_variant.get(k, 0)
             for k in set(encoded_context) | set(encoded_variant)}
        return fully_encoded_variant

    v_encode_single_variant = \
        np.vectorize(
            encode_single_variant,
            excluded=['encoded_context', 'noise', 'feature_encoder'],
            cache=True)

    for _ in tqdm(range(1000)):

        st = time.time()

        # for jl in test_jsonlines:
        #     fe.encode_variant(variant=jl['variant'], noise=np.random.rand())

        # noise = np.random.rand()
        # encoded_context = fe.encode_context(single_context, noise)
        # res = v_encode_single_variant(
        #     variant=test_variants, encoded_context=encoded_context, noise=noise,
        #     feature_encoder=fe.encode_variant)

        # res = \
        #     [{k: encoded_context.get(k, 0) + ev.get(k, 0)
        #       for k in set(encoded_context) | set(ev)} for ev in
        #      [fe.encode_variant(v, np.random.rand()) for v in test_variants]]

        # res = cfe.encode_variants_single_context(
        #     variants=test_variants,
        #     context=single_context, feature_encoder=fe.encode_variant,
        #     context_encoder=fe.encode_context)

        # kwgs = {
        #     'jsonlines': test_jsonlines[:1000],
        #     'variant_key': 'variant', 'context_key': 'context',
        #     'feature_encoder': fe.encode_variant,
        #     'context_encoder': fe.encode_context}
        #

        res = fe.encode_variants(
            variants=test_variants, contexts=single_context,
            noise=np.random.rand())

        # et1 = time.time()
        # input('sanity check')

        filled_res = \
            cfe.fill_missing_features(
                encoded_variants=res, feature_names=feature_names)

        # res = fe.encode_jsonlines(
        #     jsonlines=test_jsonlines[:1000], noise=np.random.rand(),
        #     variant_key='variant', context_key='context')

        # res = [xxhash3('abc') for _ in range(1000)]

        # res = cfe.encode_jsonlines(**kwgs)
        # res = cfe.encode_jsonlines(
        #     jsonlines=test_jsonlines[:1000],
        #     variant_key='variant', context_key='context',
        #     feature_encoder=fe.encode_variant, context_encoder=fe.encode_context)

        # res = cfe.encode_variants_multiple_contexts(
        #         variants=test_variants,
        #         contexts=test_contexts, feature_encoder=fe.encode_variant,
        #         context_encoder=fe.encode_context)

        # res = vhash(q)

        et = time.time()
        total_times.append((et - st))
        times_per_iter.append((et - st) / len(test_variants))
    # {'ad8ff43e': -12644.062851908973, 'a40e6b86': 24087.119748223067, 'b802a909': 14665.072908811017, '87ffca60': 3234.0160820949077, 'b96a6ac7': -14523.072192946465, 'c646e968': -27187.13514925237, '1c1de8c1': 29437.052969682885, '343947a': 30272.150495597674, 'e4e1d4ac': -5350.026591383446, '342fac51': 12.000023391737646, '1d142dac': 4059.020183401949, 'c3abcfc3': 2.0000053980957935, '865daab3': 31928.158728039445, '4fce86a9': 21438.106579299005, '64fe5687': 4.971305975735534e-06, '669031ec': 4.971305975735534e-06, '1c4abbd7': -23067.11466757357, '291ec7b3': 22238.040016060117, 'a133cc28': 3070.015266804781, '136ae93f': -20283.100827526538, '3ff9d0fa': -2451.0121796390677, '7b042c96': 12312.061211386206}
    # print(context)
    # print(res[:10])
    # print(filled_res[0])
    print('Total compute time: {}'.format(np.mean(total_times)))
    print('Time per iteration: {}'.format(np.mean(times_per_iter)))

# Cythonized
# 100%|██████████| 1000/1000 [00:08<00:00, 115.76it/s]
# Total compute time: 0.008623963594436646
# Time per iteration: 1.1718934086746359e-06

# Plain list comp
# 100%|██████████| 1000/1000 [00:11<00:00, 88.26it/s]
# Total compute time: 0.011307515859603882
# Time per iteration: 1.5365560347335075e-06

# without filling single context
# Total compute time: 0.00840233302116394
# Time per iteration: 8.40233302116394e-06

# with filling multi context
# Total compute time: 0.015273385286331176
# Time per iteration: 1.5273385286331178e-05


# without filling multi contexts
# Total compute time: 0.02827765154838562
# Time per iteration: 2.827765154838562e-05

# with filling multi contexts
# Total compute time: 0.035208083868026735
# Time per iteration: 3.520808386802673e-05
#
# Process finished with exit code 0

