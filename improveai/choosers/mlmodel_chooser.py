import coremltools as ct
import json
import numpy as np
from time import time
from typing import Dict, List


from improveai.feature_encoder import FeatureEncoder
from improveai.cythonized_feature_encoding import cfe
from improveai.choosers.basic_choosers import BasicChooser
from improveai.settings import USE_CYTHON_BACKEND
from improveai.utils.general_purpose_tools import constant
from improveai.utils.choosers_feature_encoding_tools import \
    encoded_variant_to_np

FastFeatureEncoder = cfe.FeatureEncoder


class MLModelChooser(BasicChooser):

    @property
    def model(self) -> ct.models.MLModel:
        return self._model

    @model.setter
    def model(self, new_val: ct.models.MLModel):
        self._model = new_val

    @property
    def model_metadata(self) -> Dict[str, object]:
        return self._model_metadata

    @model_metadata.setter
    def model_metadata(self, new_val: Dict[str, object]):
        self._model_metadata = new_val

    @property
    def feature_encoder(self) -> FeatureEncoder or FastFeatureEncoder:
        return self._feature_encoder

    @feature_encoder.setter
    def feature_encoder(self, new_val: FeatureEncoder or FastFeatureEncoder):
        self._feature_encoder = new_val

    @property
    def model_metadata_key(self) -> str:
        return self._model_metadata_key

    @model_metadata_key.setter
    def model_metadata_key(self, new_val: str):
        self._model_metadata_key = new_val

    @property
    def model_seed_key(self) -> str:
        return self._model_seed_key

    @model_seed_key.setter
    def model_seed_key(self, new_val: str):
        self._model_seed_key = new_val

    @property
    def model_seed(self):
        return self._model_seed

    @model_seed.setter
    def model_seed(self, value):
        self._model_seed = value

    @property
    def model_name_key(self):
        return self._model_name_key

    @model_name_key.setter
    def model_name_key(self, value):
        self._model_name_key = value

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, value):
        self._model_name = value

    @property
    def model_feature_names_key(self):
        return self._model_feature_names_key

    @model_feature_names_key.setter
    def model_feature_names_key(self, new_val):
        self._model_feature_names_key = new_val

    @property
    def model_feature_names(self) -> np.ndarray:
        return self._model_feature_names

    @model_feature_names.setter
    def model_feature_names(self, new_val):
        self._model_feature_names = new_val

    @constant
    def TIEBREAKER_MULTIPLIER() -> float:
        return 2e-23

    def __init__(
            self, mlmodel_metadata_key: str = 'json',
            model_feature_names_key: str = 'feature_names',
            model_seed_key: str = 'model_seed',
            model_name_key: str = 'model_name'):
        # initialize
        self.model = None
        self.model_metadata_key = mlmodel_metadata_key
        self.model_metadata = None

        self.feature_encoder = None

        self.model_seed_key = model_seed_key
        self.model_seed = None

        self.model_name_key = model_name_key
        self.model_name = None

        self.model_objective = None

        self.model_feature_names_key = model_feature_names_key
        self.model_feature_names = np.empty(shape=(1,))

    def _load_buffered_model(self, model_bytes: bytes):
        """
        # TODO make this in memory 
        Loads mlmodel using temp cache file  
        
        Parameters
        ----------
        model_bytes: bytes
            model bytes

        Returns
        -------

        """

        spec = ct.proto.Model_pb2.Model()
        spec.ParseFromString(model_bytes)
        return ct.models.MLModel(spec)

    def load_model(self, input_model_src: str, verbose: bool = False):
        """
        Loads desired model from input path.

        Parameters
        ----------
        input_model_src: str
            path to desired model
        verbose: bool
            should I print msgs

        Returns
        -------
        None
            None

        """

        failed_to_load = False
        try:
            if verbose:
                print('Attempting to load: {} model'.format(
                    input_model_src if len(input_model_src) < 100 else
                    str(input_model_src[:10]) + ' ... ' + str(
                        input_model_src[-10:])))

            raw_model_src = self.get_model_src(model_src=input_model_src)

            # model_src = self._load_buffered_model(model_bytes=raw_model_src)

            self.model = \
                ct.models.MLModel(raw_model_src) \
                if not isinstance(raw_model_src, bytes) \
                else self._load_buffered_model(model_bytes=raw_model_src)

            if verbose and not failed_to_load:
                print('Model: {} successfully loaded'.format(
                    input_model_src if len(input_model_src) < 100 else
                    str(input_model_src[:10]) + ' ... ' + str(input_model_src[-10:])))
        except Exception as exc:
            if verbose:
                print(
                    'When attempting to load the mode: {} the following error '
                    'occured: {}'.format(input_model_src, exc))

        if failed_to_load:
            raise RuntimeError('Model failed to load')

        model_metadata = self._get_model_metadata()

        self.model_seed = self._get_model_seed(model_metadata=model_metadata)
        self.model_name = self._get_model_name(model_metadata=model_metadata)
        self.model_feature_names = \
            self._get_model_feature_names(model_metadata=model_metadata)

        if USE_CYTHON_BACKEND:
            self.feature_encoder = FastFeatureEncoder(model_seed=self.model_seed)
        else:
            self.feature_encoder = FeatureEncoder(model_seed=self.model_seed)

    def _get_model_metadata(self) -> dict:
        """
        Gets model metadata either from dict or from model attributes

        Returns
        -------
        dict
            metadata dict

        """

        assert hasattr(self.model, 'user_defined_metadata')
        assert self.model_metadata_key in self.model.user_defined_metadata.keys()
        return json.loads(
            self.model.user_defined_metadata[self.model_metadata_key])

    def _score(
            self, variant: Dict[str, object], noise: float,
            givens: Dict[str, object] = None,
            encoded_givens: Dict[str, object] = None, **kwargs) -> list:
        """
        Performs scoring of a single variant using provided context and loaded
        model

        Parameters
        ----------
        variant: dict
            scored case / row as a dict
        givens: dict
            dict with lookup table and seed
        encoded_givens: Dict[str, float]
            already encoded context dict
        kwargs

        Returns
        -------
        np.ndarray
            score of provided variant

        """

        if not encoded_givens:
            encoded_givens = \
                self.feature_encoder.encode_givens(givens=givens, noise=noise)

        encoded_variant_and_context = \
            self.feature_encoder.encode_variant(
                variant=variant, noise=noise, into=encoded_givens)

        missings_filled_v = \
            encoded_variant_to_np(
                encoded_variant=encoded_variant_and_context,
                feature_names=self.model_feature_names)

        assert len(self.model_feature_names) == len(missings_filled_v)

        score_dict = \
            self.model.predict(
                dict(zip(self.model_feature_names, missings_filled_v)))

        if 'target' not in score_dict:
            raise ValueError(
                'Prediction dict has no `target` key: {}'.format(score_dict))

        score = score_dict.get('target', None)

        return score

    def score(
            self, variants: List[Dict[str, object]],
            givens: Dict[str, object], **kwargs) -> np.ndarray:
        """
        Scores all provided variants

        Parameters
        ----------
        variants: list
            list of variants to scores
        givens: dict
            dict with lookup table and seed
        kwargs
            kwargs

        Returns
        -------
        np.ndarray
            2D numpy array which contains (variant, score) pair in each row

        """

        # encoded_context = \
        #     self.feature_encoder.encode_features({'context': context})

        noise = np.random.rand()

        encoded_context = \
            self.feature_encoder.encode_givens(givens=givens, noise=noise)

        scores = \
            np.array([self._score(
                variant=variant, noise=noise, givens=givens,
                encoded_givens=encoded_context) for variant in variants])

        return scores


if __name__ == '__main__':

    mlmc = MLModelChooser()

    # test_model_pth = '../artifacts/test_artifacts/improve-messages-2.0-3.mlmodel'
    test_model_pth = '../artifacts/models/v6_conv_model.mlmodel'
    # test_model_pth = "https://improve-v5-resources-prod-models-117097735164.s3-us-west-2.amazonaws.com/models/mindful/latest/improve-messages-2.0.mlmodel"
    # test_model_pth = "/Users/os/Downloads/improve-messages-2.0.mlmodel.gz"
    mlmc.load_model(input_model_src=test_model_pth)

    with open('../artifacts/test_artifacts/context.json', 'r') as mj:
        json_str = mj.readline()
        context = json.loads(json_str)

    with open('../artifacts/test_artifacts/meditations.json', 'r') as vj:
        json_str = vj.readlines()
        variants = json.loads(''.join(json_str))

    noise = np.random.rand()
    res = mlmc._score(variant=variants[0], givens=context, noise=noise)
    print('res')
    print(res)
    # 0.0012913734900291936
    # input('check')

    batch_size = 10
    print(len(variants))
    st = time()

    for _ in range(batch_size):
        # res = mlmc.score(variant=variants[0], context=context)
        res_all = \
            mlmc.score(variants=variants, givens=context)
    et = time()
    print((et - st) / batch_size)
    input('sanity check')
    print('res_all')

    srtd_variants_w_scores = mlmc.sort(variants_w_scores=res_all)
    print('srtd_variants_w_scores')
    for row in srtd_variants_w_scores.tolist():
        print('{} -> sigmoid({}) = {}'.format(row, row[1], round(1 / (1 + np.exp(0.5 - row[1])), 4)))

    best_choice = mlmc.choose(variants_w_scores=srtd_variants_w_scores)
    print('best_choice')
    print(best_choice)


# {28: 1.0, 33: 1.0, 17: 1.0, 39: 1.0, 42: 1.0, 11: 1.0, 37: 1.0, 9: 1.0, 31: 1.0, 38: 1.0, 41: 1.0, 21: 1.0, 3: 1.0, 43: 1.0, 22: 1.0, 2: 1.0, 8: 1.0, 14: 1.0, 40: 1.0, 25: 1.0, 20: 1.0, 30: 1.0, 16: 1.0, 4: 1.0, 19: 1.0, 6: 1.0, 23: 1.0, 26: 1.0, 24: 1.0, 27: 1.0, 18: 1.0, 29: 1.0, 10: 1.0, 1: 1.0, 32: 1.0, 13: 1.0, 36: 1.0, 34: 1.0, 7: 1.0, 0: 1.0, 12: 1.0, 35: 1.0, 15: 1.0, 5: 1.0}
# {28: 1.0, 33: 1.0, 17: 1.0, 39: 1.0, 42: 1.0, 11: 1.0, 37: 1.0, 9: 1.0, 31: 1.0, 38: 1.0, 41: 1.0, 21: 1.0, 3: 1.0, 43: 1.0, 22: 1.0, 2: 1.0, 8: 1.0, 14: 1.0, 40: 1.0, 25: 1.0, 20: 1.0, 30: 1.0, 16: 1.0, 4: 1.0, 19: 1.0, 6: 1.0, 23: 1.0, 26: 1.0, 24: 1.0, 27: 1.0, 18: 1.0, 29: 1.0, 10: 1.0, 1: 1.0, 32: 1.0, 13: 1.0, 36: 1.0, 34: 1.0, 7: 1.0, 0: 1.0, 12: 1.0, 35: 1.0, 15: 1.0, 5: 1.0}
