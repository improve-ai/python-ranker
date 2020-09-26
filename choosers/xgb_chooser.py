import coremltools as ct
import json
import numpy as np
from typing import Dict, List
from xgboost import Booster, DMatrix

from encoders.feature_encoder import FeatureEncoder


class BasicNativeXGBChooser:

    @property
    def des_mlmodel(self) -> ct.models.MLModel:
        return self._des_model

    @des_mlmodel.setter
    def des_mlmodel(self, new_val: ct.models.MLModel):
        self._des_model = new_val

    def __init__(self):
        # initialize
        self.des_mlmodel = None

    def load_mlmodel(self, pth_to_model: str, verbose: bool = True):
        """
        Loads desired model from input path.

        Parameters
        ----------
        pth_to_model: str
            path to desired model
        verbose: bool
            should I print msgs

        Returns
        -------
        None
            None

        """

        try:
            if verbose:
                print('Attempting to load: {} model'.format(pth_to_model))
                self.des_mlmodel = Booster()
                self.des_mlmodel.load_model(pth_to_model)
            if verbose:
                print('Model: {} successfully loaded'.format(pth_to_model))
        except Exception as exc:
            if verbose:
                print(
                    'When attempting to load the mode: {} the following error '
                    'occured: {}'.format(pth_to_model, exc))

    def _get_encoded_features(
            self, variant: Dict[str, object],
            context: Dict[str, object],
            context_table_key: str = "table",
            seed_key: str = "model_seed") -> Dict[str, float]:
        """
        Encodes features using existing FeatureEncoder class

        Parameters
        ----------
        variant: dict
            dict of features to be encoded
        context: dict
            dict with lookup table and seed
        context_table_key: str
            table key in context dict
        seed_key: str
            seed key in context dict

        Returns
        -------
        dict
            encoded features dict

        """

        lookup_table = context.get(context_table_key, None)
        model_seed = context.get(seed_key, None)
        if not lookup_table or not model_seed:
            raise ValueError(
                'Lookup table or model seed not present in context!')

        fe = FeatureEncoder(table=lookup_table, model_seed=model_seed)
        return fe.encode_features(variant)

    def _get_feature_names(
            self, context: Dict[str, object],
            context_table_key: str = "table",
            context_table_features_idx: int = 1) -> List[str]:
        """
        Creates feature names from provided lookup table

        Parameters
        ----------
        context: dict
            context dict with lookup table and seed
        context_table_key: sre
            context lookup table key
        context_table_features_idx: int
            index in table storing features encoding representation (?)

        Returns
        -------
        list
            list of feature names extracted from context

        """

        features_count = \
            len(context[context_table_key][context_table_features_idx])
        feature_names = list(
            map(lambda i: 'f{}'.format(i), range(0, features_count)))
        return feature_names

    def score_all(
            self, variants: List[Dict[str, object]],
            context: Dict[str, object],
            context_table_key: str = "table",
            seed_key: str = "model_seed") -> np.ndarray:

        """
        Scores all provided variants

        Parameters
        ----------
        variants: list
            list of variants to scores
        context: dict
            context dict needed for encoding

        Returns
        -------
        np.ndarray
            2D numpy array which contains (variant, score) pair in each row

        """

        scores = []
        encoded_features = []

        for variant in variants:
            encoded_features.append(list(self._get_encoded_features(
                variant=variant, context=context,
                context_table_key=context_table_key, seed_key=seed_key).values()))

        print("encoded_features")
        print(encoded_features)

        x = np.array(encoded_features).reshape(len(encoded_features),len(encoded_features[0]))
        print("encoded_features array")
        print(x)

        print("score")
        print(self.score(x, context=context))





        # assert len(scores) == len(variants)
        # # variants_w_scores = np.array([[json.dumps(el) for el in variants], scores]).reshape((-1, 2))
        # variants_w_scores = np.array([variants, scores]).reshape((-1, 2))
        # return variants_w_scores.T

    def score(
            self, variant: Dict[str, object],
            context: Dict[str, object], context_table_key: str = "table",
            context_table_features_idx: int = 1,
            seed_key: str = "model_seed") -> float:

        """
        Performs scoring of a single variant using provided context and loaded
        model

        Parameters
        ----------
        variant: dict
            scored case / row as a dict
        context: dict
            dict with lookup table and seed
        context_table_key: str
            context key storing lookup table
        context_table_features_idx: int
            index of a lookup table storing features encoding info (?)
        seed_key: str
            context key storing model seed

        Returns
        -------
        float
            score of provided variant

        """

        encoded_features = \
            self._get_encoded_features(
                variant=variant, context=context,
                context_table_key=context_table_key, seed_key=seed_key)
        # rename features
        feature_names = \
            self._get_feature_names(
                context=context, context_table_key=context_table_key,
                context_table_features_idx=context_table_features_idx)

        print(dict(zip(feature_names, list(encoded_features.values()))))
        input = list(encoded_features.values())

        score = \
            self.des_mlmodel.predict(
            DMatrix(np.array(input).reshape(1,len(input))))

        return score

    def sort(
            self, variants_w_scores: np.ndarray,
            scores_col_idx: int = 1) -> np.ndarray:
        """
        Performs sorting of provided variants with scores array

        Parameters
        ----------
        variants_w_scores: np.ndarray
            array with variant, scores rows
        scores_col_idx: int
            the index of column with scores

        Returns
        -------
        np.ndarray
            2D sorted array of rows (variant, score)

        """
        desc_sorting_col = -1 * variants_w_scores[:, scores_col_idx]

        srtd_variants_w_scores = \
            variants_w_scores[np.argsort(desc_sorting_col)]

        return srtd_variants_w_scores

    def choose(
            self, variants_w_scores: np.ndarray,
            scores_col_idx: int = 1) -> np.ndarray:
        """
        Chooses the variant with the highest score. Randomly breaks ties

        Parameters
        ----------
        variants_w_scores: np.ndarray
            2D array of (variant, score) rows
        scores_col_idx: int
            int indicating column with scores

        Returns
        -------
        np.ndarray
            1D array with <best variant, best_score>

        """
        # must break ties

        scores = variants_w_scores[:, scores_col_idx]
        top_variants_w_scores = variants_w_scores[scores == scores.max()]
        chosen_top_variant_idx = \
            np.random.randint(top_variants_w_scores.shape[0])

        return top_variants_w_scores[chosen_top_variant_idx]


if __name__ == '__main__':

    mlmc = BasicNativeXGBChooser()

    test_model_pth = '../py_mlmodel_poc/model.xgb'
    mlmc.load_mlmodel(pth_to_model=test_model_pth)

    with open('../test_artifacts/model.json', 'r') as mj:
        json_str = mj.readline()
        context = json.loads(json_str)

    features_count = \
        len(context["table"][1])
    feature_names = list(
        map(lambda i: 'f{}'.format(i), range(0, features_count)))

    sample_variants = [{"arrays": [el for el in range(0, features_count)]},
                       {"arrays": [el + 1 for el in range(0, features_count)]}]

    res = mlmc.score(variant=sample_variants[0], context=context)
    res = mlmc.score_all(variants=sample_variants, context=context)

    print(res)
