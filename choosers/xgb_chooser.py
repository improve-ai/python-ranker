import json
import numpy as np
from typing import Dict, List
from xgboost import Booster, DMatrix

from choosers.basic_choosers import BasicChooser
from encoders.feature_encoder import FeatureEncoder
from utils.gen_purp_utils import constant


class BasicNativeXGBChooser(BasicChooser):

    @property
    def model(self) -> Booster:
        return self._model

    @model.setter
    def model(self, new_val: Booster):
        self._model = new_val

    @constant
    def SUPPORTED_OBJECTIVES() -> list:
        return ['reg', 'binary', 'multi']

    def __init__(self):
        # initialize
        self.model = None

    def load_model(self, pth_to_model: str, verbose: bool = True):
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
                self.model = Booster()
                self.model.load_model(pth_to_model)
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
        enc_feats = fe.encode_features(variant)
        return enc_feats

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
            context_table_features_idx: int = 1,
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
        context_table_key: str
            context key storing lookup table
        context_table_features_idx: int
            index of a lookup table storing features encoding info (?)
        seed_key: str
            context key storing model seed
        Returns
        -------
        np.ndarray
            2D numpy array which contains (variant, score) pair in each row
        """

        encoded_features = []
        encoded_features_names = []

        for variant in variants:
            encoded_features.append(list(self._get_encoded_features(
                variant=variant, context=context,
                context_table_key=context_table_key, seed_key=seed_key).values()))
            encoded_features_names.append(self._get_feature_names(
                context=context, context_table_key=context_table_key,
                context_table_features_idx=context_table_features_idx))

        scores = \
            self.model.predict(
                DMatrix(np.array(encoded_features).reshape(len(encoded_features),len(encoded_features[0]))))

        assert len(scores) == len(variants)

        variants_w_scores_list = []

        for score, variant in zip(scores, variants):
            single_score_list = score if isinstance(score, list) else [score]
            variants_w_scores_list.append(
                self._get_processed_score(
                    scores=single_score_list, variant=variant))

        return np.array(variants_w_scores_list)  # np.column_stack([variants, scores])

    def score(
            self, variant: Dict[str, object],
            context: Dict[str, object], context_table_key: str = "table",
            context_table_features_idx: int = 1,
            seed_key: str = "model_seed") -> list:

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

        input_list = list(encoded_features.values())

        single_score_list = \
            self.model\
                .predict(DMatrix(
                    data=np.array(input_list).reshape(1,len(input_list)),
                    feature_names=feature_names))

        # score = single_score_list[0]
        score = \
            self._get_processed_score(scores=single_score_list, variant=variant)

        return score

    def _get_model_objective(self) -> str:
        model_objective = ''
        try:
            model_cfg = json.loads(self.model.save_config())
            model_objective = \
                model_cfg['learner']['learner_train_param']['objective'] \
                .split(':')[0]
        except Exception as exc:
            print('Cannot extract objective info from booster`s config')

        if model_objective not in self.SUPPORTED_OBJECTIVES:
            raise ValueError(
                'Unsupported booster objective: {}'.format(model_objective))
        return model_objective

    def _get_processed_score(self, scores, variant):

        model_objective = self._get_model_objective()

        if model_objective == 'reg':
            return [variant, float(scores[0]), int(0)]
        elif model_objective == 'binary':
            return [variant, float(scores[0]), int(1)]
        elif model_objective == 'multi':
            return [variant, float(np.array(scores).max()), int(np.array(scores).argmax())]

    def sort(
            self, variants_w_scores: np.ndarray,
            scores_col_idx: int = 1, class_cols_idx: int = 2) -> np.ndarray:
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

        desc_scores_sorting_col = -1 * variants_w_scores[:, scores_col_idx]
        class_sorting_col = variants_w_scores[:, class_cols_idx]

        ind = np.lexsort((desc_scores_sorting_col, class_sorting_col))

        srtd_variants_w_scores = \
            variants_w_scores[ind]

        return srtd_variants_w_scores

    def choose(
            self, variants_w_scores: np.ndarray,
            scores_col_idx: int = 1, class_col_idx: int = 2) -> np.ndarray:
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

        choices = []

        for curr_class_id in np.unique(variants_w_scores[:, class_col_idx]):
            curr_class_variants_w_scores = \
                variants_w_scores[
                    variants_w_scores[:, class_col_idx] == curr_class_id, :]

            scores = curr_class_variants_w_scores[:, scores_col_idx]

            top_variants_w_scores = \
                curr_class_variants_w_scores[scores == scores.max()]

            chosen_top_variant_idx = \
                np.random.randint(top_variants_w_scores.shape[0])

            choices.append(top_variants_w_scores[chosen_top_variant_idx])

        return np.array(choices)


if __name__ == '__main__':

    mlmc = BasicNativeXGBChooser()

    test_model_pth = '../test_artifacts/model.xgb'
    mlmc.load_model(pth_to_model=test_model_pth)

    with open('../test_artifacts/model.json', 'r') as mj:
        json_str = mj.readline()
        context = json.loads(json_str)

    features_count = \
        len(context["table"][1])
    feature_names = list(
        map(lambda i: 'f{}'.format(i), range(0, features_count)))

    sample_variants = [{"arrays": [1 for el in range(0, features_count + 1000)]},
                       {"arrays": [el + 2 for el in range(0, features_count + 1000)]},
                       {"arrays": [el for el in range(0, features_count + 1000)]}]

    single_score = mlmc.score(variant=sample_variants[0], context=context)
    print('single score')
    print(single_score)

    score_all = mlmc.score_all(variants=sample_variants, context=context)
    print('score_all')
    print(score_all)

    sorted = mlmc.sort(variants_w_scores=score_all)
    print('sorted')
    print(sorted)

    choose = mlmc.choose(variants_w_scores=score_all)
    print('choose')
    print(choose)
