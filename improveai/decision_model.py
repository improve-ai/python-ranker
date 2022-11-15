from pathlib import Path
import re
import warnings

import numpy as np

from improveai.chooser import XGBChooser
import improveai.decision as d
import improveai.decision_context as dc
import improveai.decision_tracker as dt
import improveai.givens_provider as gp
from improveai.settings import DEBUG
from improveai.utils.general_purpose_tools import check_variants, \
    get_variants_from_args, is_valid_ksuid, is_valid_variants_type, ALLOWED_VARIANT_COLLECTION_TYPES


class DecisionModel:

    SUPPORTED_CALLS = ['score', 'rank', 'get', 'which_from', 'optimize']
    """
    Method calls supported by Improve model CLI
    """

    MODEL_NAME_REGEXP = XGBChooser.MODEL_NAME_REGEXP
    """
    Model name regexp used to verify all model names (both user provided and cached in boosters)
    """

    @property
    def model_name(self) -> str or None:
        """
        Name of DecisionModel. None value is allowed but otherwise value must be
        a string and pass `MODEL_NAME_REGEXP` check

        Returns
        -------
        str of None
            `model_name` of this DecisionModel

        """

        return self.__model_name

    @property
    def chooser(self) -> XGBChooser:
        """
        Chooser used by current DecisionModel. Currently only XGBChooser is supported.
        If not set will return None

        Returns
        -------
        XGBChooser or None
            DecisionModel's chooser

        """
        return self._chooser

    @chooser.setter
    def chooser(self, value: XGBChooser or None):
        self._chooser = value

    @property
    def givens_provider(self) -> gp.GivensProvider:
        """
        GivensProvider used by this DecisionModel

        Returns
        -------
        GivensProvider
            GivensProvider used by this DecisionModel

        """
        return self._givens_provider

    @givens_provider.setter
    def givens_provider(self, value):
        self._givens_provider = value

    @property
    def track_url(self) -> str or None:
        """
        Track endpoint URL used by this model. Also updates `track_url` of a DecisionTracker

        Returns
        -------
        str or None
            track endpoint URL string

        """
        return self._track_url

    @track_url.setter
    def track_url(self, value: str or None):
        self._track_url = value

        # cases:
        # 1. __tracker is set and is not None
        # 2. __tracker is set and is None
        # 3. __tracker is not set

        # additionally self.track_api_key:
        # 1. is set
        # 2. is not set

        # Allow None value in constructor
        if hasattr(self, '_DecisionModel__tracker') and getattr(self, '_DecisionModel__tracker') is not None:
            self.__tracker.track_url = value
        elif (not hasattr(self, '_DecisionModel__tracker') or self.__tracker is None) and value is not None:
            self.__tracker = dt.DecisionTracker(
                track_url=value,
                track_api_key=self.track_api_key if hasattr(self, '_DecisionModel__track_api_key') else None)

    @property
    def track_api_key(self):
        """
        track API key to be sent as the x-api-key HTTP header on track requests. Omit header if null

        Returns
        -------
        str
            current track API key

        """
        return self.__track_api_key

    @track_api_key.setter
    def track_api_key(self, value):
        """
        sets track API key to be sent as the x-api-key HTTP header on track requests. Omit header if null

        Parameters
        ----------
        value: str
            track API key for this instance

        """
        # set API key to decision model
        self.__track_api_key = value
        # update API key on the tracker
        if hasattr(self, '_DecisionModel__tracker') and getattr(self, '_DecisionModel__tracker') is not None:
            self.__tracker.api_key = value

    @property
    def tracker(self) -> dt.DecisionTracker:
        """
        DecisionTracker of this DecisionModel

        Returns
        -------
        DecisionTracker or None
            `tracker` of current DecisionModel or None if `tracker` was not set

        """
        return self.__tracker

    @property
    def last_decision_id(self):
        """
        DecisionModel persists last tracked decision ID in `last_decision_id` property

        Returns
        -------
        str
            Ksuid as string

        """
        return self._last_decision_id

    @last_decision_id.setter
    def last_decision_id(self, value):
        is_valid_ksuid(value)
        self._last_decision_id = value

    @property
    def TIEBREAKER_MULTIPLIER(self) -> float:
        """
        Small value randomized and added to model's scores

        Returns
        -------
        float
            Small value randomized and added to model's scores

        """
        return 2**-23

    def __init__(self, model_name: str, track_url: str = None, track_api_key: str = None):
        """
        Init with params

        Parameters
        ----------
        model_name: str
            model name for this DecisionModel; can be None
        track_url: str
            a valid URL leading to improve gym track endpoint
        track_api_key: str
            track endpoint API key (if available); can be None
        """

        self.__check_model_name(model_name=model_name)
        self.__model_name = model_name

        self.track_url = track_url
        self.track_api_key = track_api_key

        if not hasattr(self, '_DecisionModel__tracker'):
            self.__tracker = None

        self.chooser = None
        self.givens_provider = gp.GivensProvider()

        # init `self._last_decision_id` to None
        self._last_decision_id = None

    def __check_model_name(self, model_name: str):
        """
        Check if provided model name is a valid model name

        Parameters
        ----------
        model_name: str
            checked model name

        """
        if model_name is not None:
            assert isinstance(model_name, str), \
                f'Invalid model name type - expected string got {type(model_name)}'
            assert re.search(DecisionModel.MODEL_NAME_REGEXP, model_name) is not None, \
                f'Valid model name must pass regexpr: {DecisionModel.MODEL_NAME_REGEXP}'

    def load(self, model_url: str or Path):
        """
        Synchronously loads XGBoost model from provided path, creates instance
        of DecisionModel and returns it

        Parameters
        ----------
        model_url: str
            path to desired model (FS path or url)

        Returns
        -------
        DecisionModel
            new instance of decision model

        """
        self.chooser = self._get_chooser(model_url=model_url)
        self._resolve_model_name()

        return self

    def _is_loaded(self):
        """
        Checks if a valid decision model was loaded

        Returns
        -------
        bool
            was model loaded or not

        """
        return self.chooser is not None and isinstance(self.chooser, XGBChooser)

    def _resolve_model_name(self):
        """
        If model name is not set loaded model name is used, otherwise model name
        from constructor `wins`

        Returns
        -------

        """

        if self.model_name is None:
            self.__check_model_name(model_name=self.chooser.model_name)
            self.__model_name = self.chooser.model_name
        else:
            self.chooser.model_name = self.model_name

    def _get_chooser(self, model_url: str) -> XGBChooser:
        """
        Synchronously loads XGBoost model from provided path, creates instance
        of Chooser and returns it

        Parameters
        ----------
        model_url: str
            path to desired model (FS path or url)

        Returns
        -------
        XGBChooser
            loaded chooser

        """

        try:
            model_src = XGBChooser.get_model_src(model_src=model_url)
        except Exception as exc:
            raise exc

        chooser = XGBChooser()
        chooser.load_model(input_model_src=model_src)
        return chooser

    def score(self, variants: list or np.ndarray) -> np.ndarray:
        """
        Scores provided variants with available givens

        Parameters
        ----------
        variants: list or tuple or np.ndarray
            array of variants to choose from

        Returns
        -------
        np.ndarray
            array of float64 scores

        """

        # use GivensProvider for givens (this will result in empty givens for now)
        # return equivalent of double scores
        return self._score(variants=variants, givens=self.givens_provider.givens(for_model=self))

    def _score(self, variants: list or np.ndarray, givens: dict or None) -> np.ndarray:
        """
        Call predict and calculate scores for provided variants

        Parameters
        ----------
        variants: list or np.ndarray
            collection of variants to be scored
        givens: dict or None
            givens for variants

        Returns
        -------
        np.ndarray
            scores

        """

        check_variants(variants=variants)
        # log givens for DEBUG == True
        if DEBUG is True:
            print(f'[DEBUG] givens: {givens}')
        if self.chooser is None:
            return DecisionModel._generate_descending_gaussians(count=len(variants))

        # encode variants with single givens
        encoded_variants = \
            self.chooser.encode_variants_single_givens(variants=variants, givens=givens)
        # fill missing features and create numpy matrix for xgboost to predict on
        features_matrix = self.chooser.fill_missing_features(encoded_variants=encoded_variants)

        try:
            # calculate predictions
            scores = self.chooser.calculate_predictions(features_matrix=features_matrix) + \
                     np.array(np.random.rand(len(variants)), dtype='float64') * \
                     self.TIEBREAKER_MULTIPLIER
        except Exception as exc:
            warnings.warn(
                'Error when calculating predictions: {}. Returning Gaussian scores'
                .format(exc))
            scores = DecisionModel._generate_descending_gaussians(count=len(variants))

        return scores.astype(np.float64)

    def _rank(self, variants: list or np.ndarray, scores: list or np.ndarray) -> list or np.ndarray:
        """
        Helper method to rank variants. Returns a numpy array with variants ranked from best to worst

        Parameters
        ----------
        variants: np.ndarray
            collection of variants to be ranked
        scores: list or np.ndarray
            scores for provided variants

        Returns
        -------
        list or np.ndarray
            sorted variants

        """

        # assure provided variants are not empty
        assert len(variants) > 0
        # make sure score have the same length that ranked variants
        assert len(variants) == len(scores)

        # convert variants to numpy array for faster sorting
        best_to_worse_scores = np.argsort(scores)[::-1]
        # return descending sorted variants
        if isinstance(variants, np.ndarray):
            return variants[best_to_worse_scores]
        return [variants[i] for i in best_to_worse_scores]

    def rank(self, variants: list or np.ndarray) -> list:
        """
        Ranks provided variants from best to worst and creates decision from them

        Parameters
        ----------
        variants: list or np.ndarray
            variants to be ranked

        Returns
        -------
        list or np.ndarray
            best to worst ranked variants
        """
        return self.decide(variants=variants).ranked

    def given(self, givens: dict) -> dc.DecisionContext:
        """
        Wrapper for chaining.

        Parameters
        ----------
        givens: dict
            givens to be set

        Returns
        -------
        DecisionContext
            decision context for a given model

        """

        return dc.DecisionContext(decision_model=self, givens=givens)

    def which(self, *variants) -> tuple:
        """
        A short hand version of chooseFrom that returns the chosen result directly,
        automatically tracking the decision in the process. This would be a varargs
        version of chooseFrom that would also take an array as the only argument.
        If the only argument was not an array it would throw an exception.
        Since the Decision object is not returned, rewards would only be tracked
        through DecisionModel.addReward

        Parameters
        ----------
        variants: list or tuple or np.ndarray
            array of variants passed as positional parameters

        Returns
        -------
        object, str
            tuple with (<best variant>, <decision id>)

        """

        return self.which_from(variants=get_variants_from_args(variants))

    def which_from(self, variants: list or np.ndarray) -> tuple:
        """
        Makes a decision for provided variants and DecisionContext givens.

        Parameters
        ----------
        variants: list or np.ndarray
            variants from which best will be chosen

        Returns
        -------
        object, str
            a tuple of (<best variant>, <decision id>)

        """

        # create a Decision object
        decision = self.decide(variants)
        # track if possible -> tracker and track_url exist
        if self.track_url is not None and self.tracker is not None:
            decision.track()
        # return best variant and decision ID to allow rewarding
        return decision.best, decision.id_

    def add_reward(self, reward: float, decision_id: str):
        """
        Adds provided reward for a  given decision id.

        Parameters
        ----------
        reward: float or int
            reward to be assigned to a given decision
        decision_id: str
            ksuid of rewarded decision

        Returns
        -------

        """
        assert self.model_name is not None
        assert re.search(XGBChooser.MODEL_NAME_REGEXP, self.model_name) is not None

        # make sure provided decision id is valid
        assert decision_id is not None and is_valid_ksuid(decision_id)

        if self.__tracker is not None:
            return self.__tracker.add_reward(
                reward=reward, model_name=self.model_name, decision_id=decision_id)
        else:
            if self.__tracker is None:
                warnings.warn(
                    '`tracker` is not set (`tracker`is None) - reward not added')
            if self.track_url is None:
                warnings.warn('`track_url` is None - reward not added')

    def decide(self, variants: list or np.ndarray, scores: list or np.ndarray = None,
               ordered: bool = False):
        """
        Creates decision but does not track it by default.
        If not scores are provided and input variants are not ranked performs
        scoring and ranking. If scores are provided but variants are not ordered
        ranks variants before making a decision

        Parameters
        ----------
        variants: list or np.ndarray
            variants used to make decision
        scores: list or np.ndarray
            scores for variants
        ordered: bool
            are variants ordered from best to worst

        Returns
        -------
        d.Decision
            decision object with provided input

        """

        return self.given(givens=self.givens_provider.givens(for_model=self))\
            .decide(variants=variants, scores=scores, ordered=ordered)

    def optimize(self, variant_map: dict):
        """
        Get the best configuration for a given variants map

        Parameters
        ----------
        variant_map: dict
            mapping variants name -> variants list

        Returns
        -------
        object, str
            best variant and a decision ID

        """
        return self.which_from(variants=DecisionModel.full_factorial_variants(variant_map=variant_map))

    def choose_from(self, variants: list or np.ndarray, scores: list or np.ndarray):
        # TODO method deprecated - will be removed in v8 upgrade
        """
        Wrapper for chaining

        Parameters
        ----------
        variants: np.ndarray or list or tuple
            variants to be set
        scores: list or np.ndarray
            list of already calculated scores

        Returns
        -------
        Decision
            decision object with provided with already scored variants and best selected
        """
        return self.decide(variants=variants, scores=scores)

    def choose_first(self, variants: list or np.ndarray):
        # TODO method deprecated - will be removed in v8 upgrade
        """
        Chooses first from provided variants using gaussian scores (_generate_descending_gaussians())

        Parameters
        ----------
        variants: list or tuple or np.ndarray
            collection of variants passed as positional parameters

        Returns
        -------
        Decision
            A decision with first variants as the best one and gaussian scores
        """

        # make decision and do not track it
        return self.decide(variants=variants, ordered=True)

    def first(self, *variants):
        # TODO method deprecated - will be removed in v8 upgrade
        """
        Makes decision using first variant as best and tracks it.
        Accepts variants as pythonic args

        Parameters
        ----------
        *variants: list or tuple or np.ndarray
            collection of variants of which first will be chosen

        Returns
        -------
        object, str
            tuple with (<first variant>, <decision id>)
        """
        decision = self.decide(variants=get_variants_from_args(variants), ordered=True)
        return decision.get(), decision.id_

    def choose_random(self, variants: list or np.ndarray):
        # TODO method deprecated - will be removed in v8 upgrade
        """
        Shuffles variants to return Decision with gaussian scores and random best variant

        Parameters
        ----------
        *variants: list or tuple or np.ndarray
            collection of variants of which random will be chosen

        Returns
        -------
        Decision
            Decision with randomly chosen best variant
        """
        check_variants(variants)
        return self.decide(variants, scores=np.random.normal(size=len(variants)))

    def random(self, *variants):
        # TODO method deprecated - will be removed in v8 upgrade
        """
        Makes decision using randomly selected variant as best and tracks it.
        Accepts variants as pythonic args

        Parameters
        ----------
        variants: list or np.ndarray
            collection of variants of which first will be chosen

        Returns
        -------
        object, str
            tuple with (<random variant>, <decision id>)
        """
        # TODO test if it raises the same way choose_random raises
        check_variants(variants)
        unpacked_variants = get_variants_from_args(variants)
        decision = self.decide(
            variants=unpacked_variants,
            scores=np.random.normal(size=len(unpacked_variants)))
        return decision.get(), decision.id_

    def choose_multivariate(self, variant_map: dict) -> dict:
        # TODO method deprecated - will be removed in v8 upgrade
        """
        Chooses the best configuration from full factorial of input variants map

        Parameters
        ----------
        variant_map: dict
            map of key -> variants set

        Returns
        -------
        dict
            combination of the best variants in a dict

        """
        return self.given(givens=self.givens_provider.givens(for_model=self))\
            .choose_multivariate(variant_map=variant_map)

    def _track(self, variant: object, runners_up: list, sample: object, sample_pool_size: int):
        """
        Tracks provided variant with runners up and sample

        Parameters
        ----------
        variant: object
            variant to be tracked
        runners_up: list
            runners_up to be tracked
        sample: object
            a sample for an input variant
        sample_pool_size: int
            number of variants from which sample was drawn

        Returns
        -------
        str
            decision ID
        """

        return self.given(givens=self.givens_provider.givens(for_model=self)) \
            ._track(variant=variant, runners_up=runners_up, sample=sample, sample_pool_size=sample_pool_size)

    @staticmethod
    def _generate_descending_gaussians(count: int) -> np.ndarray:
        """
        Generates random floats and sorts in a descending fashion

        Parameters
        ----------
        count: int
            number of floats to generate

        Returns
        -------
        np.ndarray
            array of sorted floats

        """

        random_scores = np.random.normal(size=count)
        random_scores[::-1].sort()
        return random_scores

    @staticmethod
    def full_factorial_variants(variant_map: dict):
        """
        Creates full factorial from input variants map, i.e. for variants_map
        {
        'variants_0': ['01', '02'],
        'variants_1': ['11', '12', '13']}

        it creates a list of dicts:
        [
        {'variants_0': '01', 'variants_1': '11'},
        {'variants_0': '02', 'variants_1': '11'},
        {'variants_0': '01', 'variants_1': '12'},
        {'variants_0': '02', 'variants_1': '22'},
        {'variants_0': '01', 'variants_1': '13'},
        {'variants_0': '02', 'variants_1': '23'},
        ]

        Parameters
        ----------
        variant_map: dict
            map of key -> variants set

        Returns
        -------
        list
            list of 'fully factorial variants' dicts

        """
        # make sure variant map is a dict
        assert isinstance(variant_map, dict)
        # make sure that variant map is not empty
        assert len(variant_map) > 0
        # make sure that each key stores not None value (variants == None is not allowed)
        assert all(vs is not None for vs in variant_map.values())
        # make sure that all collections in variant_map are not empty
        assert all(len(vs) > 0 if type(vs) in ALLOWED_VARIANT_COLLECTION_TYPES else True for vs in variant_map.values())
        # check if all entries in variants_map are lists or tuples or np.arrays
        variants_map_fixed = {
            variants_key: variants if is_valid_variants_type(variants) else [variants]
            for variants_key, variants in variant_map.items()}
        # make sure all collections of variants are of a correct type
        [check_variants(vs) for vs in variants_map_fixed.values()]
        # 0. variant_map = {'variants_0': ['00', '01'], 'variants_1': ['11', '12', '13']}
        #
        # 1. meshgrid treats each sequence as a different 'variable' and prepares all vs all permutations
        #    meshgrid returns list of n-dimensional arrays where N is initial number of 'dimensions' (keys)
        #    in variant map (len(variant_map.keys()))
        #    variants_0 'dimension' meshgrid output ((3, 2) shape):
        #    [['00' '01']
        #     ['00' '01']
        #     ['00' '01']]
        #
        #    variants_1 'dimension' meshgrid output ((3, 2) shape):
        #    [['11' '11']
        #     ['12' '12']
        #     ['13' '13']]
        # 2. each element returned by meshgrid must be flattened ((1,) shape) and will represent a 'single
        #    dimension':
        #
        #    flattened variants_0 meshgrid result: ['00', '01', '00', '01', '00', '01']
        #    flattened variants_1 meshgrid result: ['11', '11', '12', '12', '13', '13']
        #
        #    Since each element was flattened calling np.vstack() will stack them 'vertically'
        #    (as if they were rows):
        #    [['00', '01', '00', '01', '00', '01'],
        #     ['11', '11', '12', '12', '13', '13']]
        #
        # 3. transposing np.vstack() result yields full factorial of values of variants map ->
        #    last step is to convert 'records' to dicts using variant_map keys

        all_variants = [np.array(vs, dtype='O') for vs in variants_map_fixed.values()]
        return [dict(zip(variants_map_fixed.keys(), variants)) for variants
                in np.vstack([v.flat for v in np.meshgrid(*all_variants)]).T]


def load_model(model_url: str, track_url: str = None):
    """
    Creates DecisionModel for provided model URL and track URL

    Parameters
    ----------
    model_url: str
        url to improve.ai model
    track_url: str
        url to improve.ai track endpoint

    Returns
    -------
    DecisionModel
        DecisionModel with Improve AI model and desired track_url

    """
    return DecisionModel(model_name=None, track_url=track_url).load(model_url=model_url)
