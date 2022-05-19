import re
import warnings

import numpy as np

from improveai.chooser import XGBChooser
import improveai.decision_context as dc
import improveai.decision_tracker as dt
import improveai.givens_provider as gp
from improveai.settings import DEBUG
from improveai.utils.general_purpose_tools import constant, check_variants, \
    get_variants_from_args, is_valid_ksuid


class DecisionModel:

    SUPPORTED_CALLS = ['score', 'top_scoring_variant', 'rank', 'get']
    MODEL_NAME_REGEXP = XGBChooser.MODEL_NAME_REGEXP

    @property
    def model_name(self) -> str or None:
        """
        Name of DecisionModel. None value is allowed but otherwise value must be a string and pass `MODEL_NAME_REGEXP` check

        Returns
        -------
        str of None
            `model_name` of this DecisionModel

        """

        return self.__model_name

    @model_name.setter
    def model_name(self, value: str or None):
        if value is not None:
            assert isinstance(value, str)
            assert re.search(DecisionModel.MODEL_NAME_REGEXP, value) is not None
        self.__model_name = value

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
        if hasattr(self, '__tracker') and getattr(self, '__tracker') is not None:
            self.__tracker.track_url = value

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

    @constant
    def TIEBREAKER_MULTIPLIER() -> float:
        """
        Small value randomized and added to model's scores

        Returns
        -------
        float
            Small value randomized and added to model's scores

        """
        return 2**-23

    def __init__(
            self, model_name: str, track_url: str = None, track_api_key: str = None):
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

        self.model_name = model_name

        self.track_url = track_url
        self.__track_api_key = track_api_key

        self.__tracker = dt.DecisionTracker(track_url=track_url, track_api_key=self.__track_api_key)

        self.chooser = None
        self.givens_provider = gp.GivensProvider()

    def load(self, model_url: str):
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

    def _resolve_model_name(self):
        """
        If model name is not set loaded model name is used, otherwise model name
        from constructor `wins`

        Returns
        -------

        """

        if self.model_name is None:
            self.model_name = self.chooser.model_name
        else:
            self.chooser.model_name = self.model_name
            warnings.warn(
                'Model name passed to the constructor: {} will not be '
                'overwritten by loaded model name: {}.'
                    .format(self.model_name, self.chooser.model_name))

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

        # get givens from provider
        givens = self.givens_provider.givens(for_model=self)
        # return equivalent of double scores
        return self._score(variants=variants, givens=givens)

    def _score(self, variants: list or np.ndarray, givens: dict) -> np.ndarray:
        """
        Call predict and calculate scores for provided variants

        Parameters
        ----------
        variants: list or np.ndarray
            collection of variants to be scored

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
            return DecisionModel.generate_descending_gaussians(count=len(variants))

        try:
            scores = \
                self.chooser.score(
                    variants=variants, givens=givens) + \
                np.array(np.random.rand(len(variants)), dtype='float64') * \
                self.TIEBREAKER_MULTIPLIER
        except Exception as exc:
            warnings.warn(
                'Error when calculating predictions: {}. Returning Gaussian scores'
                    .format(exc))
            scores = DecisionModel.generate_descending_gaussians(count=len(variants))
        return scores.astype(np.float64)

    @staticmethod
    def _validate_variants_and_scores(variants: list or np.ndarray, scores: list or np.ndarray) -> bool:
        """
        Check if variants and scores are not malformed

        Parameters
        ----------
        variants: np.ndarray
            array of variants
        scores: np.ndarray
            array of scores

        Returns
        -------
        bool
            Flag indicating whether variants and scores are valid

        """

        if variants is None or scores is None:
            raise ValueError('`variants` and `scores` can\'t be None')

        if len(variants) != len(scores):
            raise ValueError('Lengths of `variants` and `scores` mismatch!')

        if len(variants) == 0:
            return False

        return True

    @staticmethod
    def top_scoring_variant(variants: list or np.ndarray, scores: list or np.ndarray) -> object:
        """
        Gets best variant considering provided scores

        Parameters
        ----------
        variants: np.ndarray
            collection of variants to be ranked
        scores: np.ndarray
            collection of scores used for ranking

        Returns
        -------
        object
            best variant

        """

        assert variants is not None and scores is not None
        if not DecisionModel._validate_variants_and_scores(
                variants=variants, scores=scores):
            warnings.warn('The variants of length 0 were provided. Returning None')
            return None

        return variants[np.argmax(scores)]

    @staticmethod
    def rank(variants: list or np.ndarray, scores: list or np.ndarray) -> np.ndarray:
        """
        Return a list of the variants ranked from best to worst.
        DO NOT USE THIS METHOD - WILL LIKELY CHANGE SOON

        Parameters
        ----------
        variants: np.ndarray
            collection of variants to be ranked
        scores: np.ndarray
            collection of scores used for ranking

        Returns
        -------
        np.ndarray
            sorted variants

        """

        assert variants is not None and scores is not None
        if not DecisionModel._validate_variants_and_scores(
                variants=variants, scores=scores):
            return None

        variants_w_scores = np.array([variants, scores], dtype=object).T

        sorted_variants_w_scores = \
            variants_w_scores[(variants_w_scores[:, 1] * -1).argsort()]
        return sorted_variants_w_scores[:, 0]

    @staticmethod
    def generate_descending_gaussians(count: int) -> np.ndarray:
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

    def given(self, givens: dict or None) -> dc.DecisionContext:
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

    def choose_from(self, variants: np.ndarray or list or tuple, scores: list or np.ndarray = None):
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

        return dc.DecisionContext(decision_model=self, givens=None).choose_from(variants=variants, scores=scores)

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

        decision = self.choose_from(variants=get_variants_from_args(variants=variants))
        best = decision.get()
        return best, decision.id_

    def choose_first(self, variants: list or tuple or np.ndarray):
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
        return dc.DecisionContext(decision_model=self, givens=None).choose_first(variants=variants)

    def first(self, *variants: list or np.ndarray):
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

        return dc.DecisionContext(decision_model=self, givens=None).first(*variants)

    def choose_random(self, variants: list or tuple or np.ndarray):
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
        return dc.DecisionContext(decision_model=self, givens=None).choose_random(variants=variants)

    def random(self, *variants: list or np.ndarray):
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
        return dc.DecisionContext(decision_model=self, givens=None).random(*variants)

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
        assert isinstance(reward, float) or isinstance(reward, int)
        assert reward is not None
        assert not np.isnan(reward)
        assert not np.isinf(reward)

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
