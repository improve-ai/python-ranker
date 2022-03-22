import asyncio
import re
import warnings

import numpy as np

from improveai.choosers.basic_choosers import BasicChooser
from improveai.choosers.mlmodel_chooser import MLModelChooser
from improveai.choosers.xgb_chooser import NativeXGBChooser
import improveai.decision as d
import improveai.decision_context as dc
import improveai.decision_tracker as dt
import improveai.givens_provider as gp
from improveai.settings import DEBUG
from improveai.utils.general_purpose_tools import constant, check_variants, \
    get_variants_for_which


class DecisionModel:

    SUPPORTED_CALLS = ['score', 'top_scoring_variant', 'rank', 'get']
    MODEL_NAME_REGEXP = "^[a-zA-Z0-9][\w\-.]{0,63}$"

    @property
    def model_name(self) -> str:
        return self.__model_name

    @model_name.setter
    def model_name(self, value: str):

        if value is not None:
            assert isinstance(value, str)
            assert re.search(DecisionModel.MODEL_NAME_REGEXP, value) is not None
        self._model_name = value

    @property
    def _tracker(self):
        return self.__tracker

    @property
    def chooser(self) -> BasicChooser:
        return self._chooser

    @chooser.setter
    def chooser(self, value: BasicChooser):
        self._chooser = value

    @property
    def givens_provider(self):
        return self._givens_provider

    @givens_provider.setter
    def givens_provider(self, value):
        self._givens_provider = value

    @property
    def track_url(self):
        return self._track_url

    @track_url.setter
    def track_url(self, value):
        self._track_url = value

    @property
    def id_(self):
        return self._id_

    @id_.setter
    def id_(self, value):
        self._id_ = value

    @constant
    def TIEBREAKER_MULTIPLIER() -> float:
        return 2**-23

    def __init__(
            self, model_name: str, track_url: str = None, track_api_key: str = None):
        self.__set_model_name(model_name=model_name)
        self.track_url = track_url

        self.__tracker = None
        if self.track_url:
            self.__tracker = \
                dt.DecisionTracker(track_url=self.track_url, track_api_key=track_api_key)

        self.id_ = None
        self.chooser = None
        self.givens_provider = gp.GivensProvider()

    def __set_model_name(self, model_name: str):
        """
        Private helper method to set model name

        Parameters
        ----------
        model_name: str
            model name to be set
        """

        if model_name is not None:
            assert isinstance(model_name, str)
            assert re.search(DecisionModel.MODEL_NAME_REGEXP, model_name) is not None
        self.__model_name = model_name

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

        self.chooser = DecisionModel._get_chooser(model_url=model_url)
        self._resolve_model_name()

        return self

    def _resolve_model_name(self):
        """
        If model name is not set loaded model name is used, otherwise model name
        from constructor `wins`

        Returns
        -------

        """
        # TODO unittest this
        if self.model_name is None:
            self.__set_model_name(model_name=self.chooser.model_name)
        else:
            self.chooser.model_name = self.model_name
            warnings.warn(
                'Model name passed to the constructor: {} will not be '
                'overwritten by loaded model name: {}.'
                .format(self.model_name, self.chooser.model_name))

    # TODO check if it should be static or not
    # TODO remove mlmodel support
    # TODO make sure what a callback should do (probably based on iOS implementation)
    @staticmethod
    def _get_chooser(
            model_url: str, loop: asyncio.BaseEventLoop = None) -> BasicChooser:
        """
        Synchronously loads XGBoost model from provided path, creates instance
        of Chooser and returns it

        Parameters
        ----------
        model_url: str
            path to desired model (FS path or url)

        Returns
        -------
        DecisionModel
            new instance of decision model

        """

        true_exception = None
        try:
            model_src = BasicChooser.get_model_src(model_src=model_url)
        except Exception as exc:
            if loop is not None:
                loop.call_exception_handler(context={
                    'message': 'Failed to load model from url: {}'.format(
                        model_url),
                    'exception': exc})
            raise exc

        chooser = None
        chooser_constructors = [MLModelChooser, NativeXGBChooser]

        for chooser_constructor in chooser_constructors:

            chooser = None

            try:
                chooser = chooser_constructor()
                chooser.load_model(input_model_src=model_src)
                break

            except Exception as exc:

                true_exception = exc

        if (chooser is None or chooser.model is None) and loop is not None:
            loop.call_exception_handler(context={
                'message': 'Failed to load model from url: {}'.format(model_url),
                'exception': true_exception})

        return chooser

    def _load_chooser_for_async(
            self, model_url: str, loop: asyncio.BaseEventLoop = None):
        """
        Wrapper around chooser loading procedure for async operation

        Parameters
        ----------
        model_url: str
            path / url to model
        loop: object
            asyncio loop

        Returns
        -------

        """

        self.chooser = DecisionModel._get_chooser(model_url=model_url, loop=loop)
        self._resolve_model_name()

    @staticmethod
    def _exception_while_loading_chooser(loop, context):
        """
        Custom exception handler for asyncio

        Parameters
        ----------
        loop: object
            asyncio loop
        context: dict
            dict with info about error

        Returns
        -------

        """
        print(
            'Model loading failed with error: {}'.format(context.get('message', None)))
        print(context.get('exception', None))

    def load_async(self, model_url: str):
        """
        Loads model im an async fashion;

        IMPORTANT:
        Please note that this is an EXPERIMENTAL/DEPRECATED method and might be
        changed in the near future

        Parameters
        ----------
        model_url: str
            path / url to model

        Returns
        -------

        """
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(DecisionModel._exception_while_loading_chooser)
        loop.run_in_executor(None, self._load_chooser_for_async, *[model_url, loop])

        return self

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
        # TODO make sure how givens should be created
        # in iOS the call is almost identical (nil is passed to `givens` parameter of self.givens_provider.givens())
        # TODO [CHECK] how does this correspond with DecisionContext's givens?
        givens = self.givens_provider.givens(for_model=self)
        # return equivalent of double scores
        return self._score(variants=variants, givens=givens)

    def _score(self, variants: list or np.ndarray, givens: dict) -> list or np.ndarray:
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
        # TODO should chooser be settable from the outside ?
        if self.chooser is None:  # add async support
            return DecisionModel._generate_descending_gaussians(count=len(variants))

        try:
            scores = \
                self.chooser.score(
                    variants=variants, givens=givens) + \
                np.array(np.random.rand(len(variants)), dtype='float64') * \
                self.TIEBREAKER_MULTIPLIER
        except Exception as exc:
            # TODO test this scenario
            warnings.warn(
                'Error when calculating predictions: {}. Returning Gaussian scores'
                .format(exc))
            scores = DecisionModel._generate_descending_gaussians(count=len(variants))
        return scores.astype(np.float64)

    @staticmethod
    def _validate_variants_and_scores(
            variants: list or np.ndarray, scores: list or np.ndarray) -> bool:
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
            raise ValueError('`variants` and `scores` can`t be None')

        if len(variants) != len(scores):
            raise ValueError('Lengths of `variants` and `scores` mismatch!')

        if len(variants) == 0:
            # TODO is this desired ?
            return False

        return True

    @staticmethod
    def top_scoring_variant(
            variants: list or np.ndarray, scores: list or np.ndarray) -> dict:
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
        dict
            Returns best variant

        """

        assert variants is not None and scores is not None
        # null if no variants for get() implementation
        if not DecisionModel._validate_variants_and_scores(
                variants=variants, scores=scores):
            warnings.warn('The variants of length 0 were provided. Returning None')
            return None

        return variants[np.argmax(scores)]

    @staticmethod
    def rank(
            variants: list or np.ndarray, scores: list or np.ndarray) -> list or np.ndarray:
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
        # null if no variants for get() implementation
        if not DecisionModel._validate_variants_and_scores(
                variants=variants, scores=scores):
            return None

        variants_w_scores = np.array([variants, scores], dtype=object).T

        sorted_variants_w_scores = \
            variants_w_scores[(variants_w_scores[:, 1] * -1).argsort()]
        return sorted_variants_w_scores[:, 0]

    @staticmethod
    def _generate_descending_gaussians(count: int) -> list or np.ndarray:
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

    def given(self, givens: dict or None) -> dc.DecisionContext:  # returns DecisionContext?
        """
        Wrapper for chaining.

        Parameters
        ----------
        givens: dict
            givens to be set

        Returns
        -------

        """

        return dc.DecisionContext(decision_model=self, givens=givens)
        # return d.Decision(decision_model=self).given(givens=givens)

    def choose_from(self, variants: list):
        """
        Wrapper for chaining

        Parameters
        ----------
        variants: np.ndarray
            variants to be set

        Returns
        -------
        Decision
            decision object with provided with already scored variants and best selected

        """
        check_variants(variants=variants)
        # return d.Decision(decision_model=self).choose_from(variants=variants)
        # TODO how about the givens? Should GivensProvider be used here?
        #  in iOS here givens are set to nil
        return dc.DecisionContext(decision_model=self, givens=None).choose_from(variants=variants)

    def add_reward(self, reward: float):
        # void addReward(double reward)
        # Add rewards for the most recent Decision for this model name, even if
        # that Decision occurred in a previous session. Sets model on the reward
        # record to be equal to the current modelName.
        # NaN, positive infinity, and negative infinity are not allowed and should throw exceptions

        assert self.model_name is not None and self.id_ is not None
        assert isinstance(reward, float) or isinstance(reward, int)
        assert reward is not None
        assert not np.isnan(reward)
        assert not np.isinf(reward)

        if self.__tracker is not None:
            return self.__tracker.add_reward(
                reward=reward, model_name=self.model_name, decision_id=self.id_)
        else:
            if self.__tracker is None:
                warnings.warn(
                    '`tracker` is not set (`tracker`is None) - reward not added')
            if self.track_url is None:
                warnings.warn('`track_url` is None - reward not added')

    def which(self, *variants):
        """
        A short hand version of chooseFrom that returns the chosen result directly,
        automatically tracking the decision in the process. This would be a varargs
        version of chooseFrom that would also take an array as the only argument.
        If the only argument was not an array it would throw an exception.
        Since the Decision object is not returned, rewards would only be tracked
        through DecisionModel.addReward

        Parameters
        ----------
        *variants: list or tuple or np.ndarray
            array of variants passed as positional parameters

        Returns
        -------
        d.Decision
            snapshot of the Decision made with provided variants and available givens

        """
        return self.choose_from(variants=get_variants_for_which(variants=variants)).get()
