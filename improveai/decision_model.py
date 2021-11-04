import asyncio
import os
import re
import signal
import warnings

import numpy as np

from improveai.choosers.basic_choosers import BasicChooser
from improveai.choosers.mlmodel_chooser import BasicMLModelChooser
from improveai.choosers.xgb_chooser import BasicNativeXGBChooser
import improveai.decision as d
import improveai.decision_tracker as dt
from improveai.utils.general_purpose_tools import constant


class DecisionModel:

    SUPPORTED_CALLS = ['score', 'top_scoring_variant', 'rank', 'get']
    MODEL_NAME_REGEXP = "^[a-zA-Z0-9][\w\-.]{0,63}$"

    @property
    def model_name(self) -> str:
        return self._model_name

    @model_name.setter
    def model_name(self, value: str):
        # TODO if it will be decided that model_name can ne None refactor
        assert value is not None
        # if value is not None:
        assert isinstance(value, str)
        assert re.search(DecisionModel.MODEL_NAME_REGEXP, value) is not None
        self._model_name = value

    @property
    def tracker(self):
        return self._tracker

    @tracker.setter
    def tracker(self, value):
        self._tracker = value

    @property
    def chooser(self) -> BasicChooser:
        return self._chooser

    @chooser.setter
    def chooser(self, value: BasicChooser):
        self._chooser = value

    @constant
    def TIEBREAKER_MULTIPLIER() -> float:
        return 2**-23

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tracker = None
        self.chooser = None

    def track_with(self, tracker):

        if not isinstance(tracker, dt.DecisionTracker):
            raise TypeError('`tracker` should be an instance of DecisionTracker')

        self.tracker = tracker
        return self

    def _set_chooser(self, chooser: BasicChooser):
        """
        Sets chooser

        Parameters
        ----------
        chooser: BasicChooser
            chooser object to be used from within DecisionModel

        Returns
        -------

        """
        self.chooser = chooser
        # At this point mode_name is set otherwise error would be thrown
        if self.model_name != self.chooser.model_name \
                and self.model_name is not None:
            warnings.warn(
                '`model_name` passed to contructor does not match loaded '
                'model`s name -  using loaded model`s name')

        self.model_name = self.chooser.model_name

    @staticmethod
    def load(model_url: str):
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

        decision_model = DecisionModel(model_name='loading')
        decision_model._set_chooser(
            chooser=DecisionModel._get_chooser(model_url=model_url))

        return decision_model

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
        chooser_constructors = [BasicMLModelChooser, BasicNativeXGBChooser]

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

        self._set_chooser(
            chooser=DecisionModel._get_chooser(model_url=model_url, loop=loop))

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
            'Model loading failed with error: {}'.format(
                context.get('message', None)))
        print(context.get('exception', None))

        os.kill(os.getpid(), signal.SIGKILL)

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
        loop.run_in_executor(
            None, self._load_chooser_for_async, *[model_url, loop])

        return self

    def score(
            self, variants: list or np.ndarray, givens: dict) -> list or np.ndarray:
        """
        Call predict and calculate scores for provided variants

        Parameters
        ----------
        variants: list or np.ndarray
            collection of variants to be scored
        givens: dict
            context to calculating scores

        Returns
        -------
        np.ndarray
            scores

        """

        # TODO should chooser be settable from the outside ?
        if not self.chooser:  # add async support
            return DecisionModel.generate_descending_gaussians(
                count=len(variants))

        scores = \
            self.chooser.score(variants=variants, givens=givens) + \
            np.array(
                np.random.rand(len(variants)), dtype='float64') * \
            self.TIEBREAKER_MULTIPLIER
        return scores

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

        if not DecisionModel._validate_variants_and_scores(
                variants=variants, scores=scores):
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

        if not DecisionModel._validate_variants_and_scores(
                variants=variants, scores=scores):
            return None

        variants_w_scores = np.array([variants, scores], dtype=object).T

        sorted_variants_w_scores = \
            variants_w_scores[(variants_w_scores[:, 1] * -1).argsort()]
        return sorted_variants_w_scores[:, 0]

    @staticmethod
    def generate_descending_gaussians(count: int) -> list or np.ndarray:
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

    def given(self, givens: dict) -> object:  # returns Decision
        """
        Wrapper for chaining.

        Parameters
        ----------
        givens: dict
            givens to be set

        Returns
        -------

        """
        return d.Decision(decision_model=self).given(givens=givens)

    def choose_from(self, variants: list) -> object:
        """
        Wrapper for chaining

        Parameters
        ----------
        variants: np.ndarray
            variants to be set

        Returns
        -------

        """
        return d.Decision(decision_model=self).choose_from(variants=variants)
