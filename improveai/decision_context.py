import numpy as np

import improveai.decision as d
import improveai.decision_model as dm
from improveai.utils.general_purpose_tools import check_variants, get_variants_from_args


class DecisionContext:

    @property
    def decision_model(self):
        """
        DecisionModel of this DecisionContext

        Returns
        -------
        DecisionModel
            DecisionModel of this DecisionContext

        """
        return self._decision_model

    @decision_model.setter
    def decision_model(self, value):
        # DecisionContext's model must not be None and of desired type
        assert value is not None
        assert isinstance(value, dm.DecisionModel)
        self._decision_model = value

    @property
    def givens(self) -> dict:
        """
        Givens for this DecisionContext

        Returns
        -------
        dict
            Givens for this DecisionContext

        """
        return self._givens

    @givens.setter
    def givens(self, value):
        # givens can be None or dict
        assert isinstance(value, dict) or value is None
        self._givens = value

    def __init__(self, decision_model, givens: dict or None):
        """
        Init with params

        Parameters
        ----------
        decision_model: DecisionModel
            decision model for this DecisionContext
        givens: dict
            givens for this DecisionContext
        """
        self.decision_model = decision_model
        self.givens = givens

    def score(self, variants) -> np.ndarray:
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
        givens = self.decision_model.givens_provider.givens(for_model=self.decision_model, givens=self.givens)
        return self.decision_model._score(variants=variants, givens=givens)

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
        *variants: list or tuple or np.ndarray
            array of variants passed as positional parameters

        Returns
        -------
        object, str
            a tuple of (<best variant>, <decision id>)

        """

        return self.which_from(variants=get_variants_from_args(variants))

    def which_from(self, variants: list or np.ndarray) -> tuple:
        """
        Makes a decision for provided variants and DecisionContext givens.

        Parameters
        ----------
        variants: list or np.ndarray

        Returns
        -------
        object, str
            a tuple of (<best variant>, <decision id>)

        """
        decision = self.decide(variants=variants, track=True)
        return decision.get(), decision.id_

    def decide(self, variants: list or np.ndarray, scores: list or np.ndarray = None,
               ordered: bool = False, track: bool = False) -> d.Decision:
        """
        Creates decision but does not track it by default.
        If not scores are provided and input variants are not ranked performs
        scoring and ranking. If scores are provided but variants are not ordered
        ranks variants before making a decision

        Parameters
        ----------
        variants: list or np.ndarray
            variants for the decision
        scores: list or np.ndarray
            scores for variants, nullable
        ordered: bool
            flag indicating if input variants are ordered
        track: bool
            should decision be tracked

        Returns
        -------
        d.Decision
            decision object created for input data

        """

        # check variants
        check_variants(variants)
        # get givens via GivensProvider
        givens = self.decision_model.givens_provider.givens(for_model=self.decision_model, givens=self.givens)

        # rank variants if they are not ordered
        ranked_variants = variants
        if not ordered:

            if scores is None:
                scores = self.decision_model._score(variants=variants, givens=givens)

            assert scores is not None
            ranked_variants = self.decision_model._rank(variants=variants, scores=scores)

        assert len(scores) == len(variants)

        decision = d.Decision(
            decision_model=self.decision_model, ranked_variants=ranked_variants, givens=givens)

        if track:
            decision._track()

        return decision

    def rank(self, variants: list or np.ndarray) -> tuple:
        """
        Ranks input variants, creates decision from them and tracks it

        Parameters
        ----------
        variants: list or np.ndarray
            variants to be ranked

        Returns
        -------
        list or np.ndarray, str
            ranked variants and decision ID

        """
        decision = self.decide(variants=variants, track=True)
        return decision.ranked(), decision.id_

    def optimize(self):
        # TODO implement
        pass

    def choose_first(self, variants: list or tuple or np.ndarray) -> d.Decision:
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

    def first(self, *variants) -> tuple:
        """
        Makes decision using first variant as best and tracks it.
        Accepts variants as pythonic *args

        Parameters
        ----------
        variants: list or tuple or np.ndarray
            collection of variants of which first will be chosen

        Returns
        -------
        object, str
            a tuple of (<first variant>, <decision id>)
        """

        # make decision and track immediately
        decision = self.decide(
            variants=get_variants_from_args(variants), ordered=True, track=True)
        # return best and decision ID
        return decision.get(), decision.id_

    def choose_random(self, variants: list or tuple or np.ndarray) -> d.Decision:
        """
        Shuffles variants to return Decision with gaussian scores and random best variant

        Parameters
        ----------
        variants: list or tuple or np.ndarray
            collection of variants of which random will be chosen

        Returns
        -------
        Decision
            Decision with randomly chosen best variant
        """
        return self.decide(variants, scores=np.random.normal(size=len(variants)))

    def random(self, *variants) -> tuple:
        """
        Makes decision using randomly selected variant as best and tracks it.
        Accepts variants as pythonic *args

        Parameters
        ----------
        variants: list or np.ndarray
            collection of variants of which first will be chosen

        Returns
        -------
        object, str
            a tuple of (<random variant>, <decision id>)
        """
        decision = self.decide(
            variants=get_variants_from_args(variants), scores=np.random.normal(size=len(variants)), track=True)
        return decision.get(), decision.id_

    def choose_from(
            self, variants: list or tuple or np.ndarray, scores: np.ndarray or list or None) -> d.Decision:
        """
        Makes a Decision without tracking it

        Parameters
        ----------
        variants: list or tuple or np.ndarray
            array of variants to choose from
        scores: np.ndarray or list or None
            list of scores for variants

        Returns
        -------
        Decision
            snapshot of the Decision made with provided variants and available givens
        """
        return self.decide(variants=variants, scores=scores)
