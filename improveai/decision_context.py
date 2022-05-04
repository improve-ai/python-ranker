import numpy as np

import improveai.decision as d
import improveai.decision_model as dm
import improveai.givens_provider as gp
from improveai.utils.general_purpose_tools import check_variants, get_variants_from_args


class DecisionContext:

    @property
    def decision_model(self):
        return self._decision_model

    @decision_model.setter
    def decision_model(self, value):
        # DecisionContext's model must not be None and of desired type
        assert value is not None
        assert isinstance(value, dm.DecisionModel)
        self._decision_model = value

    @property
    def givens(self):
        return self._givens

    @givens.setter
    def givens(self, value):
        # givens can be None or dict
        assert isinstance(value, dict) or value is None
        self._givens = value

    def __init__(self, decision_model, givens: dict or None):
        self.decision_model = decision_model
        self.givens = givens

    def choose_from(self, variants, scores: np.ndarray or list or None = None):
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

        check_variants(variants=variants)
        # givens must be provided at this point -> they are needed for decision snapshot
        givens = gp.GivensProvider().givens(for_model=self.decision_model, givens=self.givens)
        # calculate scores (with _score() method) -> givens are provided inside it
        # if scores are provided
        scores_for_decision = scores
        if scores_for_decision is None:
            # calculate scores
            # scores will also be cached to decision object
            scores_for_decision = self.decision_model._score(variants=variants, givens=givens)

        # calculate the best variant without tracking (calling get())
        best = self.decision_model.top_scoring_variant(variants=variants, scores=scores_for_decision)
        # cache all info to decision object
        decision = d.Decision(decision_model=self.decision_model)
        decision.variants = variants
        decision.givens = givens
        decision.scores = scores_for_decision
        decision.best = best

        return decision

    def score(self, variants):
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
        givens = gp.GivensProvider().givens(for_model=self.decision_model, givens=self.givens)
        return self.decision_model._score(variants=variants, givens=givens)

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
        Decision
            snapshot of the Decision made with provided variants and available givens

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
        d.Decision
            A decision with first variants as the best one and gaussian scores

        """

        check_variants(variants=variants)
        return self.choose_from(
            variants, scores=self.decision_model.generate_descending_gaussians(count=len(variants)))

    def first(self, *variants):
        """
        Makes decision using first variant as best and tracks it.
        Accepts variants as pythonic *args

        Parameters
        ----------
        variants: list or tuple or np.ndarray
            collection of variants of which first will be chosen

        Returns
        -------
        object
            chosen and tracked variant

        """

        check_variants(variants=variants)
        decision = self.choose_first(variants=get_variants_from_args(variants))
        best = decision.get()
        return best, decision.id_

    def choose_random(self, variants: list or tuple or np.ndarray):
        """
        Shuffles variants to return Decision with gaussian scores and random best variant

        Parameters
        ----------
        variants: list or tuple or np.ndarray
            collection of variants of which random will be chosen

        Returns
        -------
        d.Decision
            Decision with randomly chosen best variant

        """
        check_variants(variants=variants)
        count = len(variants)
        return self.choose_from(variants, scores=np.random.normal(size=count))

    def random(self, *variants):
        """
        Makes decision using randomly selected variant as best and tracks it.
        Accepts variants as pythonic *args

        Parameters
        ----------
        variants: list or np.ndarray
            collection of variants of which first will be chosen

        Returns
        -------
        object
            randomly selected and tracked best variant

        """
        used_variants = get_variants_from_args(variants)
        count = len(used_variants)
        decision = self.choose_from(used_variants, scores=np.random.normal(size=count))
        best = decision.get()
        return best, decision.id_
