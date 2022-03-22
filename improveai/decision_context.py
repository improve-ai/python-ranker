import numpy as np

import improveai.decision as d
import improveai.decision_model as dm
import improveai.givens_provider as gp
from improveai.utils.general_purpose_tools import check_variants, get_variants_for_which


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

    # TODO should givens be a part of constructor call (they are in iOS im implementation)
    def __init__(self, decision_model, givens: dict or None):
        self.decision_model = decision_model
        self.givens = givens

    def choose_from(self, variants):
        """
        Makes a Decision without tracking it

        Parameters
        ----------
        variants: list or tuple or np.ndarray
            array of variants to choose from

        Returns
        -------
        d.Decision
            snapshot of the Decision made with provided variants and available givens

        """
        check_variants(variants=variants)
        # givens must be provided at this point -> they are needed for decision snapshot
        givens = gp.GivensProvider().givens(for_model=self.decision_model, givens=self.givens)
        # calculate scores (with _score() method) -> givens are provided inside it
        # scores will also be cached to decision object
        scores = self.decision_model._score(variants=variants, givens=givens)
        # calculate the best variant without tracking (calling get())
        best = self.decision_model.top_scoring_variant(variants=variants, scores=scores)
        # cache all info to decision object
        decision = d.Decision(decision_model=self.decision_model)
        decision.variants = variants
        decision.givens = givens
        decision.scores = scores
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
        # TODO how givens should be provided here?
        # I assumed that this should be done like in iOS SDK
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
        d.Decision
            snapshot of the Decision made with provided variants and available givens

        """
        return self.choose_from(variants=get_variants_for_which(variants=variants)).get()
