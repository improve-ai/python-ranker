import improveai.decision as d
import improveai.decision_model as dm
from improveai.givens_provider import GivensProvider
from improveai.utils.general_purpose_tools import check_variants


class DecisionContext:

    @property
    def decision_model(self):
        return self._decision_model

    @decision_model.setter
    def decision_model(self, value):
        assert value is not None
        assert isinstance(value, dm.DecisionModel)
        self._decision_model = value

    @property
    def givens(self):
        return self._givens

    @givens.setter
    def givens(self, value):
        assert isinstance(value, dict) or value is None
        self._givens = value

    # TODO should givens be a part of constructor call (they are in iOS im implementation)
    def __init__(self, decision_model, givens: dict or None):
        self.decision_model = decision_model
        self.givens = givens

    def choose_from(self, variants):
        check_variants(variants=variants)

        # givens must be provided at this point -> they are needed for decision snapshot
        givens = GivensProvider().givens(for_model=self.decision_model, givens=self.givens)
        # calculate scores (with _score() method) -> givens are provided inside it
        # scores will also be cached to decision object
        scores = self.decision_model._score(variants=variants, givens=givens)
        # calculate best variant without tracking (calling get())
        best = self.decision_model.top_scoring_variant(variants=variants, scores=scores)
        # cache all info to decision object
        decision = d.Decision(decision_model=self.decision_model)
        decision.variants = variants
        decision.givens = givens
        decision.scores = scores
        decision.best = best

        return decision

    def score(self, variants):
        # TODO how givens should be provided here?
        # I assumed that this should be done like in iOS SDK
        givens = GivensProvider().givens(for_model=self.decision_model, givens=self.givens)
        return self.decision_model._score(variants=variants, givens=givens)

    def which(self):
        # TODO implement
        pass
