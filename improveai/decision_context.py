from improveai import DecisionModel


class DecisionContext:

    @property
    def decision_model(self):
        return self._decision_model

    @decision_model.setter
    def decision_model(self, value):
        assert value is not None
        assert isinstance(value, DecisionModel)
        self._decision_model = value

    def __init__(self, decision_model):
        self.decision_model = decision_model

    def choose_from(self, variants):
        return self.decision_model.choose_from(variants)

    def given(self, givens: dict):
        return self.decision_model.given(givens)
