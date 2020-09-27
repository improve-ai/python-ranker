from abc import ABC, abstractmethod


class BasicChooser(ABC):
    @property
    @abstractmethod
    def model(self) -> object:
        return self._model

    @model.setter
    @abstractmethod
    def model(self, new_val: object):
        self._model = new_val

    @abstractmethod
    def load_model(self, pth_to_model, **kwargs):
        pass

    @abstractmethod
    def score(self, variant, context, **kwargs):
        pass

    @abstractmethod
    def score_all(self, variants, context, **kwargs):
        pass

    @abstractmethod
    def sort(self, variants_w_scores, **kwargs):
        pass

    @abstractmethod
    def choose(self, variants_w_scores, **kwargs):
        pass
