from abc import ABC, abstractmethod


class BasicChooser(ABC):
    @abstractmethod
    @property
    def usd_model(self) -> object:
        return self._usd_model

    @abstractmethod
    @usd_model.setter
    def usd_model(self, new_val: object):
        self._usd_model = new_val

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
