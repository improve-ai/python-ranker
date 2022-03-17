from ksuid import Ksuid
import numpy as np
from warnings import warn

import improveai.decision_model as dm
from improveai.utils.general_purpose_tools import check_variants


class Decision:

    @property
    def variants(self) -> list or np.ndarray or tuple:
        return self.__variants

    @variants.setter
    def variants(self, value: list or np.ndarray or tuple):
        # TODO test variants setter behaviour
        if self.__variants_set is False:
            check_variants(variants=value)
            self.__variants = value
            self.__variants_set = True

    @property
    def givens(self) -> dict or None:
        return self.__givens

    @givens.setter
    def givens(self, value: dict):
        # TODO test givens setter behaviour
        if self.__givens_set is False:
            assert isinstance(value, dict) or value is None
            self.__givens = value
            self.__givens_set = True

    @property
    def decision_model(self) -> object:
        return self.__decision_model

    @property
    def chosen(self):
        return self.__chosen

    @property
    def best(self):
        return self.__best

    @best.setter
    def best(self, value):
        # TODO test best setter behaviour (previously memoized_variant)
        if self.__best_set is False:
            self.__best = value
            self.__best_set = True

    @property
    def scores(self):
        return self.__scores

    @scores.setter
    def scores(self, value):
        # TODO test scores setter behaviour
        if self.__scores_set is False:
            self.__scores = value
            self.__scores_set = True

    @property
    def ranked_variants(self):
        return self.__ranked_variants

    @property
    def id_(self):
        return self.__id_

    def __init__(self, decision_model: object):

        if decision_model is None:
            raise ValueError('`decision_model` can`t be None')

        self.__decision_model = decision_model
        self.__chosen = False

        self.__variants = [None]
        self.__variants_set = False

        self.__givens = None
        self.__givens_set = False

        self.__best = None
        self.__best_set = False

        self.__scores = None
        self.__scores_set = False

        self.__ranked_variants = [None]
        self.__id_ = None

    def _set_message_id(self):
        if self.__id_ is None:
            id_ = str(Ksuid())
            assert isinstance(id_, str)
            Ksuid.from_base62(id_)
            self.__id_ = id_
        else:
            warn('`id_` has already been set for this Decision')

    def _cache_message_id_to_decision_model(self):
        assert self.__id_ is not None
        self.decision_model.id_ = self.id_

    def add_reward(self, reward: float or int):

        if not self.chosen:
            warn('`add_reward()` called before `get()`')
        # TODO add reward

        assert self.decision_model.id_ == self.id_
        assert self.decision_model.model_name is not None and self.decision_model.id_ is not None
        assert reward is not None
        assert isinstance(reward, float) or isinstance(reward, int)
        assert not np.isnan(reward)
        assert not np.isinf(reward)

        # TODO check if decision_id is set correctly
        return self.decision_model._tracker.add_reward(
            reward=reward, model_name=self.decision_model.model_name, decision_id=self.id_)

    def get(self):

        if self.chosen:
            warn('The best variant has already been chosen')
            self._cache_message_id_to_decision_model()
            return self.best

        # set message_id / decision_id only once
        self._set_message_id()
        self._cache_message_id_to_decision_model()

        # set message_id / deicsion_id to decision model

        # TODO should scores be persisted inside Decision object
        self.__scores = self.decision_model._score(variants=self.variants, givens=self.givens)
        # TODO is that needed
        # ranked_variants = None

        # TODO make sure this is bit is executed only once per Decision's
        #  lifetime ?
        if self.variants is not None and len(self.variants) != 0:
            # there should be no difference between effect of those 2 conditions
            # since this  clause is reached only once
            if self.decision_model._tracker:

                # TODO should ranked_variants be persisted inside Decision object
                track_runners_up = self.decision_model._tracker._should_track_runners_up(len(self.variants))
                if track_runners_up:
                    # TODO should ranked_variants be persisted inside Decision object
                    self.__ranked_variants = \
                        dm.DecisionModel.rank(
                            variants=self.variants, scores=self.__scores)
                    self.__best = self.ranked_variants[0]

                    self.decision_model._tracker.track(
                        variant=self.best, variants=self.ranked_variants,
                        givens=self.givens, model_name=self.decision_model.model_name,
                        variants_ranked_and_track_runners_up=True, message_id=self.id_)
                else:
                    self.__best = \
                        dm.DecisionModel.top_scoring_variant(
                            variants=self.variants, scores=self.__scores)

                    self.decision_model._tracker.track(
                        variant=self.best, variants=self.variants,
                        givens=self.givens, model_name=self.decision_model.model_name,
                        variants_ranked_and_track_runners_up=False, message_id=self.id_)

            elif self.decision_model._tracker is None:
                raise ValueError('`tracker` object can`t be None')
            else:
                self.__best = \
                    dm.DecisionModel.top_scoring_variant(
                        variants=self.variants, scores=self.__scores)

        self.__chosen = True
        return self.best
