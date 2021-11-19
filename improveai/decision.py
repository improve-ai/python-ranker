from collections.abc import Iterable
from ksuid import Ksuid
import numpy as np
from typing import Dict, List
from warnings import warn

import improveai.decision_model as dm


class Decision:

    @property
    def variants(self) -> List[Dict[str, object]] or None:
        return self.__variants

    @property
    def givens(self) -> List[Dict[str, object]] or None:
        return self.__givens

    @property
    def model(self) -> object:
        return self.__model

    @property
    def chosen(self):
        return self.__chosen

    @property
    def memoized_variant(self) -> dict or None:
        return self.__memoized_variant

    @property
    def scores(self):
        return self.__scores

    @property
    def ranked_variants(self):
        return self.__ranked_variants

    @property
    def id_(self):
        return self.__id_

    def __init__(self, decision_model: object):

        if decision_model is None:
            raise ValueError('`decision_model` can`t be None')

        self.__model = decision_model
        self.__chosen = False

        self.__variants = [None]
        self.__givens = None
        self.__memoized_variant = None
        self.__scores = None
        self.__ranked_variants = [None]
        self.__id_ = None

        self.__variants__set = False
        self.__givens_set = False

    def given(self, givens: dict):

        # TODO can given be None?
        if not isinstance(givens, dict) and givens is not None:
            raise TypeError('`givens` should be a dict')

        if self.chosen:
            warn('The best variant has already been chosen')
            return self

        if self.__givens_set:
            warn('`givens` have already been set - ignoring this call')
            return self

        self.__givens = givens
        self.__givens_set = True

        return self

    def choose_from(self, variants: list or np.ndarray):

        if self.chosen:
            warn('The best variant has already been chosen')
            return self

        if self.__variants__set:
            warn('`variants` have already been set - ignoring this call')
            return self

        # TODO is numpy array check necessary ? Maybe it is a waste of time
        if variants is None or variants == [None] or variants == [] or \
                (isinstance(variants, np.ndarray) and variants.size == 0):
            return self

        if isinstance(variants, str) or isinstance(variants, dict):
            raise TypeError(
                '`variants` should be a collection of some sort '
                '(list, np.ndarray, set, tuple) not a: {}'
                .format(type(variants)))
        elif not isinstance(variants, Iterable):
            raise TypeError(
                '`variants` should be a collection of some sort '
                '(list, np.ndarray, set, tuple) not a: {}'
                .format(type(variants)))

        self.__variants = variants
        self.__variants__set = True

        return self

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
        self.model.id_ = self.id_

    def add_reward(self, reward: float or int):

        if not self.chosen:
            warn('`add_reward()` called before `get()`')
        # TODO add reward

        assert self.model.id_ == self.id_
        assert self.model.model_name is not None and self.model.id_ is not None
        assert reward is not None
        assert isinstance(reward, float) or isinstance(reward, int)
        assert not np.isnan(reward)
        assert not np.isinf(reward)

        # TODO check if decision_id is set correctly
        return self.model.tracker.add_reward(
            reward=reward, model_name=self.model.model_name, decision_id=self.id_)

    def get(self):

        if self.chosen:
            warn('The best variant has already been chosen')
            self._cache_message_id_to_decision_model()
            return self.memoized_variant

        # set message_id / decision_id only once
        self._set_message_id()
        self._cache_message_id_to_decision_model()

        # set message_id / deicsion_id to decision model

        # TODO should scores be persisted inside Decision object
        self.__scores = self.model.score(variants=self.variants, givens=self.givens)
        # TODO is that needed
        # ranked_variants = None

        # TODO make sure this is bit is executed only once per Decision's
        #  lifetime ?
        if self.variants is not None and len(self.variants) != 0:
            # there should be no difference between effect of those 2 conditions
            # since this  clause is reached only once
            # if self.model.tracker and not self.tracked:
            if self.model.tracker:

                # TODO should ranked_variants be persisted inside Decision object
                track_runners_up = self.model.tracker.should_track_runners_up(len(self.variants))
                if track_runners_up:
                    # # TODO should ranked_variants be persisted inside Decision object
                    self.__ranked_variants = \
                        dm.DecisionModel.rank(
                            variants=self.variants, scores=self.__scores)
                    self.__memoized_variant = self.ranked_variants[0]

                    self.model.tracker.track(
                        variant=self.memoized_variant, variants=self.ranked_variants,
                        givens=self.givens, model_name=self.model.model_name,
                        variants_ranked_and_track_runners_up=True, message_id=self.id_)
                else:
                    self.__memoized_variant = \
                        dm.DecisionModel.top_scoring_variant(
                            variants=self.variants, scores=self.__scores)

                    self.model.tracker.track(
                        variant=self.memoized_variant, variants=self.variants,
                        givens=self.givens, model_name=self.model.model_name,
                        variants_ranked_and_track_runners_up=False, message_id=self.id_)

            elif self.model.tracker is None:
                raise ValueError('`tracker` object can`t be None')
            else:
                self.__memoized_variant = \
                    dm.DecisionModel.top_scoring_variant(
                        variants=self.variants, scores=self.__scores)

        self.__chosen = True
        return self.memoized_variant
