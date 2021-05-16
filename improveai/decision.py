from collections.abc import Iterable
import numpy as np
from typing import Dict, List
from warnings import warn

import improveai.model as dm


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

    def __init__(self, decision_model: object):

        if decision_model is None:
            raise ValueError('`decision_model` can`t be None')

        self.__model = decision_model
        self.__chosen = False

        self.__variants = [None]
        self.__givens = None
        self.__memoized_variant = None

        self.__variants__set = False
        self.__givens_set = False

    def given(self, givens: dict):

        if not isinstance(givens, dict):
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

        if variants is None or variants == [None]:
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

        if self.chosen:
            warn('The best variant has already been chosen')
            return self

        if self.__variants__set:
            warn('`variants` have already been set - ignoring this call')
            return self

        self.__variants = variants
        self.__variants__set = True

        return self

    def get(self):

        if self.chosen:
            warn('The best variant has already been chosen')
            return self.memoized_variant

        scores = self.model.score(variants=self.variants, givens=self.givens)
        # TODO is that needed
        # ranked_variants = None

        # TODO make sure this is bit is executed only once per Decision's
        #  lifetime
        if self.variants is not None and len(self.variants) != 0:
            # there should be no difference between effect of those 2 conditions
            # since this  clause is reached only once
            # if self.model.tracker and not self.tracked:
            if self.model.tracker:

                if self.model.tracker.should_track_runners_up(len(self.variants)):
                    ranked_variants = \
                        dm.DecisionModel.rank(
                            variants=self.variants, scores=scores)
                    self.__memoized_variant = ranked_variants[0]

                    self.model.tracker.track(
                        variant=self.memoized_variant, variants=ranked_variants,
                        givens=self.givens, model_name=self.model.model_name,
                        variants_ranked_and_track_runners_up=True)
                else:
                    self.__memoized_variant = \
                        dm.DecisionModel.top_scoring_variant(
                            variants=self.variants, scores=scores)

                    self.model.tracker.track(
                        variant=self.memoized_variant, variants=self.variants,
                        givens=self.givens, model_name=self.model.model_name,
                        variants_ranked_and_track_runners_up=False)

            else:
                self.__memoized_variant = \
                    dm.DecisionModel.top_scoring_variant(
                        variants=self.variants, scores=scores)

        self.__chosen = True
        return self.memoized_variant
