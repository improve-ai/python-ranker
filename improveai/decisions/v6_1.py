import numpy as np
from typing import Dict, List

import models.v6_1 as dm
# from models.v6_1 import DecisionModel


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

    def __init__(self, decision_model):
        self.__model = decision_model
        self.__chosen = False

        self.__variants = None
        self.__givens = None
        self.__memoized_variant = None

    def given(self, givens: dict):

        if self.chosen:
            print('The best variant has already been chosen')
            return self

        self.__givens = givens
        return self

    def choose_from(self, variants: list or np.ndarray):
        if self.chosen:
            print('The best variant has already been chosen')
            return self

        self.__variants = variants
        return self

    def get(self):
        if self.chosen:
            print('The best variant has already been chosen')
            return self.memoized_variant

        scores = self.model.score(variants=self.variants, givens=self.givens)
        # TODO is that needed
        # ranked_variants = None

        # TODO make sure this is bit is executed only once per Decision's
        #  lifetime
        if len(self.variants) != 0:
            # there should be no difference between effect of those 2 conditions
            # since this  clause is reached only once
            # if self.model.tracker and not self.tracked:
            if self.model.tracker:

                if self.model.tracker.should_track_runners_up(len(self.variants)):
                    ranked_variants = \
                        dm.DecisionModel.rank(variants=self.variants, scores=scores)
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
