from ksuid import Ksuid
from numbers import Number
import numpy as np
from warnings import warn

import improveai.decision_model as dm
from improveai.utils.general_purpose_tools import check_variants, is_valid_ksuid


class Decision:

    @property
    def variants(self) -> list or np.ndarray or tuple:
        """
        Collection of variants used to make Decision; Can't be None

        Returns
        -------
        list or np.ndarray or tuple
            Collection of variants used to make Decision

        """
        return self.__variants

    @variants.setter
    def variants(self, value: list or np.ndarray or tuple):
        if self.__variants_set is False:
            check_variants(variants=value)
            self.__variants = value
            self.__variants_set = True
        else:
            warn('`variants` have already been set for this Decision')

    @property
    def givens(self) -> dict or None:
        """
        Givens used to make Decision; Can be None and {}

        Returns
        -------
        dict or None
            Givens used to make Decision

        """
        return self.__givens

    @givens.setter
    def givens(self, value: dict):
        if self.__givens_set is False:
            assert isinstance(value, dict) or value is None
            self.__givens = value
            self.__givens_set = True
        else:
            warn('`givens` have already been set for this Decision')

    @property
    def decision_model(self):
        """
        DecisionModel used to make Decision

        Returns
        -------
        DecisionModel
            DecisionModel used to make Decision

        """
        return self.__decision_model

    @property
    def chosen(self) -> bool:
        """
        A flag indicating if the best variant has already been chosen

        Returns
        -------
        bool
            A flag indicating if the best variant has already been chosen

        """
        return self.__chosen

    @property
    def tracked(self) -> bool:
        """
        A flag indicating if the Decision has already been tracked

        Returns
        -------
        bool
            A flag indicating if the Decision has already been tracked

        """
        return self.__tracked

    @property
    def best(self) -> object:
        """
       Best variant for this Decision

        Returns
        -------
        object
            Best variant for this Decision

        """
        return self.__best

    @best.setter
    def best(self, value):
        if self.__best_set is False:
            self.__best = value
            self.__best_set = True
            self.__chosen = True
        else:
            warn('`best` has already been calculated / set for this Decision')

    @property
    def scores(self) -> np.ndarray:
        """
        Scores calculated for this Decision

        Returns
        -------
        np.ndarray
            Scores calculated for this Decision

        """
        return self.__scores

    @scores.setter
    def scores(self, value):

        if self.__scores_set is False:
            assert value is not None
            assert (isinstance(value, list) or isinstance(value, np.ndarray))
            assert len(value) > 0
            assert all([isinstance(s, Number) or s is None for s in value])
            assert self.variants is not None
            assert len(value) == len(self.variants)

            self.__scores = value
            self.__scores_set = True
        else:
            warn('`scores` have already been calculated / set for this Decision')

    @property
    def ranked_variants(self) -> list or tuple or np.ndarray:
        """
        Variants ranked from best to worst (using scores)

        Returns
        -------
        list or tuple or np.ndarray
            Variants ranked from best to worst (using scores)

        """
        return self.__ranked_variants

    @property
    def id_(self):
        """
        ID (ksuid) if this decision

        Returns
        -------
        str
            ID of this Decision

        """
        return self.__id_

    def __init__(self, decision_model: object, ranked_variants: list, givens: dict or None):
        """
        Init with params

        Parameters
        ----------
        decision_model: DecisionModel
            DecisionModel for this Decision
        """

        assert isinstance(decision_model, dm.DecisionModel)

        self.__decision_model = decision_model
        self.__chosen = False
        self.__tracked = False

        # self.__variants = [None]
        # self.__variants_set = False

        # self.__givens = None
        # self.__givens_set = False
        self.__givens = givens
        self.__givens_set = True

        self.__best = None
        self.__best_set = False

        self.__scores = None
        self.__scores_set = False

        # self.__ranked_variants = [None]
        self.__ranked_variants = ranked_variants
        self.__variants_already_ranked = False
        self.__id_ = None

    def _set_message_id(self):
        """
        Sets Decision's message ID (once per lifetime)

        Returns
        -------
        None
            None

        """
        if self.__id_ is None:
            id_ = str(Ksuid())
            assert isinstance(id_, str)
            Ksuid.from_base62(id_)
            self.__id_ = id_
        else:
            warn('`id_` has already been set for this Decision')

    def add_reward(self, reward: float or int):
        """
        Adds provided reward to this Decision. Reward must be a number, must not be None, nan or +- inf

        Parameters
        ----------
        reward: float
            reward to be added

        Returns
        -------
        str
            reward request message ID

        """

        if not self.chosen:
            warn('`add_reward()` called before `get()`')

        assert is_valid_ksuid(self.id_)
        assert self.decision_model.model_name is not None
        assert reward is not None
        assert isinstance(reward, float) or isinstance(reward, int)
        assert not np.isnan(reward)
        assert not np.isinf(reward)

        return self.decision_model.tracker.add_reward(
            reward=reward, model_name=self.decision_model.model_name, decision_id=self.id_)

    def get(self, track_once: bool = True):
        """
        Gets best variant for this Decision and tracks this Decision.

        Returns
        -------
        object
            best variant

        """

        # if best already chosen and tracked simply return best
        if self.chosen and self.__tracked:
            return self.best

        # calculate scores if it was not already set
        if self.scores is None or len(self.__scores) == 0:
            self.__scores = self.decision_model._score(variants=self.variants, givens=self.givens)
        else:
            # check if scores have proper length
            assert len(self.scores) == len(self.variants)

        if self.variants is not None and len(self.variants) != 0 and not self.__tracked:
            # there should be no difference between effect of those 2 conditions
            # since this  clause is reached only once
            if self.decision_model.track_url:
                # set message_id / decision_id only once
                self._set_message_id()
                track_runners_up = self.decision_model.tracker._should_track_runners_up(len(self.variants))
                if track_runners_up:
                    if not self.__variants_already_ranked:
                        # if variants are not ranked yet rank them
                        self.__ranked_variants = \
                            dm.DecisionModel._rank(variants=self.variants, scores=self.__scores)
                        self.__variants_already_ranked = True
                    if not self.chosen:
                        # if best is not set yet do it
                        self.__best = self.ranked_variants[0]
                    else:
                        # check if best is equal to first ranked variant
                        assert self.best == self.ranked_variants[0]

                    self.decision_model.tracker.track(
                        variant=self.best, variants=self.ranked_variants,
                        givens=self.givens, model_name=self.decision_model.model_name,
                        variants_ranked_and_track_runners_up=True, message_id=self.id_)
                else:
                    # calculate best if it was not already set
                    if not self.chosen:
                        self.__best = \
                            dm.DecisionModel.top_scoring_variant(variants=self.variants, scores=self.__scores)

                    self.decision_model.tracker.track(
                        variant=self.best, variants=self.variants,
                        givens=self.givens, model_name=self.decision_model.model_name,
                        variants_ranked_and_track_runners_up=False, message_id=self.id_)

                self.__tracked = True
            elif self.decision_model.track_url is None:
                warn('`track_url` can`t be None for tracked decisions (get() calls) -> this decision is not tracked')
            else:
                self.__best = \
                    dm.DecisionModel.top_scoring_variant(variants=self.variants, scores=self.__scores)

        self.__chosen = True
        return self.best

    def peek(self):
        # TODO Deprecated; Remove in 8.0
        """
        Calculates (or returns) best variant without tracking it

        Returns
        -------
        object
            best variant

        """

        if self.chosen:
            return self.best
        # set message_id / decision_id only once
        self._set_message_id()
        # set message_id / deicsion_id to decision model
        # TODO cover this by tests
        if self.scores is None or len(self.__scores) == 0:
            self.scores = self.decision_model._score(variants=self.variants, givens=self.givens)

        # Memoizes the chosen variant so same value is returned on subsequent calls
        self.__best = dm.DecisionModel.top_scoring_variant(variants=self.variants, scores=self.__scores)
        # The chosen variant is not tracked
        self.__chosen = True
        return self.best

    def ranked(self):
        # return .rankedVariants
        # If self.model.tracker && trackOnce == true, call self.model.tracker.track(self) one time ever,
        # no matter how many times get() or ranked() is called.
        if self.decision_model.tracker and self.tracked is False:
            self._track()

        return self.__ranked_variants

    def _track(self):
        # TODO this is new syntax - verify that this is desired
        #  also verify method name track() or trackOnce()
        # If trackOnce = true, the decision is tracked once and only once, no matter how many times get() is called.
        # If trackOnce = false, the decision is not tracked.
        assert self.chosen is True
        assert self.ranked_variants is not None

        if not self.tracked:
            track_runners_up = self.decision_model.tracker._should_track_runners_up(len(self.variants))
            decision_id = self.decision_model.tracker.track(
                variant=self.best, ranked_variants=self.variants,
                givens=self.givens, model_name=self.decision_model.model_name,
                variants_ranked_and_track_runners_up=track_runners_up, message_id=self.id_)
            # TODO What should happen on tracker.track() error? Warning will be returned
            #  but self.__tracked flag will still be set to True
            assert self.id_ == decision_id
            self.__tracked = True
        else:
            raise ValueError('Decision has already been tracked')

        assert decision_id is not None
        return decision_id
