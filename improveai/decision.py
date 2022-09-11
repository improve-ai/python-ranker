import numpy as np
from warnings import warn

import improveai.decision_model as dm
from improveai.utils.general_purpose_tools import check_variants, is_valid_ksuid


class Decision:

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

        assert isinstance(givens, dict) or givens is None
        self.__givens = givens

        check_variants(ranked_variants)
        self.__ranked_variants = ranked_variants
        self.__id_ = None

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

        assert is_valid_ksuid(self.id_)
        assert self.decision_model.model_name is not None
        assert reward is not None
        assert isinstance(reward, float) or isinstance(reward, int)
        assert not np.isnan(reward)
        assert not np.isinf(reward)

        self.decision_model.tracker.add_reward(
            reward=reward, model_name=self.decision_model.model_name, decision_id=self.id_)

    def get(self):
        """
        return best variant for this decision
        tracks this decision

        Returns
        -------
        object
            best variant

        """

        return self.ranked_variants[0]

    def ranked(self):
        """
        return ranked variants for this decision
        tracks this decision

        Returns
        -------
        list
            ranked variants of this decision

        """
        return self.ranked_variants

    def _track(self):
        """
        track this decision

        Returns
        -------
        str
            decision ID (ksuid) obtained during tracking

        """
        assert self.ranked_variants is not None
        # make sure that self.__id_ is set for the first time
        assert self.id_ is None
        # track() message ID for current decision -> decision ID
        self.__id_ = self.decision_model.tracker.track(
            ranked_variants=self.ranked_variants, givens=self.givens, model_name=self.decision_model.model_name)
        # if self.id_ is not None at this point it means that track() was called successfully
        assert self.id_ is not None, 'decision tracking failed -> please check console for tracking error.'
        # persist most recent decision ID to `tracked` decision model
        self.decision_model.last_decision_id = self.id_

        return self.id_
