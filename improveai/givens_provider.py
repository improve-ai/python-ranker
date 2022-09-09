import improveai.decision_model as dm


class GivensProvider:

    def givens(self, for_model, givens: dict or None = None) -> dict:
        """
        Provides givens for input `for_model`. Input `givens` are included in
        returned value

        Parameters
        ----------
        for_model: dm.DecisionModel
            instance of DecisionModel for which givens should be provided
        givens: dict or None
            givens which will be returned

        Returns
        -------
        dict or None
            givens which will be returned

        """
        assert isinstance(for_model, dm.DecisionModel)
        assert isinstance(givens, dict) or givens is None
        return givens
