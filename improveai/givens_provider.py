import improveai.decision_model as dm


class GivensProvider:

    # TODO how givens should be provided - should it be some sort of customized
    #  implementation for each model dynamically providing givens?
    #  also in iOS this method returns @{} and is named givensForModel
    #  (but maybe I got this wrong)
    #  I made it return provided givens -> which implementation is correct
    def givens(self, for_model, givens: dict or None = {}) -> dict:
        """
        Passthrough method

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
