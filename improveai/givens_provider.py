import improveai.decision_model as dm


class GivensProvider:

    def givens(self, for_model, context=None) -> tuple:
        """
        Provides givens for input `for_model`. Input `context` are included in
        returned value

        Parameters
        ----------
        for_model: dm.DecisionModel
            instance of DecisionModel for which givens should be provided
        context:
            context which will be included as the first element of the givens

        Returns
        -------
        tuple
            givens containing only the provided context

        """
        assert isinstance(for_model, dm.DecisionModel)
        return context
