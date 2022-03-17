class GivensProvider:

    # TODO how givens should be provided - should it be some sort of customized
    #  implementation for each model dynamically providing givens?
    def givens(self, for_model, givens: dict = {}) -> dict:
        return givens
