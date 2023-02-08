import numpy as np

from improveai.scorer import Scorer


class Ranker:

    # TODO I think we might go with scorer and model_url as properties with only getter.
    #  If we won't define them as properties anyone will be able to mutate them
    #  Exposing only getters and leaving underlying attributes private allows to make them
    #  immutable and nobody will be able to tinker with them as one could with '
    #  normal' instance attributes
    @property
    def scorer(self):
        """
        A Scorer is used to calculate scores for provided items.
        Items must be JSON encodable. If both scorer and model_url are provided
         scorer is preferred.

        Returns
        -------
        Scorer
             a scorer for this Ranked
        """
        return self.__scorer


    @property
    def model_url(self):
        """
        A URL leading either to local file or remote ImproveAI model. If both scorer
        and model_url are provided in the constructor model_url is ignored.

        Returns
        -------
        str
            a model URL
        """
        return self.__model_url

    def __init__(self, scorer: Scorer = None, model_url: str = None):

        # instantiate scorer and model_url
        self.__scorer = scorer
        self.__model_url = model_url
        # perform sanity check -> if scorer parameter is None set model_url with
        # provided URL and use it to create Scorer object and set scorer property
        if scorer is None and model_url is not None:
            self.__model_url = model_url
            self.__scorer = Scorer(model_url=self.__model_url)
        elif scorer is not None and model_url is not None:
            # if both are provided choose scorer over model_url
            self.__scorer = scorer
            self.__model_url = self.__scorer.model_url

    def rank(self, items: list or tuple or np.ndarray, context: object) -> list:
        """
        Ranks and returns provided items ordered best to worst

        Parameters
        ----------
        items: list or tuple or np.ndarray
            list of items to be ranked
        context: object
            any JSON encodable object serving as a context for ranking

        Returns
        -------
        list
            a list of ranked items
        """
        # assure provided items are not empty
        assert len(items) > 0

        # calculate scores for provided items
        scores = self.scorer.score(items=items, context=context)

        # convert variants to numpy array for faster sorting
        best_to_worse_scores = np.argsort(scores)[::-1]
        # return descending sorted variants
        if isinstance(items, np.ndarray):
            return items[best_to_worse_scores]
        elif isinstance(items, tuple):
            return tuple(items[i] for i in best_to_worse_scores)
        else:
            return [items[i] for i in best_to_worse_scores]
