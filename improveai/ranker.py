import numpy as np

from improveai.scorer import Scorer


class Ranker:
    """
    A utility for ranking items based on their scores. The Ranker struct takes
    a ImproveAI model to evaluate and rank the given items.
    """

    # TODO I think we might go with scorer and model_url as properties with only getter.
    #  If we won't define them as properties anyone will be able to mutate them
    #  Exposing only getters and leaving underlying attributes private allows to make them
    #  immutable and nobody will be able to tinker with them as one could with
    #  'normal' instance attributes

    @property
    def scorer(self):
        """
        A Scorer is used to calculate scores for items.
        Items must be JSON encodable. If both `scorer` and `model_url` are provided
        `scorer` is preferred.

        Returns
        -------
        Scorer
             a Scorer object for this Ranker
        """
        return self.__scorer

    @property
    def model_url(self):
        """
        URL or local FS path leading to Improve AI booster to be used with this
        scorer. Allows both raw xgboost files (most commonly having \*.xgb extension)
        as well as gzipped boosters (\*.xgb.gz). If both `scorer` and `model_url`
        are provided to Ranker's constructor `scorer` is preferred.
        Can only be set during object initialization.

        Returns
        -------
        str
            model URL for this Ranker

        """
        return self.__model_url

    def __init__(self, scorer: Scorer = None, model_url: str = None):
        """
        Init Ranker with params. Either `scorer` or `model_url` must be provided.
        If both are provided Scorer is preferred.

        Parameters
        ----------
        scorer: Scorer
            a Scorer object to be used with this Ranker
        model_url: str
            URL or local FS of a plain or gzip compressed Improve AI model resource
        """

        if scorer is not None:
            assert isinstance(scorer, Scorer)
            # if both are provided choose scorer over model_url
            self.__scorer = scorer
            self.__model_url = self.__scorer.model_url
        elif scorer is None and model_url is not None:
            self.__model_url = model_url
            self.__scorer = Scorer(model_url=self.__model_url)
        else:
            raise ValueError('Either `scorer` or `model_url` must be provided')

    def rank(self, items: list or tuple or np.ndarray, context: object = None) -> list or tuple or np.ndarray:
        """
        Ranks items and returns them ordered best to worst

        Parameters
        ----------
        items: list or tuple or np.ndarray
            list of items to be ranked
        context: object
            any JSON encodable extra context info that will be used with each of
            the item to get its score

        Returns
        -------
        list or tuple or np.ndarray
            a collection of ranked items, sorted by their scores in descending order.
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
