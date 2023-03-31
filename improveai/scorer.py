import numpy as np

from improveai.chooser import XGBChooser
from improveai.settings import DEBUG
from improveai.utils.general_purpose_tools import check_candidates


class Scorer:

    # TODO do not expose any attribute which is not strictly necessary as a property
    # exposed as read-only
    @property
    def model_url(self) -> str:
        """
        URL or local FS path leading to Improve AI booster to be used with this
        scorer. Allows both raw xgboost files (most commonly having \*.xgb extension) as well as
        gzipped boosters (\*.xgb.gz).
        Can only be set during object initialization.

        Returns
        -------
        str
            model URL for this Scorer

        """
        return self.__model_url

    @property
    def chooser(self) -> XGBChooser:
        """
        Chooser object (for more info please check Chooser tab -> XGBChooser class)
        for this scorer.
        Can only be set during object initialization.

        Returns
        -------
        XGBChooser
            a chooser object for this Scorer

        """
        return self.__chooser

    @property
    def TIEBREAKER_MULTIPLIER(self) -> float:
        """
        Small float value used to randomize model's scores (random numbers
        sampled uniformly from [0, 1) are multiplied by `TIEBREAKER_MULTIPLIER` and added to scores)

        Returns
        -------
        float
            Small float value used to randomize model's scores

        """
        return 2**-23

    def __init__(self, model_url: str):
        """
        Init with params

        Parameters
        ----------
        model_url: str
            URL or local FS path leading to ImproveAI model to be used with this Scorer

        """
        assert model_url is not None and isinstance(model_url, str)

        self.__model_url = model_url
        self.__chooser = XGBChooser()
        self.__chooser.load_model(model_url)

    def score(self, items: list or tuple or np.ndarray, context: object = None) -> np.ndarray:
        """
        Calculate scores for provided items and a context

        Parameters
        ----------
        items: list or tuple or np.ndarray
            list of items to be scored
        context: object
            any JSON encodable object serving as a context for scoring

        Returns
        -------
        np.ndarray
            an array of scores for items of type double

        """
        check_candidates(candidates=items)
        # log givens for DEBUG == True
        if DEBUG is True:
            print(f'[DEBUG] givens: {context}')

        # encode variants with single givens
        # TODO Check that it raises if there is a problem during feature encoding
        encoded_candidates_matrix = self.__chooser.encode_candidates_with_context(
            candidates=items, context=context)

        # TODO Check that it raises for model error, such as non-JSON encodeable data type
        scores = self.__chooser.calculate_predictions(features_matrix=encoded_candidates_matrix) + \
            np.array(np.random.rand(len(items)), dtype='float64') * self.TIEBREAKER_MULTIPLIER

        return scores.astype(np.float64)