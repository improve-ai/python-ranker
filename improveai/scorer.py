import numpy as np

from improveai.chooser import XGBChooser
from improveai.settings import DEBUG
from improveai.utils.general_purpose_tools import check_candidates


class Scorer:

    # TODO do not expose any attribute which is not strictly necessary as a property
    # exposed as read-only
    @property
    def model_url(self):
        return self.__model_url

    @property
    def chooser(self):
        return self.__chooser

    @property
    def TIEBREAKER_MULTIPLIER(self) -> float:
        """
        Small value randomized and added to model's scores

        Returns
        -------
        float
            Small value randomized and added to model's scores

        """
        return 2**-23

    def __init__(self, model_url: str):
        assert model_url is not None and isinstance(model_url, str)

        self.__model_url = model_url
        # attempt to create a chooser object
        try:
            model_src = XGBChooser.get_model_src(model_src=model_url)
        except Exception as exc:
            raise exc

        self.__chooser = XGBChooser()
        self.__chooser.load_model(model_src)

    def score(self, items: list or tuple or np.ndarray, context: object = None) -> np.ndarray:
        """
        Calculate scores for provided items given a context

        Parameters
        ----------
        items: list or tuple or np.ndarray
            list of items to be scored
        context: object
            any JSON encodable object serving as a context for scoring

        Returns
        -------
        np.ndarray
            an array of double scores

        """
        check_candidates(candidates=items)
        # log givens for DEBUG == True
        if DEBUG is True:
            print(f'[DEBUG] givens: {context}')

        # encode variants with single givens
        # TODO Check that it raises if there is a problem during feature encoding
        encoded_candidates_matrix = self.__chooser.encode_candidates_single_context(
            candidates=items, context=context)

        # TODO Check that it raises for model error, such as non-JSON encodeable data type
        scores = self.__chooser.calculate_predictions(features_matrix=encoded_candidates_matrix) + \
            np.array(np.random.rand(len(items)), dtype='float64') * self.TIEBREAKER_MULTIPLIER

        return scores.astype(np.float64)
