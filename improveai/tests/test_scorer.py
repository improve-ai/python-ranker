import numpy as np
import os
from pytest import fixture, raises

from improveai.chooser import XGBChooser
from improveai.ranker import Scorer


class TestScorer:

    @property
    def valid_fs_model_url(self):
        return self._valid_fs_model_url

    @valid_fs_model_url.setter
    def valid_fs_model_url(self, value):
        self._valid_fs_model_url = value

    @property
    def invalid_fs_model_url(self):
        return self._invalid_fs_model_url

    @invalid_fs_model_url.setter
    def invalid_fs_model_url(self, value):
        self._invalid_fs_model_url = value

    @property
    def valid_context(self):
        return self._valid_context

    @valid_context.setter
    def valid_context(self, value):
        self._valid_context = value

    @fixture(autouse=True)
    def prep_env(self):
        self.valid_fs_model_url = os.getenv('DUMMY_MODEL_PATH', None)
        assert self.valid_fs_model_url is not None

        self.invalid_fs_model_url = os.getenv('DUMMY_MODEL_INVALID_PATH', None)
        assert self.invalid_fs_model_url is not None

        # context = {'ga': 1, 'gb': 0}
        self.valid_context = {'ga': 1, 'gb': 0}

    def test_constructor_with_valid_model_url(self):
        scorer = Scorer(model_url=self.valid_fs_model_url)
        assert isinstance(scorer.chooser, XGBChooser)

    def test_constructor_raises_for_invalid_model_url(self):
        with raises(AssertionError) as aerr:
            Scorer(model_url=None)

        with raises(AssertionError) as aerr:
            Scorer(model_url=123)

        with raises(AssertionError) as aerr:
            Scorer(model_url=[1,2, 3])

        with raises(AssertionError) as aerr:
            Scorer(model_url=['a','b', 'b'])

    def test_constructor_raises_for_non_existing_file(self):
        with raises(FileNotFoundError) as fnferr:
            Scorer(model_url=self.invalid_fs_model_url)

    def test_score_no_context(self):
        scorer = Scorer(model_url=self.valid_fs_model_url)
        items = ['a', 'b', 'c', 'd']
        np.random.seed(0)
        scores = scorer.score(items=items, context=None)

        expected_scores = \
            np.array([-0.5532910567296306, 1.7772981646970958, -0.561980838193514, -0.5601502151900769]).astype(np.float64)
        np.testing.assert_array_equal(scores, expected_scores)

    def test_score_valid_context(self):
        # TODO perhaps a model with `context` features should be trained
        scorer = Scorer(model_url=self.valid_fs_model_url)

        items = ['a', 'b', 'c', 'd']
        np.random.seed(0)
        scores = scorer.score(items=items, context=self.valid_context)

        expected_scores = \
            np.array([1.7398908990847308, 1.8778475049452037, -0.5571367091079732, 0.22394380553421966]).astype(np.float64)
        np.testing.assert_array_equal(scores, expected_scores)
