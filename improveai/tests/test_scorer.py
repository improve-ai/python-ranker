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
            Scorer(model_url=[1, 2, 3])

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

        # 1.7733567613589007
        # 1.7772981646970958
        # -0.561980838193514
        # -0.5601502151900769

        expected_scores = \
            np.array([1.7733567613589007, 1.7772981646970958, -0.561980838193514, -0.5601502151900769])
        np.testing.assert_array_equal(scores, expected_scores)

    def test_score_valid_context(self):
        # TODO perhaps a model with `context` features should be trained
        scorer = Scorer(model_url=self.valid_fs_model_url)

        items = ['a', 'b', 'c', 'd']
        np.random.seed(0)
        scores = scorer.score(items=items, context=self.valid_context)

        # 1.7755810874926288
        # 1.775398326249525
        # -0.5622938817878743
        # -0.5623421163970471

        expected_scores = \
            np.array([1.7755810874926288, 1.775398326249525, -0.5622938817878743, -0.5623421163970471])
        np.testing.assert_array_equal(scores, expected_scores)

    def test_score_tuple_items_valid_context(self):
        # TODO perhaps a model with `context` features should be trained
        scorer = Scorer(model_url=self.valid_fs_model_url)

        items = tuple(['a', 'b', 'c', 'd'])
        np.random.seed(0)
        scores = scorer.score(items=items, context=self.valid_context)

        # 1.7755810874926288
        # 1.775398326249525
        # -0.5622938817878743
        # -0.5623421163970471

        expected_scores = \
            np.array([1.7755810874926288, 1.775398326249525, -0.5622938817878743, -0.5623421163970471])
        np.testing.assert_array_equal(scores, expected_scores)

    def test_score_ndarray_items_valid_context(self):
        # TODO perhaps a model with `context` features should be trained
        scorer = Scorer(model_url=self.valid_fs_model_url)

        items = np.array(['a', 'b', 'c', 'd'])
        np.random.seed(0)
        scores = scorer.score(items=items, context=self.valid_context)

        # 1.7755810874926288
        # 1.775398326249525
        # -0.5622938817878743
        # -0.5623421163970471

        expected_scores = \
            np.array([1.7755810874926288, 1.775398326249525, -0.5622938817878743, -0.5623421163970471])
        np.testing.assert_array_equal(scores, expected_scores)

    def test_scorer_with_gzipped_model(self):
        gzipped_model_url = self.valid_fs_model_url + '.gz'
        # attempt to load gzipped model
        scorer = Scorer(model_url=gzipped_model_url)

        assert scorer.chooser is not None
        assert isinstance(scorer.chooser, XGBChooser)

    def test_score_raises_for_bad_items_type(self):
        scorer = Scorer(model_url=self.valid_fs_model_url)

        with raises(AssertionError) as aerr:
            scorer.score(items=None, context=None)

        with raises(AssertionError) as aerr:
            scorer.score(items=123, context=None)

        with raises(AssertionError) as aerr:
            scorer.score(items=1.23, context=None)

        with raises(AssertionError) as aerr:
            scorer.score(items={1.23}, context=None)

        with raises(AssertionError) as aerr:
            scorer.score(items={'a': 1.23}, context=None)

    def test_score_raises_for_non_json_encodable_items(self):
        scorer = Scorer(model_url=self.valid_fs_model_url)
        items = [np.array([1, 2, 3]) for _ in range(10)]
        with raises(ValueError) as verr:
            scorer.score(items=items, context=None)

    def test_score_raises_for_non_json_encodable_context(self):
        scorer = Scorer(model_url=self.valid_fs_model_url)
        items = [1, 2, 3]
        context = np.array([1, 2, 3])
        with raises(ValueError) as verr:
            scorer.score(items=items, context=context)

