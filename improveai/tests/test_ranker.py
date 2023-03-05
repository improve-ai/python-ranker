import numpy as np
import os
from pytest import raises

from improveai.ranker import Ranker
from improveai.scorer import Scorer


class TestRanker:

    def test_constructor_prefers_scorer_over_model_url(self):
        model_url = os.getenv('DUMMY_MODEL_PATH', None)
        assert model_url is not None

        scorer = Scorer(model_url=model_url)
        ranker = Ranker(scorer=scorer)

        assert isinstance(ranker.scorer, Scorer)

        assert ranker.model_url == scorer.model_url

    def test_constructor_with_model_url(self):
        model_url = os.getenv('DUMMY_MODEL_PATH', None)
        assert model_url is not None

        ranker = Ranker(model_url=model_url)
        assert isinstance(ranker.scorer, Scorer)
        assert ranker.model_url == model_url

    def test_constructor_raises_for_bad_scorer_type(self):
        with raises(AssertionError) as aerr:
            Ranker(scorer=123)

        with raises(AssertionError) as aerr:
            Ranker(scorer='abc')

        with raises(AssertionError) as aerr:
            Ranker(scorer=[1,2,3])

    def test_rank_no_context(self):
        model_url = os.getenv('DUMMY_MODEL_PATH', None)
        assert model_url is not None

        ranker = Ranker(model_url=model_url)
        items = ['b', 'a', 'd']
        np.random.seed(0)
        ranked_items = ranker.rank(items=items, context=None)
        expected_ranked_items = ['b', 'a', 'd']
        np.testing.assert_array_equal(ranked_items, expected_ranked_items)

    def test_rank(self):
        model_url = os.getenv('DUMMY_MODEL_PATH', None)
        assert model_url is not None

        ranker = Ranker(model_url=model_url)
        items = ['c', 'a', 'd']
        # TODO move this to a fixture
        context = {'ga': 1, 'gb': 0}
        np.random.seed(0)
        ranked_items = ranker.rank(items=items, context=context)
        expected_ranked_items = ['a', 'c', 'd']
        np.testing.assert_array_equal(ranked_items, expected_ranked_items)
