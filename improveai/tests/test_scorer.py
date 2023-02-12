from copy import deepcopy
import json
import math
import numpy as np
import requests_mock as rqm
import os
from pytest import fixture, raises
import sys
from warnings import catch_warnings, simplefilter

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

        self.valid_context = {'a': 1, 'b': 2, 'c': 3}

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
        np.random.seed(0)
        scores = scorer.score(items=[1, 2, 3], context=None)
        expected_scores = \
            np.array([-0.07262340603001519, 2.5619021180586072, 2.5619021111587506]).astype(np.float64)
        np.testing.assert_array_equal(scores, expected_scores)

    def test_score_valid_context(self):
        # TODO perhaps a model with `context` features should be trained
        scorer = Scorer(model_url=self.valid_fs_model_url)
        np.random.seed(0)
        scores = scorer.score(items=[1, 2, 3], context=self.valid_context)
        expected_scores = \
            np.array([-0.07262340603001519, 2.5619021180586072, 2.5619021111587506]).astype(np.float64)
        np.testing.assert_array_equal(scores, expected_scores)
