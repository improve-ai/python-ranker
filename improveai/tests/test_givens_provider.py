import numpy as np
from pytest import fixture, raises
from unittest import TestCase

import improveai.decision_model as dm
import improveai.givens_provider as gp


class TestGivensProvider(TestCase):

    @property
    def test_decision_model(self):
        return self._test_decision_model

    @test_decision_model.setter
    def test_decision_model(self, value):
        self._test_decision_model = value

    @property
    def valid_test_givens(self):
        return self._valid_test_givens

    @valid_test_givens.setter
    def valid_test_givens(self, value):
        self._valid_test_givens = value

    @fixture(autouse=True)
    def prepare_test_artifacts(self):
        self.test_decision_model = dm.DecisionModel('dummy-model')
        self.valid_test_givens = {'a': [1, 2, 3], 'b': {'c': 1, 'd': 'e'}}

    def test_givens_provider_none_givens(self):
        givens = gp.GivensProvider().givens(for_model=self.test_decision_model, context=None)
        assert givens is None

    def test_givens_provider_empty_givens(self):
        givens = gp.GivensProvider().givens(for_model=self.test_decision_model, context={})
        assert givens == {}

    def test_givens_provider_no_givens(self):
        givens = gp.GivensProvider().givens(for_model=self.test_decision_model)
        assert givens is None

    def test_givens_provider_good_givens(self):
        givens = gp.GivensProvider().givens(for_model=self.test_decision_model, context=self.valid_test_givens)
        assert givens == self.valid_test_givens

    def test_givens_provider_raises_for_bad_givens_types(self):
        # bad givens types:
        # string, list, tuple, np.array, int, float
        bad_givens = ['abc', ['a', 'b', 'c'], ('a', 'b', 'c'), np.array(['a', 'b', 'c']), 1234, 1234.1234]
        for g in bad_givens:
            with raises(AssertionError) as aerr:
                gp.GivensProvider().givens(for_model=self.test_decision_model, context=g)
                assert str(aerr.value)

    def test_givens_provider_raises_for_bad_decision_model_types(self):
        # bad DecisionModel types:
        # NoneType, string, list, tuple, np.array, int, float
        bad_models = [None, 'abc', ['a', 'b', 'c'], ('a', 'b', 'c'), np.array(['a', 'b', 'c']), 1234, 1234.1234]
        for decision_model in bad_models:
            with raises(AssertionError) as aerr:
                gp.GivensProvider().givens(for_model=decision_model)
                assert str(aerr.value)
