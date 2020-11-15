from collections.abc import Iterable
from copy import deepcopy
from frozendict import frozendict
import json
import numpy as np
import os
from pytest import fixture, raises
from typing import List
from unittest import TestCase

from decisions.v6_fast import Decision
from models.decision_models import DecisionModel

# All getters and setters should be tested for (aka. generic tests):
#
# getters:
# test if attribute/property is accessible from the outside of Decision (should be)
# test if attribute`s/property`s getter returns completely immutable data structure
# (e.g. i python it should return tuple of frozendicts)
#
# setters:
# test if it is possible to variants more than once per instance (should not be possible)
# test if attribute/property can be mutated from inside of Decision class
#
# getters + setters:
# test case: if setter is provided with mutable data structure (e.g. list of dicts)
# does it get properly converted into immutable collection (e.g. tuple of frozendicts in py).
# Both setter and getter probably need assertions for this test


# class MyTest(unittest.TestCase):
#     @pytest.fixture(autouse=True)
#     def initdir(self, tmpdir):
#         tmpdir.chdir() # change to pytest-provided temporary directory
#         tmpdir.join("samplefile.ini").write("# testdata")
#
#     def test_method(self):
#         s = open("samplefile.ini").read()
#         assert "testdata" in s

class TestDecisionProperties(TestCase):

    @property
    def artifacts_variants(self) -> list:
        return self._artifacts_variants

    @artifacts_variants.setter
    def artifacts_variants(self, new_val: list):
        self._artifacts_variants = new_val
        
    @property
    def artifacts_context(self):
        return self._artifacts_context
    
    @artifacts_context.setter
    def artifacts_context(self, new_val: dict):
        self._artifacts_context = new_val
        
    @property
    def artifacts_model(self) -> DecisionModel:
        return self._artifacts_model
    
    @artifacts_model.setter
    def artifacts_model(self, new_val: DecisionModel):
        self._artifacts_model = new_val

    @property
    def artifacts_modeln(self) -> DecisionModel:
        return self._artifacts_modeln

    @artifacts_modeln.setter
    def artifacts_modeln(self, new_val: DecisionModel):
        self._artifacts_modeln = new_val

    @property
    def decisions_kwargs(self) -> dict:
        return self._decisions

    @decisions_kwargs.setter
    def decisions_kwargs(self, new_val: dict):
        self._decisions = new_val

    @property
    def modeln_dec_kwgs_keys(self) -> List:
        return self._modeln_dec_kwgs_keys

    @modeln_dec_kwgs_keys.setter
    def modeln_dec_kwgs_keys(self, new_val: List):
        self._modeln_dec_kwgs_keys = new_val

    @fixture(autouse=True)
    def prepare_env(self):
        model_kind = os.getenv('MODEL_KIND')
        model_pth = os.getenv('MODEL_PTH')

        self.artifacts_model = \
            DecisionModel(model_kind=model_kind, model_pth=model_pth)

        self.artifacts_modeln = \
            self.artifacts_model.chooser._get_model_metadata()['model']

        # context = frozendict({})
        with open(os.getenv('CONTEXT_JSON_PTH'), 'r') as cjson:
            read_str = ''.join(cjson.readlines())
            self.context = frozendict(json.loads(read_str))

        with open(os.getenv('VARIANTS_JSON_PTH')) as mjson:
            read_str = ''.join(mjson.readlines())
            self.variants = \
                tuple(frozendict(v) for v in json.loads(read_str))[:10]

        self.decisions_kwargs = {
            # 'plain': Decision(),
            'varaiants_model': {
                'variants': self.variants, 'model': self.artifacts_model},
            'variants_model_context': {
                'variants': self.variants, 'model': self.artifacts_model,
                'context': self.context},
            'variant_model_context_nulled': {
                'variants': self.variants, 'model': self.artifacts_model,
                'context': None},
            'ranked_variants_modeln': {
                'ranked_variants': self.variants,
                'model_name': self.artifacts_modeln},
            'ranked_variants_modeln_context': {
                'ranked_variants': self.variants,
                'model_name': self.artifacts_modeln, 'context': self.context},
            'ranked_variants_modeln_context_nulled':
                {'ranked_variants': self.variants,
                 'model_name': self.artifacts_modeln, 'context': None}}

        self.modeln_dec_kwgs_keys = \
            ['ranked_variants_modeln', 'ranked_variants_modeln_context',
             'ranked_variants_modeln_context_nulled']

        self.context_dec_kwgs_keys = \
            ['variants_model_context', 'variant_model_context_nulled',
             'ranked_variants_modeln_context',
             'ranked_variants_modeln_context_nulled']

    def _generic_test_getter_outside(
            self, attr_names, dec_kwargs, kwargs_keys):

        true_vals_list = \
            [d_kwgs.get(kwgs_k) for d_kwgs, kwgs_k
             in zip(dec_kwargs.values(), kwargs_keys)]

        for case, decision_kwgs, true_val, attr_n, kwgs_key in zip(
                dec_kwargs.keys(), dec_kwargs.values(),
                true_vals_list, attr_names, kwargs_keys):
            print('asserting: {}'.format(case))
            curr_decision = Decision(**decision_kwgs)
            assert hasattr(curr_decision, attr_n)
            assert getattr(curr_decision, attr_n) == true_val

    def _generic_test_getter_outside_after_callable(
            self, attr_names, dec_kwargs, callable_n):

        for case, decision_kwgs, attr_n in zip(
                dec_kwargs.keys(), dec_kwargs.values(), attr_names):

            print('asserting: {} with: {}'.format(case, callable_n))

            curr_decision = Decision(**decision_kwgs)
            assert hasattr(curr_decision, callable_n)
            dec_callable = getattr(curr_decision, callable_n)
            dec_callable()
            assert hasattr(curr_decision, attr_n)
            assert getattr(curr_decision, attr_n) is not None

    def _generic_test_getter_returns_des_type(
            self, attr_names, dec_kwargs, type_checker):

        for case, decision_kwgs, attr_n in zip(
                    dec_kwargs.keys(), dec_kwargs.values(), attr_names):
            print('asserting: {}'.format(case))
            curr_decision = Decision(**decision_kwgs)
            print(getattr(curr_decision, attr_n))
            type_checker(getattr(curr_decision, attr_n))

    def _generic_test_getter_returns_des_type_after_callable(
            self, attr_names, dec_kwargs, callable_n, type_checker):

        for case, decision_kwgs, attr_n in zip(
                dec_kwargs.keys(), dec_kwargs.values(), attr_names):

            curr_decision = Decision(**decision_kwgs)

            assert hasattr(curr_decision, callable_n)
            dec_callable = getattr(curr_decision, callable_n)
            dec_callable()

            type_checker(getattr(curr_decision, attr_n))

    def _generic_test_callable_returns_des_type(
            self, dec_kwargs, callable_n, type_checker):

        for case, decision_kwgs in zip(
                dec_kwargs.keys(), dec_kwargs.values()):

            curr_decision = Decision(**decision_kwgs)

            assert hasattr(curr_decision, callable_n)
            dec_callable = getattr(curr_decision, callable_n)
            res = dec_callable()

            type_checker(res)

    def _generic_test_set_once(
            self, attr_names, set_attempt_vals, dec_kwargs, kwargs_keys):

        true_vals_list = \
            [d_kwgs.get(kwgs_k) for d_kwgs, kwgs_k
             in zip(dec_kwargs.values(), kwargs_keys)]

        for case, decision_kwgs, set_attempt_v, attr_n, true_v in zip(
                dec_kwargs.keys(), dec_kwargs.values(),
                set_attempt_vals, attr_names, true_vals_list):

            print('asserting: {}'.format(case))
            curr_decision = Decision(**decision_kwgs)

            with raises(AttributeError):
                setattr(curr_decision, attr_n, set_attempt_v)

    def _generic_test_set_once_after_callable(
            self, attr_names, set_attempt_vals, dec_kwargs, callable_n):

        for case, decision_kwgs, set_attempt_v, attr_n in zip(
                dec_kwargs.keys(), dec_kwargs.values(),
                set_attempt_vals, attr_names):

            print('asserting: {}'.format(case))
            curr_decision = Decision(**decision_kwgs)

            dec_callable = getattr(curr_decision, callable_n)
            dec_callable()

            with raises(AttributeError):
                setattr(curr_decision, attr_n, set_attempt_v)

    def _generic_test_set_mutable_get_immutable(
            self, attr_names, dec_kwargs, kwargs_keys, mutable_maker,
            type_checker):

        for case, decision_kwgs, attr_n, kwg_k in zip(
                dec_kwargs.keys(), dec_kwargs.values(),
                attr_names, kwargs_keys):

            ref_values = deepcopy(decision_kwgs[kwg_k])
            decision_kwgs[kwg_k] = mutable_maker(decision_kwgs[kwg_k])

            curr_decision = Decision(**decision_kwgs)
            type_checker(getattr(curr_decision, attr_n))
            assert ref_values == getattr(curr_decision, attr_n)

    @staticmethod
    def _variants_type_checker(variants):
        assert isinstance(variants, list) or isinstance(variants, np.ndarray)\
               or isinstance(variants, tuple)
        for v in variants:
            assert isinstance(v, dict) or isinstance(v, frozendict)

    @staticmethod
    def _modeln_type_checker(modeln):
        assert isinstance(modeln, str)

    @staticmethod
    def max_runners_up_type_checker(max_runners_up):
        assert isinstance(max_runners_up, int)

    @staticmethod
    def _context_type_checker(context):
        # print(context)
        assert isinstance(context, dict) or isinstance(context, frozendict) \
               or context is None

    @staticmethod
    def _memoized_scores_type_checker(memoized_scores):
        assert isinstance(memoized_scores, list) \
               or isinstance(memoized_scores, tuple) \
               or isinstance(memoized_scores, np.ndarray) \
               or memoized_scores is None

        if memoized_scores is not None:
            for el in memoized_scores:
                assert isinstance(el, float)

    @staticmethod
    def _memoized_ranked_type_checker(memoized_scores):
        assert isinstance(memoized_scores, list) \
               or isinstance(memoized_scores, tuple) \
               or isinstance(memoized_scores, np.ndarray) \
               or memoized_scores is None

        if memoized_scores is not None:
            for el in memoized_scores:
                assert isinstance(el, dict) or isinstance(el, frozendict)

    @staticmethod
    def _memoized_best_type_checker(memoized_best):
        assert isinstance(memoized_best, dict) \
               or isinstance(memoized_best, frozendict) or memoized_best is None

    @staticmethod
    def _return_scored_type_checker(scored_return):
        assert not isinstance(scored_return, str)

        assert isinstance(scored_return, Iterable) or scored_return is None
        if scored_return is not None:
            for row in scored_return:
                assert isinstance(row[0], dict) or isinstance(row[0], frozendict)
                assert isinstance(row[1], float)

    def test_variants_getter_outside(self):

        kwargs_keys = \
            ['variants' for _ in range(3)] + \
            ['ranked_variants' for _ in range(3)]

        attr_names = ['variants' for _ in range(6)]

        self._generic_test_getter_outside(
            attr_names=attr_names, dec_kwargs=self.decisions_kwargs,
            kwargs_keys=kwargs_keys)

    def test_variants_getter_returns_des_type(self):

        attr_names = ['variants' for _ in range(6)]

        self._generic_test_getter_returns_des_type(
            attr_names=attr_names, dec_kwargs=self.decisions_kwargs,
            type_checker=TestDecisionProperties._variants_type_checker)

    def test_variants_set_once(self):

        true_variants = \
            [self.variants for _ in range(len(self.decisions_kwargs))]  # + [None]

        set_attempt_variants = \
            [{'text': 'an example variant'} for _ in range(len(true_variants))]

        assert true_variants != set_attempt_variants

        attr_names = ['variants' for _ in range(6)]
        kwargs_keys = \
            ['variants' for _ in range(3)] + \
            ['ranked_variants' for _ in range(3)]

        self._generic_test_set_once(
            attr_names=attr_names, set_attempt_vals=set_attempt_variants,
            dec_kwargs=self.decisions_kwargs, kwargs_keys=kwargs_keys)

    def test_modeln_getter_outside(self):

        kwargs_keys = ['model_name' for _ in range(3)]
        modeln_dec_kwgs = \
            dict((k, v) for k, v in self.decisions_kwargs.items()
                 if k in self.modeln_dec_kwgs_keys)

        print(modeln_dec_kwgs)
        print(kwargs_keys)

        self._generic_test_getter_outside(
            attr_names=kwargs_keys, dec_kwargs=modeln_dec_kwgs,
            kwargs_keys=kwargs_keys)

    def test_modeln_getter_returns_des_type(self):

        kwargs_keys = ['model_name' for _ in range(3)]
        modeln_dec_kwgs = \
            dict((k, v) for k, v in self.decisions_kwargs.items()
                 if k in self.modeln_dec_kwgs_keys)

        self._generic_test_getter_returns_des_type(
            attr_names=kwargs_keys, dec_kwargs=modeln_dec_kwgs,
            type_checker=TestDecisionProperties._modeln_type_checker)

    def test_modeln_set_once(self):
        # TODO finish up

        modeln_dec_kwgs = \
            dict((k, v) for k, v in self.decisions_kwargs.items()
                 if k in self.modeln_dec_kwgs_keys)

        true_modelns = \
            [self.artifacts_modeln for _ in range(len(modeln_dec_kwgs))]  # + [None]

        set_attempt_modeln = \
            ['set_attempt_model_name' for _ in range(len(modeln_dec_kwgs))]

        assert true_modelns != set_attempt_modeln

        kwargs_keys = ['model_name' for _ in range(3)]

        self._generic_test_set_once(
            attr_names=kwargs_keys, set_attempt_vals=set_attempt_modeln,
            dec_kwargs=modeln_dec_kwgs, kwargs_keys=kwargs_keys)

    def test_context_getter_outside(self):

        context_dec_kwargs = \
            dict((k, v) for k, v in self.decisions_kwargs.items()
                 if k in self.context_dec_kwgs_keys)

        kwargs_keys = \
            ['context' for _ in range(4)]

        self._generic_test_getter_outside(
            attr_names=kwargs_keys, dec_kwargs=context_dec_kwargs,
            kwargs_keys=kwargs_keys)

    def test_context_getter_returns_des_type(self):

        context_dec_kwargs = \
            dict((k, v) for k, v in self.decisions_kwargs.items()
                 if k in self.context_dec_kwgs_keys)

        kwargs_keys = \
            ['context' for _ in range(4)]

        self._generic_test_getter_returns_des_type(
            attr_names=kwargs_keys, dec_kwargs=context_dec_kwargs,
            type_checker=TestDecisionProperties._context_type_checker)

    def test_context_set_once(self):

        context_dec_kwargs = \
            dict((k, v) for k, v in self.decisions_kwargs.items()
                 if k in self.context_dec_kwgs_keys)

        true_contexts = \
            [v for k, v in self.decisions_kwargs.items()
             if k in self.context_dec_kwgs_keys]  # + [None]

        set_attempt_contexts = \
            [{'text': 'an example variant'}
             for _ in range(len(context_dec_kwargs))]

        assert true_contexts != set_attempt_contexts

        kwargs_keys = \
            ['context' for _ in range(4)]

        self._generic_test_set_once(
            attr_names=kwargs_keys, set_attempt_vals=set_attempt_contexts,
            dec_kwargs=context_dec_kwargs, kwargs_keys=kwargs_keys)

    def test_max_runners_up_getter_outside(self):

        max_runners_up_dec_kwargs = {}
        for dec_kwgs_k, dec_kwgs in self.decisions_kwargs.items():
            curr_dec_kwgs = dict(zip(dec_kwgs.keys(), dec_kwgs.values()))
            curr_dec_kwgs['max_runners_up'] = 15
            max_runners_up_dec_kwargs[dec_kwgs_k] = curr_dec_kwgs

        kwargs_keys = \
            ['max_runners_up' for _ in range(len(max_runners_up_dec_kwargs))]

        self._generic_test_getter_outside(
            attr_names=kwargs_keys, dec_kwargs=max_runners_up_dec_kwargs,
            kwargs_keys=kwargs_keys)

    def test_max_runners_up_getter_returns_des_type(self):

        max_runners_up_dec_kwargs = {}
        for dec_kwgs_k, dec_kwgs in self.decisions_kwargs.items():
            curr_dec_kwgs = dict(zip(dec_kwgs.keys(), dec_kwgs.values()))
            curr_dec_kwgs['max_runners_up'] = 15
            max_runners_up_dec_kwargs[dec_kwgs_k] = curr_dec_kwgs

        kwargs_keys = \
            ['max_runners_up' for _ in range(len(max_runners_up_dec_kwargs))]

        self._generic_test_getter_returns_des_type(
            attr_names=kwargs_keys, dec_kwargs=max_runners_up_dec_kwargs,
            type_checker=TestDecisionProperties.max_runners_up_type_checker)

    def test_max_runners_up_set_once(self):

        max_runners_up_dec_kwargs = {}
        for dec_kwgs_k, dec_kwgs in self.decisions_kwargs.items():
            curr_dec_kwgs = dict(zip(dec_kwgs.keys(), dec_kwgs.values()))
            curr_dec_kwgs['max_runners_up'] = 15
            max_runners_up_dec_kwargs[dec_kwgs_k] = curr_dec_kwgs

        kwargs_keys = \
            ['max_runners_up' for _ in range(len(max_runners_up_dec_kwargs))]

        truemax_runners_up = \
            [v['max_runners_up'] for k, v
             in max_runners_up_dec_kwargs.items()]  # + [None]

        set_attempt_max_runners_up = \
            [100 for _ in range(len(max_runners_up_dec_kwargs))]

        assert truemax_runners_up != set_attempt_max_runners_up

        self._generic_test_set_once(
            attr_names=kwargs_keys, set_attempt_vals=set_attempt_max_runners_up,
            dec_kwargs=max_runners_up_dec_kwargs, kwargs_keys=kwargs_keys)

    def test_memoized_scores_getter_outside(self):

        kwargs_keys = \
            ['memoized_scores' for _ in range(len(self.decisions_kwargs))]

        self._generic_test_getter_outside_after_callable(
            attr_names=kwargs_keys, dec_kwargs=self.decisions_kwargs,
            callable_n='scores')

    def test_memoized_scores_getter_returns_des_type(self):

        attr_names = \
            ['memoized_scores' for _ in range(len(self.decisions_kwargs))]

        # attr_names, dec_kwargs, callable_n, type_checker

        self._generic_test_getter_returns_des_type_after_callable(
            attr_names=attr_names, dec_kwargs=self.decisions_kwargs,
            callable_n='scores',
            type_checker=TestDecisionProperties._memoized_scores_type_checker)

    def test_memoized_scores_set_once(self):

        set_attempt_scores = \
            [list(range(1000)) for _ in range(len(self.decisions_kwargs))]

        attr_names = \
            ['memoized_scores' for _ in range(len(self.decisions_kwargs))]

        # self, attr_names, set_attempt_vals, dec_kwargs, callable_n

        self._generic_test_set_once_after_callable(
            attr_names=attr_names, set_attempt_vals=set_attempt_scores,
            dec_kwargs=self.decisions_kwargs, callable_n='scores')

    def test_memoized_ranked_getter_outside(self):

        kwargs_keys = \
            ['memoized_ranked' for _ in range(len(self.decisions_kwargs))]

        self._generic_test_getter_outside_after_callable(
            attr_names=kwargs_keys, dec_kwargs=self.decisions_kwargs,
            callable_n='ranked')

    def test_memoized_ranked_getter_returns_des_type(self):

        attr_names = \
            ['memoized_ranked' for _ in range(len(self.decisions_kwargs))]

        # attr_names, dec_kwargs, callable_n, type_checker

        self._generic_test_getter_returns_des_type_after_callable(
            attr_names=attr_names, dec_kwargs=self.decisions_kwargs,
            callable_n='ranked',
            type_checker=TestDecisionProperties._memoized_ranked_type_checker)

    def test_memoized_ranked_set_once(self):

        set_attempt_contexts = \
            [[{'text': 'Example variant'} for _ in range(10)]
             for _ in range(len(self.decisions_kwargs))]

        attr_names = \
            ['memoized_ranked' for _ in range(len(self.decisions_kwargs))]

        # self, attr_names, set_attempt_vals, dec_kwargs, callable_n

        self._generic_test_set_once_after_callable(
            attr_names=attr_names, set_attempt_vals=set_attempt_contexts,
            dec_kwargs=self.decisions_kwargs, callable_n='ranked')

    def test_memoized_best_getter_outside(self):

        kwargs_keys = \
            ['memoized_best' for _ in range(len(self.decisions_kwargs))]

        self._generic_test_getter_outside_after_callable(
            attr_names=kwargs_keys, dec_kwargs=self.decisions_kwargs,
            callable_n='best')

    def test_memoized_best_getter_returns_des_type(self):

        attr_names = \
            ['memoized_best' for _ in range(len(self.decisions_kwargs))]

        # attr_names, dec_kwargs, callable_n, type_checker

        self._generic_test_getter_returns_des_type_after_callable(
            attr_names=attr_names, dec_kwargs=self.decisions_kwargs,
            callable_n='best',
            type_checker=TestDecisionProperties._memoized_best_type_checker)

    def test_memoized_best_set_once(self):

        set_attempt_contexts = \
            [{'text': 'Example best variant'}
             for _ in range(len(self.decisions_kwargs))]

        attr_names = \
            ['memoized_best' for _ in range(len(self.decisions_kwargs))]

        # self, attr_names, set_attempt_vals, dec_kwargs, callable_n

        self._generic_test_set_once_after_callable(
            attr_names=attr_names, set_attempt_vals=set_attempt_contexts,
            dec_kwargs=self.decisions_kwargs, callable_n='best')

    def test_scores_returns_des_type(self):

        # attr_names, dec_kwargs, callable_n, type_checker

        self._generic_test_callable_returns_des_type(
            dec_kwargs=self.decisions_kwargs, callable_n='scores',
            type_checker=TestDecisionProperties._memoized_scores_type_checker)

    def test_ranked_returns_des_type(self):

        # attr_names, dec_kwargs, callable_n, type_checker

        self._generic_test_callable_returns_des_type(
            dec_kwargs=self.decisions_kwargs, callable_n='ranked',
            type_checker=TestDecisionProperties._memoized_ranked_type_checker)

    def test_scored_returns_des_type(self):

        # attr_names, dec_kwargs, callable_n, type_checker

        self._generic_test_callable_returns_des_type(
            dec_kwargs=self.decisions_kwargs, callable_n='scored',
            type_checker=TestDecisionProperties._return_scored_type_checker)

    def test_best_returns_des_type(self):

        # attr_names, dec_kwargs, callable_n, type_checker

        self._generic_test_callable_returns_des_type(
            dec_kwargs=self.decisions_kwargs, callable_n='best',
            type_checker=TestDecisionProperties._memoized_best_type_checker)

