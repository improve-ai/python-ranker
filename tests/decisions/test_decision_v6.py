from  collections.abc import Iterable
from copy import deepcopy
from frozendict import frozendict
import json
import os
from pytest import fixture
from typing import List
from unittest import TestCase

from decisions.v6 import Decision
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

    def _generic_test_getter_returns_immutable(
            self, attr_names, dec_kwargs, kwargs_keys, type_checker):
        true_vals_list = \
            [d_kwgs.get(kwgs_k) for d_kwgs, kwgs_k
             in zip(dec_kwargs.values(), kwargs_keys)]

        for case, decision_kwgs, true_v, attr_n in zip(
                dec_kwargs.keys(), dec_kwargs.values(),
                true_vals_list, attr_names):
            print('asserting: {}'.format(case))
            curr_decision = Decision(**decision_kwgs)
            type_checker(getattr(curr_decision, attr_n))

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
            setattr(curr_decision, attr_n, set_attempt_v)
            assert getattr(curr_decision, attr_n) != set_attempt_v

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
    def _modeln_type_checker(modeln):
        assert isinstance(modeln, str)

    @staticmethod
    def max_runners_up_type_checker(max_runners_up):
        assert isinstance(max_runners_up, int)

    @staticmethod
    def _collection_immutable_type_checker(chkd_val):
        if chkd_val is None:
            assert True
            return

        if isinstance(chkd_val, Iterable):
            assert not isinstance(chkd_val, str)
            assert isinstance(chkd_val, frozendict) or isinstance(chkd_val, tuple)
            if isinstance(chkd_val, tuple):
                for el in chkd_val:
                    if not isinstance(el, str) and isinstance(el, Iterable):
                        TestDecisionProperties._collection_immutable_type_checker(el)
            elif isinstance(chkd_val, frozendict):
                for el_k, el in chkd_val.items():
                    if not isinstance(el, str) and isinstance(el, Iterable):
                        TestDecisionProperties._collection_immutable_type_checker(el)
            else:
                assert False

    def test_variants_getter_outside(self):

        kwargs_keys = \
            ['variants' for _ in range(3)] + \
            ['ranked_variants' for _ in range(3)]

        attr_names = ['variants' for _ in range(6)]

        self._generic_test_getter_outside(
            attr_names=attr_names, dec_kwargs=self.decisions_kwargs,
            kwargs_keys=kwargs_keys)

    def test_variants_getter_returns_immutable(self):

        kwargs_keys = \
            ['variants' for _ in range(3)] + \
            ['ranked_variants' for _ in range(3)]

        attr_names = ['variants' for _ in range(6)]

        self._generic_test_getter_returns_immutable(
            attr_names=attr_names, dec_kwargs=self.decisions_kwargs,
            kwargs_keys=kwargs_keys,
            type_checker=TestDecisionProperties._collection_immutable_type_checker)

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

    def test_variants_set_mutable_get_immutable(self):

        attr_names = ['variants' for _ in range(6)]
        kwargs_keys = \
            ['variants' for _ in range(3)] + \
            ['ranked_variants' for _ in range(3)]

        self._generic_test_set_mutable_get_immutable(
            attr_names=attr_names, dec_kwargs=self.decisions_kwargs,
            kwargs_keys=kwargs_keys, mutable_maker=lambda x: [dict(el) for el in x],
            type_checker=TestDecisionProperties._collection_immutable_type_checker)

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

    def test_modeln_getter_returns_immutable(self):

        kwargs_keys = ['model_name' for _ in range(3)]
        modeln_dec_kwgs = \
            dict((k, v) for k, v in self.decisions_kwargs.items()
                 if k in self.modeln_dec_kwgs_keys)

        self._generic_test_getter_returns_immutable(
            attr_names=kwargs_keys, dec_kwargs=modeln_dec_kwgs,
            kwargs_keys=kwargs_keys,
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

    def test_context_getter_returns_immutable(self):

        context_dec_kwargs = \
            dict((k, v) for k, v in self.decisions_kwargs.items()
                 if k in self.context_dec_kwgs_keys)

        kwargs_keys = \
            ['context' for _ in range(4)]

        self._generic_test_getter_returns_immutable(
            attr_names=kwargs_keys, dec_kwargs=context_dec_kwargs,
            kwargs_keys=kwargs_keys,
            type_checker=TestDecisionProperties._collection_immutable_type_checker)

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

    def test_context_set_mutable_get_immutable(self):

        context_dec_kwargs = \
            dict((k, v) for k, v in self.decisions_kwargs.items()
                 if k in self.context_dec_kwgs_keys)

        kwargs_keys = \
            ['context' for _ in range(4)]

        self._generic_test_set_mutable_get_immutable(
            attr_names=kwargs_keys, dec_kwargs=context_dec_kwargs,
            kwargs_keys=kwargs_keys, mutable_maker=lambda x: dict(x) if x else x,
            type_checker=TestDecisionProperties._collection_immutable_type_checker)

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

    def test_max_runners_up_getter_returns_immutable(self):

        max_runners_up_dec_kwargs = {}
        for dec_kwgs_k, dec_kwgs in self.decisions_kwargs.items():
            curr_dec_kwgs = dict(zip(dec_kwgs.keys(), dec_kwgs.values()))
            curr_dec_kwgs['max_runners_up'] = 15
            max_runners_up_dec_kwargs[dec_kwgs_k] = curr_dec_kwgs

        kwargs_keys = \
            ['max_runners_up' for _ in range(len(max_runners_up_dec_kwargs))]

        self._generic_test_getter_returns_immutable(
            attr_names=kwargs_keys, dec_kwargs=max_runners_up_dec_kwargs,
            kwargs_keys=kwargs_keys,
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

    def test_context_getter_outside(self):
        context_dec_kwargs = \
            dict((k, v) for k, v in self.decisions_kwargs.items()
                 if k in self.context_dec_kwgs_keys)

        kwargs_keys = \
            ['context' for _ in range(4)]

        self._generic_test_getter_outside(
            attr_names=kwargs_keys, dec_kwargs=context_dec_kwargs,
            kwargs_keys=kwargs_keys)

    def test_context_getter_returns_immutable(self):
        context_dec_kwargs = \
            dict((k, v) for k, v in self.decisions_kwargs.items()
                 if k in self.context_dec_kwgs_keys)

        kwargs_keys = \
            ['context' for _ in range(4)]

        self._generic_test_getter_returns_immutable(
            attr_names=kwargs_keys, dec_kwargs=context_dec_kwargs,
            kwargs_keys=kwargs_keys,
            type_checker=TestDecisionProperties._collection_immutable_type_checker)

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

    def test_context_set_mutable_get_immutable(self):
        context_dec_kwargs = \
            dict((k, v) for k, v in self.decisions_kwargs.items()
                 if k in self.context_dec_kwgs_keys)

        kwargs_keys = \
            ['context' for _ in range(4)]

        self._generic_test_set_mutable_get_immutable(
            attr_names=kwargs_keys, dec_kwargs=context_dec_kwargs,
            kwargs_keys=kwargs_keys, mutable_maker=lambda x: dict(x) if x else x,
            type_checker=TestDecisionProperties._collection_immutable_type_checker)