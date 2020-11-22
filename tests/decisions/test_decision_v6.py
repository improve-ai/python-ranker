from collections.abc import Iterable
from copy import deepcopy
from frozendict import frozendict
import json
import numpy as np
import os
from pytest import fixture, raises
import sys
from typing import Dict, List
from unittest import TestCase

sys.path.append(os.sep.join(str(os.path.abspath(__file__)).split(os.sep)[:-3]))

from decisions.v6_fast import Decision
from models.decision_models import DecisionModel


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

    @property
    def model_seed_tracks(self) -> int:
        return self._model_seed_tracks

    @model_seed_tracks.setter
    def model_seed_tracks(self, new_val: int):
        assert isinstance(new_val, int)
        self._model_seed_tracks = new_val

    @property
    def model_seed_not_tracks(self) -> int:
        return self._model_seed_not_tracks

    @model_seed_not_tracks.setter
    def model_seed_not_tracks(self, new_val: int):
        assert isinstance(new_val, int)
        self._model_seed_not_tracks = new_val

    @property
    def max_runners_up_test_val(self) -> int:
        return self._max_runners_up_test_val

    @max_runners_up_test_val.setter
    def max_runners_up_test_val(self, new_val: int):
        assert isinstance(new_val, int)
        self._max_runners_up_test_val = new_val

    @property
    def scores_seed(self) -> int:
        return self._scores_seed

    @scores_seed.setter
    def scores_seed(self, new_val: int):
        assert isinstance(new_val, int)
        self._scores_seed = new_val

    @property
    def model_scored_benchmark_variants(self) -> List[Dict[str, object]]:
        return self._model_scored_benchmark_variants

    @model_scored_benchmark_variants.setter
    def model_scored_benchmark_variants(self, new_val: List[Dict[str, object]]):
        self._model_scored_benchmark_variants = new_val

    @property
    def model_scored_benchmark_variants_scores(self) -> List[float]:
        return self._model_scored_benchmark_variants_scores

    @model_scored_benchmark_variants_scores.setter
    def model_scored_benchmark_variants_scores(self, new_val: List[float]):
        self._model_scored_benchmark_variants_scores = new_val

    @property
    def ranked_seed(self) -> List[Dict[str, object]]:
        return self._ranked_seed

    @ranked_seed.setter
    def ranked_seed(self, new_val: List[Dict[str, object]]):
        self._ranked_seed = new_val

    @property
    def model_ranked_benchmark_variants(self) -> List[Dict[str, object]]:
        return self._model_ranked_benchmark_variants

    @model_ranked_benchmark_variants.setter
    def model_ranked_benchmark_variants(self, new_val: List[Dict[str, object]]):
        self._model_ranked_benchmark_variants = new_val

    @property
    def model_ranked_benchmark_variants_scores(self) -> List[float]:
        return self._model_ranked_benchmark_variants_scores

    @model_ranked_benchmark_variants_scores.setter
    def model_ranked_benchmark_variants_scores(self, new_val: List[float]):
        self._model_ranked_benchmark_variants_scores = new_val

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
            # self.variants = \
            #     tuple(frozendict(v) for v in json.loads(read_str))[:10]
            self.variants = json.loads(read_str)[:10]

        self.decisions_kwargs = {
            # 'plain': Decision(),
            'variants_model': {
                'variants': self.variants, 'model': self.artifacts_model},
            'variants_model_context': {
                'variants': self.variants, 'model': self.artifacts_model,
                'context': self.context},
            'variants_model_context_nulled': {
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

        with open(os.getenv('VARIANTS_FOR_MODEL_SCORES_PTH')) as tvmjson:
            read_str = ''.join(tvmjson.readlines())
            self.model_scored_benchmark_variants = json.loads(read_str)

        with open(os.getenv('SCORES_FOR_MODEL_SCORES_PTH')) as tvsjson:
            read_str = ''.join(tvsjson.readlines())
            self.model_scored_benchmark_variants_scores = json.loads(read_str)

        with open(os.getenv('RANKED_VARIANTS_FOR_MODEL_PTH')) as rvmjson:
            read_str = ''.join(rvmjson.readlines())
            self.model_ranked_benchmark_variants = json.loads(read_str)

        with open(os.getenv('RANKED_SCORES_FOR_MODEL_PTH')) as rvsjson:
            read_str = ''.join(rvsjson.readlines())
            self.model_ranked_benchmark_variants_scores = json.loads(read_str)

        self.model_seed_tracks = int(os.getenv("MODEL_SEED_TRACKS"))
        self.model_seed_not_tracks = int(os.getenv("MODEL_SEED_NOT_TRACKS"))

        self.scores_seed = int(os.getenv("SCORES_SEED"))
        self.ranked_seed = int(os.getenv("RANKED_SEED"))

        self.max_runners_up_test_val = int(os.getenv("MAX_RUNNERS_UP_TEST_VAL"))

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
            print('############')
            print(getattr(curr_decision, attr_n))

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
            self, attr_names, set_attempt_vals, dec_kwargs):

        for case, decision_kwgs, set_attempt_v, attr_n in zip(
                dec_kwargs.keys(), dec_kwargs.values(),
                set_attempt_vals, attr_names):

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
        print('### memoized_best ###')
        print(memoized_best)
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
        # kwargs_keys = \
        #     ['variants' for _ in range(3)] + \
        #     ['ranked_variants' for _ in range(3)]

        self._generic_test_set_once(
            attr_names=attr_names, set_attempt_vals=set_attempt_variants,
            dec_kwargs=self.decisions_kwargs)

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
            dec_kwargs=modeln_dec_kwgs)

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
            dec_kwargs=context_dec_kwargs)

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
            dec_kwargs=max_runners_up_dec_kwargs)

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

    def test_track_runners_up_set_once(self):

        attr_names = \
            ['track_runners_up' for _ in range(len(self.decisions_kwargs))]

        set_attempt_variants = [-1 for _ in attr_names]

        self._generic_test_set_once(
            attr_names=attr_names, set_attempt_vals=set_attempt_variants,
            dec_kwargs=self.decisions_kwargs)

    def test_track_runners_up_return_value(self):

        for case, dec_kwgs in self.decisions_kwargs.items():
            np.random.seed(self.model_seed_not_tracks)
            curr_dec = Decision(**dec_kwgs)
            assert not curr_dec.track_runners_up

        for case, dec_kwgs in self.decisions_kwargs.items():
            np.random.seed(self.model_seed_tracks)
            curr_dec = Decision(**dec_kwgs)
            assert curr_dec.track_runners_up

    def test_track_runners_up_return_value_single_variant(self):

        variants_keys = \
            ['variants' for _ in range(3)] + \
            ['ranked_variants' for _ in range(3)]

        for seed in [self.model_seed_not_tracks, self.model_seed_tracks]:
            for case, dec_kwgs, v_key in zip(
                    self.decisions_kwargs.keys(),
                    self.decisions_kwargs.values(), variants_keys):
                print(dec_kwgs[v_key][0])
                dec_kwgs[v_key] = [dec_kwgs[v_key][0]]
                np.random.seed(seed)
                curr_dec = Decision(**dec_kwgs)
                assert not curr_dec.track_runners_up

    def test_top_runners_up_return_type(self):

        for case, dec_kwgs in self.decisions_kwargs.items():
            print('asserting: {}'.format(case))
            np.random.seed(self.model_seed_tracks)
            curr_dec = Decision(**dec_kwgs)
            curr_dec.ranked()
            tru = curr_dec.top_runners_up()

            assert \
                isinstance(tru, list) or isinstance(tru, np.ndarray) \
                or tru is None

            if isinstance(tru, list):
                for v in tru:
                    assert isinstance(v, dict)

    def test_top_runners_up_return_value(self):

        for case, dec_kwgs in self.decisions_kwargs.items():
            np.random.seed(self.model_seed_not_tracks)
            curr_dec = Decision(**dec_kwgs)
            curr_dec.ranked()
            runners_up = curr_dec.top_runners_up()
            assert runners_up is not None

        for case, dec_kwgs in self.decisions_kwargs.items():
            np.random.seed(self.model_seed_tracks)
            curr_dec = Decision(**dec_kwgs)
            curr_dec.ranked()
            runners_up = curr_dec.top_runners_up()
            assert runners_up is not None

    def test__set_track_runners_up_single_variant(self):

        variants_keys = \
            ['variants' for _ in range(3)] + \
            ['ranked_variants' for _ in range(3)]

        for case, dec_kwgs, v_key in zip(
                self.decisions_kwargs.keys(), self.decisions_kwargs.values(),
                variants_keys):
            dec_kwgs[v_key] = [dec_kwgs[v_key][0]]
            np.random.seed(self.model_seed_tracks)
            curr_dec = Decision(**dec_kwgs)
            assert not curr_dec.track_runners_up

    def test_scores_first_call_no_model(self):
        # this should return random numbers largest to smallest

        des_decision_kwgs_keys = \
            ['ranked_variants_modeln', 'ranked_variants_modeln_context',
             'ranked_variants_modeln_context_nulled']

        des_decision_kwgs = \
            dict(zip(
                des_decision_kwgs_keys,
                [self.decisions_kwargs.get(k) for k in des_decision_kwgs_keys]))

        np.random.seed(self.scores_seed)
        benchmark_cached_scores = np.random.normal(size=len(self.variants))
        benchmark_cached_scores[::-1].sort()

        for case, dec_kwgs in des_decision_kwgs.items():

            np.random.seed(self.scores_seed)
            curr_dec = Decision(**dec_kwgs)
            np.random.seed(self.scores_seed)
            curr_scores = curr_dec.scores()

            np.testing.assert_array_equal(benchmark_cached_scores, curr_scores)

    def test_scores_second_call_no_model(self):

        des_decision_kwgs_keys = \
            ['ranked_variants_modeln', 'ranked_variants_modeln_context',
             'ranked_variants_modeln_context_nulled']

        des_decision_kwgs = \
            dict(zip(
                des_decision_kwgs_keys,
                [self.decisions_kwargs.get(k) for k in des_decision_kwgs_keys]))

        np.random.seed(self.scores_seed)
        benchmark_cached_scores = np.random.normal(size=len(self.variants))
        benchmark_cached_scores[::-1].sort()

        for case, dec_kwgs in des_decision_kwgs.items():

            np.random.seed(self.scores_seed)
            curr_dec = Decision(**dec_kwgs)
            np.random.seed(self.scores_seed)
            first_call_curr_scores = curr_dec.scores()
            second_call_curr_scores = curr_dec.scores()

            assert second_call_curr_scores is not None

            np.testing.assert_array_equal(
                benchmark_cached_scores, first_call_curr_scores)

            np.testing.assert_array_equal(
                benchmark_cached_scores, second_call_curr_scores)

            np.testing.assert_array_equal(
                first_call_curr_scores, second_call_curr_scores)

    def test_scores_first_call_w_model(self):

        des_decision_kwgs_keys = \
            ['variants_model', 'variants_model_context',
             'variants_model_context_nulled']

        des_decision_kwgs = \
            dict(zip(
                des_decision_kwgs_keys,
                [self.decisions_kwargs.get(k) for k in des_decision_kwgs_keys]))

        for case, dec_kwgs in des_decision_kwgs.items():
            dec_kwgs['variants'] = self.model_scored_benchmark_variants
            curr_dec = Decision(**dec_kwgs)
            np.random.seed(self.scores_seed)
            curr_scores = curr_dec.scores()

            asserted_curr_scores = curr_scores
            if not isinstance(asserted_curr_scores, list):
                asserted_curr_scores = list(curr_scores)

            np.testing.assert_array_almost_equal(
                self.model_scored_benchmark_variants_scores,
                asserted_curr_scores)

    def test_scores_second_call_w_model(self):

        des_decision_kwgs_keys = \
            ['variants_model', 'variants_model_context',
             'variants_model_context_nulled']

        des_decision_kwgs = \
            dict(zip(
                des_decision_kwgs_keys,
                [self.decisions_kwargs.get(k) for k in des_decision_kwgs_keys]))

        for case, dec_kwgs in des_decision_kwgs.items():
            dec_kwgs['variants'] = self.model_scored_benchmark_variants
            curr_dec = Decision(**dec_kwgs)
            np.random.seed(self.scores_seed)
            first_call_curr_scores = curr_dec.scores()
            second_call_curr_scores = curr_dec.scores()

            assert second_call_curr_scores is not None

            asserted_first_curr_scores_curr_scores = first_call_curr_scores
            asserted_second_curr_scores_curr_scores = second_call_curr_scores

            if not isinstance(asserted_first_curr_scores_curr_scores, list):
                asserted_first_curr_scores_curr_scores = \
                    list(first_call_curr_scores)

            if not isinstance(asserted_second_curr_scores_curr_scores, list):
                asserted_second_curr_scores_curr_scores = \
                    list(second_call_curr_scores)

            np.testing.assert_array_almost_equal(
                self.model_scored_benchmark_variants_scores,
                asserted_first_curr_scores_curr_scores)

            np.testing.assert_array_almost_equal(
                self.model_scored_benchmark_variants_scores,
                asserted_second_curr_scores_curr_scores)

    def test_ranked_first_call_no_scores_no_model(self):

        des_decision_kwgs_keys = \
            ['ranked_variants_modeln', 'ranked_variants_modeln_context',
             'ranked_variants_modeln_context_nulled']

        des_decision_kwgs = \
            dict(zip(
                des_decision_kwgs_keys,
                [self.decisions_kwargs.get(k) for k in des_decision_kwgs_keys]))

        np.random.seed(self.ranked_seed)
        benchmark_cached_ranked_scores = np.random.normal(size=len(self.variants))
        benchmark_cached_ranked_scores[::-1].sort()

        for case, dec_kwgs in des_decision_kwgs.items():
            curr_dec = Decision(**dec_kwgs)
            np.random.seed(self.ranked_seed)
            curr_ranked = curr_dec.ranked()
            curr_scores = curr_dec.memoized_scores

            assert curr_ranked is not None
            assert curr_scores is not None

            np.testing.assert_array_equal(
                benchmark_cached_ranked_scores, curr_scores)
            np.testing.assert_array_equal(
                dec_kwgs['ranked_variants'], curr_ranked)

    def test_ranked_second_call_no_scores_no_model(self):

        des_decision_kwgs_keys = \
            ['ranked_variants_modeln', 'ranked_variants_modeln_context',
             'ranked_variants_modeln_context_nulled']

        des_decision_kwgs = \
            dict(zip(
                des_decision_kwgs_keys,
                [self.decisions_kwargs.get(k) for k in des_decision_kwgs_keys]))

        np.random.seed(self.ranked_seed)
        benchmark_cached_ranked_scores = \
            np.random.normal(size=len(self.variants))
        benchmark_cached_ranked_scores[::-1].sort()

        for case, dec_kwgs in des_decision_kwgs.items():
            curr_dec = Decision(**dec_kwgs)
            np.random.seed(self.ranked_seed)
            first_call_curr_ranked = deepcopy(curr_dec.ranked())
            first_call_curr_scores = deepcopy(curr_dec.memoized_scores)
            second_call_curr_ranked = deepcopy(curr_dec.ranked())
            second_call_curr_scores = deepcopy(curr_dec.memoized_scores)

            for el in [first_call_curr_ranked, first_call_curr_scores,
                       second_call_curr_ranked, second_call_curr_scores]:
                assert el is not None

            np.testing.assert_array_equal(
                benchmark_cached_ranked_scores, first_call_curr_scores)

            np.testing.assert_array_equal(
                benchmark_cached_ranked_scores, second_call_curr_scores)

            np.testing.assert_array_equal(
                dec_kwgs['ranked_variants'], first_call_curr_ranked)

            np.testing.assert_array_equal(
                dec_kwgs['ranked_variants'], second_call_curr_ranked)

    def test_ranked_first_call_w_model(self):

        des_decision_kwgs_keys = \
            ['variants_model', 'variants_model_context',
             'variants_model_context_nulled']

        des_decision_kwgs = \
            dict(zip(
                des_decision_kwgs_keys,
                [self.decisions_kwargs.get(k) for k in des_decision_kwgs_keys]))

        for case, dec_kwgs in des_decision_kwgs.items():
            dec_kwgs['variants'] = self.model_scored_benchmark_variants
            curr_dec = Decision(**dec_kwgs)
            np.random.seed(self.scores_seed)
            curr_ranked = curr_dec.ranked()
            curr_scores = curr_dec.memoized_scores

            assert curr_ranked is not None
            assert curr_dec.memoized_ranked is not None

            asserted_curr_ranked = curr_ranked
            if not isinstance(asserted_curr_ranked, list):
                asserted_curr_ranked = list(curr_ranked)

            np.testing.assert_array_equal(
                self.model_ranked_benchmark_variants,
                asserted_curr_ranked)

            np.testing.assert_array_equal(
                curr_dec.memoized_ranked
                if isinstance(curr_dec.memoized_ranked, list)
                else list(curr_dec.memoized_ranked),
                asserted_curr_ranked)

            np.testing.assert_array_almost_equal(
                [el for el in
                 reversed(sorted(curr_scores if isinstance(curr_scores, list)
                                 else list(curr_scores)))],
                self.model_ranked_benchmark_variants_scores)

    def test_ranked_second_call_w_model(self):

        des_decision_kwgs_keys = \
            ['variants_model', 'variants_model_context',
             'variants_model_context_nulled']

        des_decision_kwgs = \
            dict(zip(
                des_decision_kwgs_keys,
                [self.decisions_kwargs.get(k) for k in des_decision_kwgs_keys]))

        for case, dec_kwgs in des_decision_kwgs.items():
            dec_kwgs['variants'] = self.model_scored_benchmark_variants
            curr_dec = Decision(**dec_kwgs)
            np.random.seed(self.scores_seed)

            first_call_curr_ranked = deepcopy(curr_dec.ranked())
            first_call_curr_scores = deepcopy(curr_dec.memoized_scores)
            second_call_curr_ranked = deepcopy(curr_dec.ranked())
            second_call_curr_scores = deepcopy(curr_dec.memoized_scores)

            for el in [first_call_curr_ranked, first_call_curr_scores,
                       second_call_curr_ranked, second_call_curr_scores]:
                assert el is not None

            asserted_first_call_curr_ranked = first_call_curr_ranked
            if not isinstance(asserted_first_call_curr_ranked, list):
                asserted_first_call_curr_ranked = list(first_call_curr_ranked)

            asserted_first_call_curr_scores = first_call_curr_scores
            if not isinstance(asserted_first_call_curr_scores, list):
                asserted_first_call_curr_scores = list(first_call_curr_scores)

            asserted_second_call_curr_ranked = second_call_curr_ranked
            if not isinstance(asserted_second_call_curr_ranked, list):
                asserted_second_call_curr_ranked = list(second_call_curr_ranked)

            asserted_second_call_curr_scores = second_call_curr_scores
            if not isinstance(asserted_second_call_curr_scores, list):
                asserted_second_call_curr_scores = list(second_call_curr_scores)

            np.testing.assert_array_equal(
                self.model_ranked_benchmark_variants,
                asserted_first_call_curr_ranked)

            np.testing.assert_array_equal(
                self.model_ranked_benchmark_variants,
                asserted_second_call_curr_ranked)

            np.testing.assert_array_equal(
                curr_dec.memoized_ranked
                if isinstance(curr_dec.memoized_ranked, list)
                else list(curr_dec.memoized_ranked),
                asserted_first_call_curr_ranked)

            np.testing.assert_array_equal(
                curr_dec.memoized_ranked
                if isinstance(curr_dec.memoized_ranked, list)
                else list(curr_dec.memoized_ranked),
                asserted_first_call_curr_ranked)

            np.testing.assert_array_almost_equal(
                [el for el in
                 reversed(sorted(
                     asserted_first_call_curr_scores
                     if isinstance(asserted_first_call_curr_scores, list)
                     else list(asserted_first_call_curr_scores)))],
                self.model_ranked_benchmark_variants_scores)

            np.testing.assert_array_almost_equal(
                [el for el in
                 reversed(sorted(
                     asserted_second_call_curr_scores
                     if isinstance(asserted_second_call_curr_scores, list)
                     else list(asserted_second_call_curr_scores)))],
                self.model_ranked_benchmark_variants_scores)

            np.testing.assert_array_almost_equal(
                [el for el in
                 reversed(sorted(
                     curr_dec.memoized_scores
                     if isinstance(curr_dec.memoized_scores, list)
                     else list(curr_dec.memoized_scores)))],
                self.model_ranked_benchmark_variants_scores)
