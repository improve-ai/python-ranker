from collections.abc import Iterable
from frozendict import frozendict
import inspect
import json
import numpy as np
from pprint import pprint
import sys
from typing import Dict, List, Tuple, Union
from warnings import warn

# TODO -> this is just pre-refactor convenience alias
from models.decision_models import DecisionModel
from utils.gen_purp_utils import constant, get_immutable_iterable


class DecisionUtils:
    @staticmethod
    def simple_context():
        return frozendict({})


class Decision(object):

    @constant
    def __PROTECTED_MEMBERS_ATTR_NAMES() -> Tuple:

        protected_no_prfx = [
            'variants',
            'model',
            'model_name',
            'ranked_variants',
            'context',
            'max_runners_up',
            'memoized_scores',
            'memoized_ranked',
            'memoized_best',
            'track_runners_up']
        return tuple(['__{}'.format(attr_n) for attr_n in protected_no_prfx])

    @constant
    def PROTECTED_MEMBER_INIT_VAL() -> None:
        return None

    @property
    def variants(self) -> Tuple[frozendict] or None:
        variants: tuple = self.__get_protected_member()
        assert isinstance(variants, tuple) or variants is None
        return variants

    @variants.setter
    def variants(
            self, new_val: Union[List[Dict[str, object]], List[frozendict],
                                 Tuple[Dict[str, object]], Tuple[frozendict]]):

        assert not isinstance(new_val, str)

        conv_new_val = get_immutable_iterable(input_val=new_val)

        self.__set_protected_member_once(new_val=conv_new_val)

    @property
    def model(self) -> DecisionModel or None:
        model: DecisionModel = self.__get_protected_member()
        return model

    @model.setter
    def model(self, new_val: object):
        self.__set_protected_member_once(new_val=new_val)

    @property
    def model_name(self) -> str or None:
        model_name: str = self.__get_protected_member()
        assert isinstance(model_name, str) or model_name is None
        return model_name

    @model_name.setter
    def model_name(self, new_val: str):
        assert isinstance(new_val, str) or new_val is None
        self.__set_protected_member_once(new_val=new_val)

    @property
    def context(self) -> frozendict or None:
        context: dict or frozendict = self.__get_protected_member()

        if context and not isinstance(context, frozendict):
            context = frozendict(context)

        assert isinstance(context, frozendict) or context is None
        return context

    @context.setter
    def context(self, new_val: Dict[str, object] or frozendict):
        assert isinstance(new_val, dict) or isinstance(new_val, frozendict) or \
               new_val is None

        # pprint(inspect.stack()[0][0].f_locals['self'].__class__.__name__)
        # pprint(inspect.stack()[1])

        if new_val and not isinstance(new_val, frozendict):
            new_val = frozendict(new_val)

        self.__set_protected_member_once(new_val=new_val)

    @property
    def max_runners_up(self) -> int or None:
        max_runners_up: int = self.__get_protected_member()
        assert isinstance(max_runners_up, int) or max_runners_up is None
        return max_runners_up

    @max_runners_up.setter
    def max_runners_up(self, new_val: int):
        assert isinstance(new_val, int) or new_val is None
        self.__set_protected_member_once(new_val=new_val)

    @property
    def track_runners_up(self) -> int or None:
        track_runners_up: bool = self.__get_protected_member()
        assert isinstance(track_runners_up, int) or track_runners_up is None
        return track_runners_up

    @track_runners_up.setter
    def track_runners_up(self, new_val: bool):
        assert isinstance(new_val, int) or new_val is None
        self.__set_protected_member_once(new_val=new_val)

    @property
    def memoized_scores(self) -> Tuple[float, ...] or None:
        memoized_scores: Tuple[float, ...] = self.__get_protected_member()
        assert isinstance(memoized_scores, tuple) or memoized_scores is None
        return memoized_scores

    @memoized_scores.setter
    def memoized_scores(self, new_val: Tuple[float, ...]):
        assert isinstance(new_val, tuple) or new_val is None
        self.__set_protected_member_once(new_val=new_val)

    @property
    def memoized_ranked(self) -> Tuple[frozendict] or None:
        memoized_ranked: Tuple[frozendict] = self.__get_protected_member()
        assert isinstance(memoized_ranked, tuple) or memoized_ranked is None
        return memoized_ranked

    @memoized_ranked.setter
    def memoized_ranked(self, new_val: Tuple[frozendict]):
        assert isinstance(new_val, tuple) or new_val is None
        self.__set_protected_member_once(new_val=new_val)

    @property
    def memoized_top_runners_up(self) -> Tuple[frozendict] or None:
        memoized_top_runners_up: Tuple[frozendict] = \
            self.__get_protected_member()
        assert isinstance(memoized_top_runners_up, tuple) \
               or memoized_top_runners_up is None
        return memoized_top_runners_up

    @memoized_top_runners_up.setter
    def memoized_top_runners_up(self, new_val: Tuple[frozendict]):
        assert isinstance(new_val, tuple) or new_val is None
        self.__set_protected_member_once(new_val=new_val)

    @property
    def memoized_best(self) -> frozendict:
        memoized_best: frozendict = self.__get_protected_member()
        assert isinstance(memoized_best, frozendict) or memoized_best is None
        return memoized_best

    @memoized_best.setter
    def memoized_best(self, new_val: dict or frozendict or None):
        assert isinstance(new_val, frozendict) or new_val is None
        self.__set_protected_member_once(new_val=new_val)

    def __init__(
            self, variants: Union[List[Dict[str, object]], List[frozendict],
                                  Tuple[Dict[str, object]], Tuple[frozendict]] = None,
            model: DecisionModel = None,
            ranked_variants: Union[List[Dict[str, object]], List[frozendict],
                                   Tuple[Dict[str, object]], Tuple[frozendict],
                                   None] = None,
            model_name: str = None,
            context: Union[Dict[str, object], frozendict, None] = DecisionUtils.simple_context(),
            max_runners_up: int = 50, **kwargs):

        self.__init_protected_members()

        assert variants or ranked_variants
        self.__set_variants(variants=variants, ranked_variants=ranked_variants)

        self.model = model
        self.model_name = model_name
        self.context = context
        self.max_runners_up = max_runners_up
        self.__set_track_runners_up()

    def __setattr__(self, key, value):
        caller_class_details = \
            inspect.stack()[1][0].f_locals  # ['self'].__class__.__name__
        # print('__setattr__ call')
        # pprint(caller_class_details)
        # TODO finish this up

        if 'self' not in caller_class_details.keys():
            warn('Setting from outside of Decision class is not allowed')
            return

        if not id(caller_class_details['self']) == id(self):
            warn('Setting from outside of Decision class is not allowed')
            return

        super(Decision, self).__setattr__(key, value)

    def __set_track_runners_up(self):
        """
        Wrapper the track_runners_up attr setting procedure

        Returns
        -------
        None
            None

        """

        if len(self.variants) == 1:
            return False

        self.track_runners_up = \
            np.random.rand() < 1 / min(
                len(self.variants) - 1, self.max_runners_up)

    def __set_variants(self, variants: Iterable, ranked_variants: Iterable):
        if ranked_variants:
            self.variants = ranked_variants
            # Old implementation
            # self.ranked_variants = ranked_variants
        else:
            self.variants = variants

    def __init_protected_members(self):
        """
        Wrapper for initial values setting of instance's pritected members

        Returns
        -------
        None
            None

        """
        for prot_attr_n in self.__PROTECTED_MEMBERS_ATTR_NAMES:
            setattr(self, prot_attr_n, self.PROTECTED_MEMBER_INIT_VAL)

    def __set_protected_member_once(
            self, new_val: object, prot_mem_n: str = None):
        """
        Setter of protected members attributes. Allows to set value only if
        previous attribute value is None

        Parameters
        ----------
        new_val: object
            value to be set for the attribute
        prot_mem_n: str
            name of protected member

        Returns
        -------
        None
            None

        """

        caller_class_details = \
            inspect.stack()[2][0].f_locals  # ['self'].__class__.__name__

        if 'self' not in caller_class_details.keys():
            warn('Setting from outside of Decision class is not allowed')
            return

        if not id(caller_class_details['self']) == id(self):
            warn('Setting from outside of Decision class is not allowed')
            return

        if not prot_mem_n:
            prot_mem_n = '__' + str(inspect.stack()[1].function)

        des_prot_mem_val = self.__get_protected_member(prot_mem_n=prot_mem_n)

        if des_prot_mem_val is None:
            setattr(self, prot_mem_n, new_val)
        else:
            warn('This attribute has already been set!')

    def __get_protected_member(self, prot_mem_n: str = None) -> object:
        """
        In-class getter of protected member for property setters convenience

        Parameters
        ----------
        prot_mem_n: str
            name of protected member to fetch

        Returns
        -------
        object
            desired protected member

        """

        if not prot_mem_n:
            prot_mem_n = '__' + str(inspect.stack()[1].function)

        assert prot_mem_n[:2] == '__'
        assert hasattr(self, prot_mem_n)
        return getattr(self, prot_mem_n)

    def __get_variants_for_model(self) -> List[Dict[str, object]]:
        """
        Wrapper creating list of di8cts ingestable by DecisionModel class

        Returns
        -------
        List[Dict[str, object]]
            list of variants

        """
        return [dict(variant) for variant in self.variants]

    def __get_context_for_model(self) -> dict:
        """
        Wrapper creating dict from context

        Returns
        -------
        dict
            self.context converted to a dict

        """
        return dict(self.context)

    def scores(self):
        """
        Returns scores calculated by a cached model

        Returns
        -------
        tuple
            tuple of float scores for variants

        """
        # cached_scores = self.cached_scores
        if self.memoized_scores:
            return self.memoized_scores

        if not self.memoized_scores:
            # call score from model
            if not self.model:
                cached_scores = np.random.normal(size=len(self.variants))
                cached_scores[::-1].sort()
                self.memoized_scores = tuple(cached_scores)
                # ind = np.lexsort(-1 * random_scores)
                # cached_scores = random_scores[ind]
            else:
                # TODO self.variants will be a list or a tuple -> DecisionModel
                #  must support that
                self.memoized_scores = \
                    tuple(self.model.score(
                        variants=self.__get_variants_for_model(),
                        context=self.__get_context_for_model(),
                        return_plain_scores=True,
                        plain_scores_idx=1))
        # self.cached_scores = cached_scores
        return self.memoized_scores

    def ranked(self) -> Tuple[frozendict]:
        """
        Returns ranked variants

        Returns
        -------
        tuple
            tuple of variants ordered from best to worst

        """

        # if ranked are cached then return them
        if self.memoized_ranked:
            return self.memoized_ranked

        # if no model and no cached ranked set cached_ranked to variants and
        # return it
        if not self.model:
            self.memoized_ranked = tuple(self.variants)
            return self.memoized_ranked

        if not self.memoized_ranked:
            # if scores are not yet calculated calculate them
            if not self.memoized_scores:
                self.scores()
            variants_w_scores = \
                np.array([self.variants, self.memoized_scores], dtype=object).T

            sorted_variants_w_scores = \
                variants_w_scores[(variants_w_scores[:, 1] * -1).argsort()]
            # and make sure ranked are cached
            self.memoized_ranked = tuple(sorted_variants_w_scores[:, 0])

            return self.memoized_ranked

    def scored(self) -> List[Tuple[frozendict, float]]:
        """
        Returns list of tuples (<variant>, <variant`s score>)

        Returns
        -------
        List[Tuple[Dict[str, object], float]]
            list of tuples

        """
        # if scores are not yet calculated calculate them
        if not self.memoized_scores:
            self.scores()

        scored_variants = \
            list((variant, score) for variant, score
                 in zip(self.variants, self.memoized_scores))

        return scored_variants

    def best(self) -> frozendict or None:
        """
        Returns best of all cached variants. For behabiour check comments.

        Returns
        -------
        frozendict or None
            frozendict with best variant or None

        """

        if self.track_runners_up and not self.memoized_ranked:
            self.ranked()

        # if result already cached return it
        if self.memoized_best:
            return self.memoized_best

        # if variants are ranked cache best and return it
        # this step assumes that there is no best cached
        if self.memoized_ranked:
            self.memoized_best = self.memoized_ranked[0]
            return self.memoized_best

        # if model is provided use static best_of() to get best and cache it
        # this step assumes that there is no best cached and ranked variants
        # are not yet cached
        if self.model:
            self.memoized_best = \
                self.model.best_of(
                    variants=self.__get_variants_for_model(),
                    context=self.__get_context_for_model())
            return self.memoized_best

        # Use gaussian score to score, cache and return best
        # this probably does not need check for variants since without vairants
        # instance of Decision is pointless
        # this step assumes that there is no best cached, ranked variants
        # are not yet cached and there is no model
        if self.variants:
            self.memoized_best = self.variants[0]
            return self.memoized_best

        return None

    def top_runners_up(self):
        return \
            self.memoized_ranked[1:min(
                len(self.memoized_ranked), self.max_runners_up)]

    @staticmethod
    def simple_context():
        return DecisionUtils.simple_context()


if __name__ == '__main__':
    model_kind = 'mlmodel'
    model_pth = '../artifacts/models/meditations.mlmodel'

    dm = DecisionModel(model_kind=model_kind, model_pth=model_pth)

    # context = frozendict({})
    with open('../artifacts/test_artifacts/sorting_context.json', 'r') as cjson:
        read_str = ''.join(cjson.readlines())
        context = frozendict(json.loads(read_str))

    with open('../artifacts/data/real/meditations.json') as mjson:
        read_str = ''.join(mjson.readlines())
        variants = tuple(json.loads(read_str))

    d = Decision(
        variants=variants[:10], model=dm, context=context)

    def scores(self, variants):
        self.__variants = variants
        print(variants)

    d.scores = scores

    # d.scores(self=d, variants=variants[:3])
    print(d.variants)

    # print(d.scores())
    # print(d.memoized_scores)
    print(d.ranked()[0])
    print(max(d.memoized_scores))
    for el in d.scored():
        print(el)
    # print(d.scored())
    print('getting best')
    print(d.best())

    print('###### NO VARIANTS CHECK ######')
    d1 = Decision(variants=variants[:10])
    print('\n####')
    for variant in variants[:10]:
        print(variant)

    print('\n#### SCORES CALL')
    print(d1.scores())
    print(np.median(d1.scores()))

    print('\n#### RANKED CALL')
    for ranked_v in d1.ranked():
        print(ranked_v)

    print('\n#### SCORED CALL')
    for scored_v, score_val in zip(d1.scored(), d1.memoized_scores):
        print('{} -> {}'.format(scored_v, score_val))

    print('\n#### BEST CALL')
    print(d1.best())
    # print(d1.track_runners_up)

    tc = 0
    for _ in range(100):
        _d = Decision(variants=variants[:10])
        if _d.track_runners_up is True:
            tc += 1
            print('tracking')
    print(tc)
