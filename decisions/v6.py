from collections.abc import Iterable
from frozendict import frozendict
import inspect
import numpy as np
from typing import Dict, List, Tuple
from warnings import warn

# TODO -> this is just pre-refactor convenience alias
from models.decision_models import DecisionModel
from improve_model_cli import DecisionModel as DecisionModel
from utils.gen_purp_utils import constant


class Decision:

    @constant
    def __PROTECTED_MEMBERS_ATTR_NAMES() -> Tuple:

        protected_no_prfx = [
            'variants',
            'model',
            'model_name',
            'ranked_variants',
            'context',
            'max_runners_up',
            'scores',
            'ranked',
            'scored']
        return tuple(['__{}'.format(attr_n) for attr_n in protected_no_prfx])

    @constant
    def PROTECTED_MEMBER_INIT_VAL() -> None:
        return None

    @property
    def variants(self) -> Tuple[Dict[str, object]] or None:
        variants: tuple = self.__get_protected_member()
        assert isinstance(variants, tuple) or variants is None
        return variants

    @variants.setter
    def variants(self, new_val: Tuple[Dict[str, object]]):
        self.__set_protected_member_once(new_val=new_val)

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

    # @property
    # def ranked_variants(self) -> Tuple[Dict[str, object]] or None:
    #     ranked_variants: tuple = self.__get_protected_member()
    #     assert isinstance(ranked_variants, tuple) or ranked_variants is None
    #     return ranked_variants
    #
    # @ranked_variants.setter
    # def ranked_variants(self, new_val: Tuple[Dict[str, object]]):
    #     assert isinstance(new_val, tuple) or new_val is None
    #     self.__set_protected_member_once(new_val=new_val)

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
    def cached_scores(self) -> Tuple[object, ...] or None:
        cached_scores: Tuple[object, ...] = self.__get_protected_member()
        assert isinstance(cached_scores, tuple) or cached_scores is None
        return cached_scores

    @cached_scores.setter
    def cached_scores(self, new_val: Tuple[Dict[str, object]]):
        assert isinstance(new_val, tuple) or new_val is None
        self.__set_protected_member_once(new_val=new_val)

    @property
    def cached_ranked(self) -> Tuple[Dict[str, object]] or None:
        cached_ranked: Tuple[Dict[str, object]] = self.__get_protected_member()
        assert isinstance(cached_ranked, tuple) or cached_ranked is None
        return cached_ranked

    @cached_ranked.setter
    def cached_ranked(self, new_val: Tuple[Dict[str, object]]):
        assert isinstance(new_val, tuple) or new_val is None
        self.__set_protected_member_once(new_val=new_val)

    @property
    def cached_top_runners_up(self) -> Tuple[Dict[str, object]] or None:
        cached_top_runners_up: Tuple[Dict[str, object]] = \
            self.__get_protected_member()
        assert isinstance(cached_top_runners_up, tuple) \
               or cached_top_runners_up is None
        return cached_top_runners_up

    @cached_top_runners_up.setter
    def cached_top_runners_up(self, new_val: Tuple[Dict[str, object]]):
        assert isinstance(new_val, tuple) or new_val is None
        self.__set_protected_member_once(new_val=new_val)

    def __init__(
            self, variants: List[Dict[str, object]] = None,
            model: DecisionModel = None,
            ranked_variants: List[Dict[str, object]] = None,
            model_name: str = None,
            context: Dict[str, object] = None,
            max_runners_up: int = 50, **kwargs):

        self.__init_protected_members()

        assert variants or ranked_variants
        self.__set_variants(variants=variants, ranked_variants=ranked_variants)

        self.model = model
        self.decision_model_name = model_name
        self.context = context
        self.max_runners_up = max_runners_up

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
            value to be set for the attribuetd
        prot_mem_n: str
            name of protected member

        Returns
        -------
        None
            None

        """

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

    def scores(self):
        cached_scores = self.cached_scores
        if cached_scores:
            return cached_scores

        if not cached_scores:
            # call score from model
            if not self.model:
                cached_scores = np.random.normal(size=len(self.variants))
                cached_scores[::-1].sort()
                # ind = np.lexsort(-1 * random_scores)
                # cached_scores = random_scores[ind]
            else:
                # TODO self.variants will be a list or a tuple -> DecisionModel
                #  must support that
                cached_scores = \
                    self.model.score(
                        variants=self.variants, context=self.context,
                        return_plain_scores=True, plain_scores_idx=1)
        self.cached_scores = cached_scores

    def ranked(self):

        # if ranked are cached then return them
        if self.cached_ranked:
            return self.cached_ranked

        # if no model and no cached ranked set cached_ranked to variants and
        # return it
        if not self.model:
            self.cached_ranked = tuple(self.variants)
            return self.cached_ranked

        # if scores are not yet calculated calculate them
        if not self.cached_scores:
            self.scores()

        if not self.cached_ranked:
            variants_w_scores = \
                np.array([self.variants, self.cached_scores]).T
            sorted_variants_w_scores = \
                variants_w_scores[(variants_w_scores[: 1] * -1).argsort()]
            # and make sure ranked are cached
            self.cached_ranked = tuple(sorted_variants_w_scores[:, 0])

            return self.cached_ranked

    def scored(self):
        pass

    def best(self):
        pass

    def top_runners_up(self):
        pass

    @staticmethod
    def simple_context():
        return {}


if __name__ == '__main__':
    d = Decision()
    d.context = {'a': [1, 2, 3]}
    print('should warn')
    d.context = {'a': [1, 2]}
    print('should warn')
    d.context = None
    print('should warn')
    d.context = {'a': [1, 2]}
    print(d.context)

    # print(inspect.stack()[0].function)

    pass
