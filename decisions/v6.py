from frozendict import frozendict
import inspect
import numpy as np
from typing import Dict, List, Tuple
from warnings import warn

# TODO -> this is just pre-refactor convenience alias
from improve_model_cli import ImproveModel as DecisionModel
from utils.gen_purp_utils import constant


class Decision:

    @constant
    def __PROTECTED_MEMBERS_ATTR_NAMES():

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
    def PROTECTED_MEMBER_INIT_VAL():
        return None

    @property
    def variants(self) -> Tuple[Dict[str, object]]:
        variants: tuple = self.__get_protected_member()
        assert isinstance(variants, tuple) or variants is None
        return variants

    @variants.setter
    def variants(self, new_val: object):
        self.__set_protected_member_once(new_val=new_val)

    @property
    def model(self) -> object:
        model: object = self.__get_protected_member()
        return model

    @model.setter
    def model(self, new_val: object):
        self.__set_protected_member_once(new_val=new_val)

    @property
    def model_name(self) -> str:
        model_name: str = self.__get_protected_member()
        assert isinstance(model_name, str) or model_name is None
        return model_name

    @model_name.setter
    def model_name(self, new_val: str):
        assert isinstance(new_val, str) or new_val is None
        self.__set_protected_member_once(new_val=new_val)

    @property
    def ranked_variants(self) -> Tuple[Dict[str, object]]:
        ranked_variants: tuple = self.__get_protected_member()
        assert isinstance(ranked_variants, tuple) or ranked_variants is None
        return ranked_variants

    @ranked_variants.setter
    def ranked_variants(self, new_val: Tuple[Dict[str, object]]):
        assert isinstance(new_val, tuple) or new_val is None
        self.__set_protected_member_once(new_val=new_val)

    @property
    def context(self) -> frozendict:
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
    def max_runners_up(self) -> int:
        max_runners_up: int = self.__get_protected_member()
        assert isinstance(max_runners_up, int) or max_runners_up is None
        return max_runners_up

    @max_runners_up.setter
    def max_runners_up(self, new_val: int):
        assert isinstance(new_val, int) or new_val is None
        self.__set_protected_member_once(new_val=new_val)

    @property
    def track_runners_up(self) -> int:
        track_runners_up: bool = self.__get_protected_member()
        assert isinstance(track_runners_up, int) or track_runners_up is None
        return track_runners_up

    @track_runners_up.setter
    def track_runners_up(self, new_val: bool):
        assert isinstance(new_val, int) or new_val is None
        self.__set_protected_member_once(new_val=new_val)

    def __init__(
            self, variants: List[Dict[str, object]] = None,
            model: DecisionModel = None,
            ranked_variants: List[Dict[str, object]] = None,
            model_name: str = None,
            context: Dict[str, object] = None, **kwargs):

        self.__init_protected_members()

        self.variants = variants
        self.model = model
        self.ranked_variants = ranked_variants
        self.decision_model_name = model_name
        self.context = context

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
        pass

    def ranked(self):
        pass

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
    pass
