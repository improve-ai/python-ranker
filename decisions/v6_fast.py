from collections.abc import Iterable
import json
import numpy as np
from time import time
import os
import sys
from typing import Dict, List, Tuple

sys.path.append('/Users/os/Projs/python-sdk')

# TODO -> this is just pre-refactor convenience alias
from models.decision_models import DecisionModel


class Decision(object):

    @property
    def variants(self) -> List[Dict[str, object]] or None:
        return self.__variants

    @property
    def model(self) -> DecisionModel or None:
        return self.__model

    @property
    def model_name(self) -> str or None:
        return self.__model_name

    @property
    def context(self) -> Dict[str, object] or None:
        return self.__context

    @property
    def max_runners_up(self) -> int or None:
        return self.__max_runners_up

    @property
    def track_runners_up(self) -> int or None:
        return self.__track_runners_up

    @property
    def memoized_scores(self) -> List[float] or None:
        return self.__memoized_scores

    @property
    def memoized_ranked(self) -> List[Dict[str, object]] or None:
        return self.__memoized_ranked

    # @property
    # def memoized_top_runners_up(self) -> List[Dict[str, object]] or None:
    #     return self.__memoized_top_runners_up

    @property
    def memoized_best(self) -> dict or None:
        return self.__memoized_best

    def __init__(
            self, variants: List[Dict[str, object]] = None,
            model: DecisionModel = None,
            ranked_variants: List[Dict[str, object]] = None,
            model_name: str = None,
            context: dict = dict(),
            max_runners_up: int = 50, **kwargs):

        assert variants or ranked_variants
        self.__set_variants(variants=variants, ranked_variants=ranked_variants)

        self.__model = model
        self.__model_name = model_name
        self.__context = context
        self.__max_runners_up = max_runners_up
        self.__set_track_runners_up()
        self.__memoized_scores = None
        self.__memoized_ranked = None
        # self.__memoized_top_runners_up = None
        self.__memoized_best = None

    def __set_track_runners_up(self):
        """
        Wrapper the track_runners_up attr setting procedure

        Returns
        -------
        None
            None

        """

        if len(self.__variants) == 1:
            self.__track_runners_up = False
        else:
            self.__track_runners_up = \
                np.random.rand() < 1 / min(
                    len(self.__variants) - 1, self.__max_runners_up)

    def __set_variants(
            self, variants: List[Dict[str, object]] or None,
            ranked_variants: List[Dict[str, object]] or None):

        # TODO maybe assert for type? Dict has length but is not a list.
        #  Would this break any functionality?

        if ranked_variants is not None:
            self.__variants = ranked_variants
            # Old implementation
            # self.ranked_variants = ranked_variants
        else:
            self.__variants = variants

    def scores(self):
        """
        Returns scores calculated by a cached model

        Returns
        -------
        tuple
            tuple of float scores for variants

        """
        # cached_scores = self.cached_scores
        if self.__memoized_scores is not None:
            return self.__memoized_scores

        if self.__memoized_scores is None:
            # call score from model
            if self.__model is None:
                # TODO is this valid when ranked_variants are provided ?
                #  I suppose so
                cached_scores = np.random.normal(size=len(self.__variants))
                cached_scores[::-1].sort()
                self.__memoized_scores = cached_scores

            else:
                self.__memoized_scores = \
                    self.model.score(
                        variants=self.__variants,
                        context=self.__context,
                        return_plain_results=True, sigmoid_correction=False,
                        sigmoid_const=0.5, plain_scores_idx=1, be_quick=True)

        return self.__memoized_scores

    def ranked(self) -> List[Dict[str, object]]:
        """
        Returns ranked variants

        Returns
        -------
        tuple
            tuple of variants ordered from best to worst

        """

        # if ranked are cached then return them
        if self.__memoized_ranked is not None:
            return self.__memoized_ranked

        # if no model and no cached ranked set cached_ranked to variants and
        # return it
        if self.__model is None:
            if self.__memoized_scores is None:
                self.scores()
            self.__memoized_ranked = self.__variants
            return self.__memoized_ranked

        if self.__memoized_ranked is None:
            # if scores are not yet calculated calculate them
            if self.__memoized_scores is None:
                self.scores()
            # this is the fast way

            # print(self.__variants)
            # print(self.__memoized_scores)
            #
            # print(any([fel == sel for fel in self.__memoized_scores for sel in self.__memoized_scores]))

            variants_w_scores = \
                np.array(
                    [self.__variants, self.__memoized_scores]).T

            sorted_variants_w_scores = \
                variants_w_scores[(variants_w_scores[:, 1] * -1).argsort()]
            # and make sure ranked are cached
            self.__memoized_ranked = sorted_variants_w_scores[:, 0]

            return self.__memoized_ranked

    def scored(self, max_speed: bool = True) -> Iterable:
        """
        Returns list of tuples (<variant>, <variant`s score>)

        Returns
        -------
        List[Tuple[Dict[str, object], float]]
            list of tuples

        """
        # if scores are not yet calculated calculate them
        if self.__memoized_scores is None:
            self.scores()

        scored_variants = \
            ((variant, score) for variant, score
             in zip(self.__variants, self.__memoized_scores)) if max_speed else\
            [(variant, score) for variant, score
             in zip(self.__variants, self.__memoized_scores)]

        return scored_variants

    def best(self) -> dict or None:
        """
        Returns best of all cached variants. For behabiour check comments.

        Returns
        -------
        frozendict or None
            frozendict with best variant or None

        """

        if self.__track_runners_up and self.__memoized_ranked is None:
            self.ranked()

        # if result already cached return it
        if self.__memoized_best is not None:
            return self.__memoized_best

        # if variants are ranked cache best and return it
        # this step assumes that there is no best cached
        if self.__memoized_ranked is not None:
            self.__memoized_best = self.__memoized_ranked[0]
            return self.__memoized_best

        # if model is provided use static best_of() to get best and cache it
        # this step assumes that there is no best cached and ranked variants
        # are not yet cached
        if self.__model is not None:
            self.__memoized_best = \
                self.model.best_of(
                    variants=self.__variants,
                    context=self.__context)
            return self.__memoized_best

        # Use gaussian score to score, cache and return best
        # this probably does not need check for variants since without vairants
        # instance of Decision is pointless
        # this step assumes that there is no best cached, ranked variants
        # are not yet cached and there is no model
        if self.__variants is not None:
            self.__memoized_best = self.__variants[0]
            return self.__memoized_best

        return None

    def top_runners_up(self) -> Iterable or None:

        # TODO ask if max_runners_up should indicate index or ocount -
        #  I would assume count

        return \
            self.__memoized_ranked[1:min(
                len(self.__memoized_ranked), (self.__max_runners_up + 1))] \
            if self.__memoized_ranked is not None else None

    @staticmethod
    def simple_context() -> dict:
        return dict()


if __name__ == '__main__':
    # model_kind = 'mlmodel'
    model_kind = 'xgb_native'
    # model_pth = '../artifacts/test_artifacts/'
    xgb_model_pth = 'artifacts/models/12_11_2020_verses_conv.xgb'
    # xgb_model_pth = 'artifacts/models/improve-stories-2.0.xgb'
    # xgb_model_pth = 'artifacts/models/improve-messages-2.0.xgb'
    dm = DecisionModel(model_kind=model_kind, model_pth=xgb_model_pth)

    # context = frozendict({})
    with open('artifacts/test_artifacts/sorting_context.json', 'r') as cjson:
        read_str = ''.join(cjson.readlines())
        context = json.loads(read_str)

    # with open('artifacts/data/real/2bs_bible_verses_full.json') as mjson:
    #     read_str = ''.join(mjson.readlines())
    #     variants = json.loads(read_str)

    with open('artifacts/data/real/meditations.json') as mjson:
        read_str = ''.join(mjson.readlines())
        variants = json.loads(read_str)

    # d = Decision(
    #     variants=variants[:10], model=dm, context=context)
    #
    # d.ranked()

    # print(d.variants)
    # print(d.memoized_scores)
    # print(d.memoized_ranked)
    # input('check')
    # st = time()
    # Decision(variants=variants[:400], model=dm, context=context).scores()
    # et = time()
    # print(et - st)

    st = time()
    batch_size = 1000
    # res = [Decision(variants=variants[:500], model=dm, context=context).scored()
    #        for _ in range(batch_size)]
    for seed in range(batch_size):

        np.random.seed(seed)
        d = Decision(variants=variants[:300], model=dm, context=context)

        if d.track_runners_up:
            print(seed)
            break

        # d.best()

    et = time()
    # print((et - st) / batch_size)
    # for el in res[0]:
    #     print(el)
    # input('speed test')
    #
    # # d.scores(self=d, variants=variants[:3])
    # print(d.variants)
    #
    # # print(d.scores())
    # # print(d.memoized_scores)
    # print('d.ranked()')
    # print(d.ranked())
    # # print('d.memoized_scores')
    # # print(d.memoized_scores)
    # # input('sanity check')
    # print(d.ranked()[0])
    # print(max(d.memoized_scores))
    # for el in d.scored():
    #     print(el)
    # # print(d.scored())
    # print('getting best')
    # print(d.best())
    #
    # print('###### NO VARIANTS CHECK ######')
    # d1 = Decision(variants=variants[:10])
    # print('\n####')
    # for variant in variants[:10]:
    #     print(variant)
    #
    # # input('sanity check')
    #
    # print('\n#### SCORES CALL')
    # print(d1.scores())
    # print(np.median(d1.scores()))
    #
    # print('\n#### RANKED CALL')
    # for ranked_v in d1.ranked():
    #     print(ranked_v)
    #
    # print('\n#### SCORED CALL 1')
    # for scored_v, score_val in zip(d1.scored(), d1.memoized_scores):
    #     print('FIRST CALL {} -> {}'.format(scored_v, score_val))
    #
    # print('\n#### SCORED CALL 2')
    # for scored_v, score_val in zip(d1.scored(), d1.memoized_scores):
    #     print('SECOND CALL {} -> {}'.format(scored_v, score_val))
    #
    # print('\n#### BEST CALL')
    # print(d1.best())
    # # print(d1.track_runners_up)
    #
    # tc = 0
    # for _ in range(100):
    #     _d = Decision(variants=variants[:10])
    #     if _d.track_runners_up is True:
    #         tc += 1
    #         print('tracking')
    # print(tc)
