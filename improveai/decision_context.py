from copy import deepcopy

import numpy as np

import improveai.decision as d
import improveai.decision_model as dm
from improveai.utils.general_purpose_tools import check_variants, get_variants_from_args


class DecisionContext:

    @property
    def decision_model(self):
        """
        DecisionModel of this DecisionContext

        Returns
        -------
        DecisionModel
            DecisionModel of this DecisionContext

        """
        return self._decision_model

    @decision_model.setter
    def decision_model(self, value):
        # DecisionContext's model must not be None and of desired type
        assert value is not None
        assert isinstance(value, dm.DecisionModel)
        self._decision_model = value

    @property
    def givens(self) -> dict:
        """
        Givens for this DecisionContext

        Returns
        -------
        dict
            Givens for this DecisionContext

        """
        return self._givens

    @givens.setter
    def givens(self, value):
        # givens can be None or dict
        assert isinstance(value, dict) or value is None
        self._givens = value

    def __init__(self, decision_model, givens: dict or None):
        """
        Init with params

        Parameters
        ----------
        decision_model: DecisionModel
            decision model for this DecisionContext
        givens: dict
            givens for this DecisionContext
        """
        self.decision_model = decision_model
        self.givens = givens

    def score(self, variants) -> np.ndarray:
        """
        Scores provided variants with available givens

        Parameters
        ----------
        variants: list or tuple or np.ndarray
            array of variants to choose from

        Returns
        -------
        np.ndarray
            array of float64 scores

        """

        # get givens from provider
        givens = self.decision_model.givens_provider.givens(for_model=self.decision_model, givens=self.givens)
        return self.decision_model._score(variants=variants, givens=givens)

    def which(self, *variants) -> tuple:
        """
        A short hand version of chooseFrom that returns the chosen result directly,
        automatically tracking the decision in the process. This would be a varargs
        version of chooseFrom that would also take an array as the only argument.
        If the only argument was not an array it would throw an exception.
        Since the Decision object is not returned, rewards would only be tracked
        through DecisionModel.addReward

        Parameters
        ----------
        *variants: list or tuple or np.ndarray
            array of variants passed as positional parameters

        Returns
        -------
        object, str
            a tuple of (<best variant>, <decision id>)

        """

        return self.which_from(variants=get_variants_from_args(variants))

    def which_from(self, variants: list or np.ndarray) -> tuple:
        """
        Makes a decision for provided variants and DecisionContext givens.

        Parameters
        ----------
        variants: list or np.ndarray
            variants from which best will be chosen

        Returns
        -------
        object, str
            a tuple of (<best variant>, <decision id>)

        """
        # create a Decision object
        decision = self.decide(variants=variants)
        # track if possible -> tracker and track_url exist
        if self.decision_model.track_url is not None and self.decision_model.tracker is not None:
            decision.track()
        # return best variant and decision ID to allow rewarding
        return decision.best, decision.id_

    def decide(self, variants: list or np.ndarray, scores: list or np.ndarray = None,
               ordered: bool = False):
        """
        Creates decision but does not track it by default.
        If not scores are provided and input variants are not ranked performs
        scoring and ranking. If scores are provided but variants are not ordered
        ranks variants before making a decision

        Parameters
        ----------
        variants: list or np.ndarray
            variants for the decision
        scores: list or np.ndarray
            scores for variants, nullable
        ordered: bool
            flag indicating if input variants are ordered

        Returns
        -------
        d.Decision
            decision object created for input data

        """
        # TODO test with scores == None and scores != None
        # get givens via GivensProvider
        givens = self.decision_model.givens_provider.givens(for_model=self.decision_model, givens=self.givens)
        # rank variants if they are not ordered
        assert isinstance(ordered, bool)
        if scores is not None and ordered is True:
            raise ValueError('Both `scores` and `ordered` are not None. One of them must be None (please check docs).')

        if not ordered:
            # if variants are not ordered scoring or ranking must be performed
            if scores is None:
                # if scores were not provided they must be calculated
                scores = self.decision_model._score(variants=variants, givens=givens)

            # at this point scores must not be None and must be of the same length as variants
            assert scores is not None
            assert len(scores) == len(variants)
            # rank variants using scores
            ranked_variants = self.decision_model._rank(variants=variants, scores=scores)
        else:
            # variants are already ordered
            ranked_variants = deepcopy(variants)

        decision = d.Decision(
            decision_model=self.decision_model, ranked=ranked_variants, givens=givens)
        return decision

    def rank(self, variants: list or np.ndarray) -> list or np.ndarray:
        """
        Ranks provided variants from best to worst and creates decision from them

        Parameters
        ----------
        variants: list or np.ndarray
            variants to be ranked

        Returns
        -------
        list or np.ndarray
            ranked variants
        """
        return self.decide(variants=variants).ranked

    def optimize(self, variant_map: dict):
        """
        Get the best configuration for a given variants map

        Parameters
        ----------
        variant_map: dict
            mapping variants name -> variants list

        Returns
        -------
        object, str
            best variant and a decision ID

        """
        return self.which_from(
            variants=dm.DecisionModel.full_factorial_variants(variant_map=variant_map))

    def choose_first(self, variants: list or tuple or np.ndarray):
        # TODO method deprecated - will be removed in v8 upgrade
        """
        Chooses first from provided variants using gaussian scores (_generate_descending_gaussians())

        Parameters
        ----------
        variants: list or tuple or np.ndarray
            collection of variants passed as positional parameters

        Returns
        -------
        Decision
            A decision with first variants as the best one and gaussian scores
        """

        # make decision and do not track it
        return self.decide(variants=variants, ordered=True)

    def first(self, *variants) -> tuple:
        # TODO method deprecated - will be removed in v8 upgrade
        """
        Makes decision using first variant as best and tracks it.
        Accepts variants as pythonic *args

        Parameters
        ----------
        variants: list or tuple or np.ndarray
            collection of variants of which first will be chosen

        Returns
        -------
        object, str
            a tuple of (<first variant>, <decision id>)
        """
        # TODO think about get_variants_from_args() and checking variants for
        #  numpy types
        # make decision and track immediately
        decision = self.decide(variants=get_variants_from_args(variants), ordered=True)
        # return best and decision ID
        return decision.get(), decision.id_

    def choose_random(self, variants: list or tuple or np.ndarray):
        # TODO method deprecated - will be removed in v8 upgrade
        """
        Shuffles variants to return Decision with gaussian scores and random best variant

        Parameters
        ----------
        variants: list or tuple or np.ndarray
            collection of variants of which random will be chosen

        Returns
        -------
        Decision
            Decision with randomly chosen best variant
        """
        check_variants(variants)
        return self.decide(variants, scores=np.random.normal(size=len(variants)))

    def random(self, *variants) -> tuple:
        # TODO method deprecated - will be removed in v8 upgrade
        """
        Makes decision using randomly selected variant as best and tracks it.
        Accepts variants as pythonic *args

        Parameters
        ----------
        variants: list or np.ndarray
            collection of variants of which first will be chosen

        Returns
        -------
        object, str
            a tuple of (<random variant>, <decision id>)
        """
        # check variants before scoring
        check_variants(variants)
        # unpack variants -> avoid pitfall of [[variants]] in random scores generation
        unpacked_variants = get_variants_from_args(variants)
        # decide which one is best
        decision = self.decide(
            variants=unpacked_variants, scores=np.random.normal(size=len(unpacked_variants)))
        return decision.get(), decision.id_

    def choose_from(
            self, variants: list or tuple or np.ndarray, scores: np.ndarray or list or None):
        # TODO method deprecated - will be removed in v8 upgrade
        """
        Makes a Decision without tracking it

        Parameters
        ----------
        variants: list or tuple or np.ndarray
            array of variants to choose from
        scores: np.ndarray or list or None
            list of scores for variants

        Returns
        -------
        Decision
            snapshot of the Decision made with provided variants and available givens
        """
        return self.decide(variants=variants, scores=scores)

    def choose_multivariate(self, variant_map: dict) -> dict:
        """
        Chooses the best configuration from full factorial of input variants map

        Parameters
        ----------
        variant_map: dict
            map of key -> variants set

        Returns
        -------
        dict
            combination of the best variants in a dict

        """
        # check_variant_map(variant_map=variant_map)
        return self.choose_from(
            variants=dm.DecisionModel.full_factorial_variants(variant_map=variant_map),
            scores=None)

    def _track(self, variant: object, runners_up: list or np.ndarray, sample: object, sample_pool_size: int) -> str:
        """
        Tracks provided variant with runners up and sample

        Parameters
        ----------
        variant: object
            variant to be tracked
        runners_up: list
            runners_up to be tracked
        sample: object
            a sample for an input variant
        sample_pool_size: int
            number of variants from which sample was drawn

        Returns
        -------
        str
            decision ID
        """
        assert self.decision_model.track_url is not None
        assert self.decision_model.tracker is not None

        if runners_up is not None:
            check_variants(runners_up)

        if runners_up is None:
            runners_up_count = 0
        else:
            if not isinstance(runners_up, list):
                runners_up = list(runners_up)
            runners_up_count = len(runners_up)

        variants_count = 1 + runners_up_count + sample_pool_size
        body = self.decision_model.tracker._get_decision_track_body(
            variant=variant, model_name=self.decision_model.model_name, variants_count=variants_count,
            givens=self.givens, runners_up=runners_up, sample=sample,
            has_sample=True if sample_pool_size > 0 else False)

        return self.decision_model.tracker.post_improve_request(body_values=body)
