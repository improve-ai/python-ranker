import re
import warnings
from collections.abc import Iterable
from copy import deepcopy
from datetime import datetime
import numpy as np
import orjson
import requests as rq
from typing import Dict
from warnings import warn

from ksuid import Ksuid

from improveai.chooser import XGBChooser
from improveai.utils.general_purpose_tools import constant


class DecisionTracker:

    @constant
    def MODEL_KEY() -> str:
        return "model"

    @constant
    def TIMESTAMP_KEY() -> str:
        return "timestamp"

    @constant
    def MESSAGE_ID_KEY() -> str:
        return "message_id"

    @constant
    def TYPE_KEY() -> str:
        return "type"

    @constant
    def VARIANT_KEY() -> str:
        return "variant"

    @constant
    def GIVENS_KEY() -> str:
        return "givens"

    @constant
    def VARIANTS_COUNT_KEY() -> str:
        return "count"

    @constant
    def REWARD_TYPE() -> str:
        return 'reward'

    @constant
    def REWARD_KEY():
        return 'reward'

    @constant
    def DECISION_TYPE() -> str:
        return "decision"

    @constant
    def DECISION_ID_KEY() -> str:
        return "decision_id"

    @constant
    def API_KEY_HEADER() -> str:
        return "x-api-key"

    @constant
    def PAYLOAD_FOR_ERROR_KEY():
        return 'ERROR_WITH_PAYLOAD'

    @constant
    def REQUEST_ERROR_CODE_KEY():
        return 'REQUEST_ERROR_CODE'

    @constant
    def RUNNERS_UP_KEY() -> str:
        return 'runners_up'

    @constant
    def VARIANTS_COUNT_KEY() -> str:
        return 'count'

    @constant
    def SAMPLE_KEY() -> str:
        return 'sample'

    @property
    def track_url(self) -> str:
        return self._track_url

    @track_url.setter
    def track_url(self, new_val: str):
        self._track_url = new_val

    @property
    def api_key(self) -> str:
        return self._api_key

    @api_key.setter
    def api_key(self, new_val: str):
        self._api_key = new_val

    @property
    def max_runners_up(self):
        return self._max_runners_up

    @max_runners_up.setter
    def max_runners_up(self, new_val):
        self._max_runners_up = new_val

    def __init__(
            self, track_url: str, max_runners_up: int = 50, track_api_key: str = None):

        self.track_url = track_url
        self.api_key = track_api_key
        self.max_runners_up = max_runners_up

    def _should_track_runners_up(self, variants_count: int):
        """
        Returns bool that indicates whether runners up should be tracked


        Parameters
        ----------
        variants_count: int
            number of variants


        Returns
        -------
        bool
            flag indicating if runners up will be tracked

        """

        if variants_count == 1 or self.max_runners_up == 0:
            return False
        elif variants_count == 2:
            return True
        else:
            return np.random.rand() < 1 / min(
                variants_count - 1, self.max_runners_up)

    def _top_runners_up(self, ranked_variants: list) -> Iterable or None:
        """
        Select top N runners up from `ranked_variants`


        Parameters
        ----------
        ranked_variants: list
            variants ordered descending by scores


        Returns
        -------
        Iterable or None
            None if there are no runners up to track otherwise list of tracked runners up

        """

        # len(ranked_variants) - 1 -> this will not include last element of collection
        top_runners_up = ranked_variants[
             1:min(len(ranked_variants), self.max_runners_up + 1)]\
            if ranked_variants is not None else None

        # if there is a positive max_runners_up and more than 1 variant there must be at least 1 runner up
        if len(ranked_variants) > 1 and self.max_runners_up > 0:
            assert top_runners_up is not None and len(top_runners_up) > 0

        # if max_runners_up == 0 there are no runners up
        if self.max_runners_up == 0:
            assert not top_runners_up  # None or []

        # If `top_runners_up` == [] return None
        if top_runners_up is None or len(top_runners_up) == 0:
            return None

        returned_top_runners_up = top_runners_up
        if isinstance(top_runners_up, np.ndarray):
            returned_top_runners_up = top_runners_up.tolist()

        return returned_top_runners_up

    def _is_sample_available(self, variants: list or None, runners_up: list):
        """
        Returns True / False flag indicating whether sample is available


        Parameters
        ----------
        variants: list or None
            collection of evaluated variants
        runners_up: list
            list of tracked runners up


        Returns
        -------
        bool
            True if sample is available False otherwise

        """

        variants_count = len(variants)
        runners_up_count = len(runners_up) if runners_up else 0

        if variants_count - runners_up_count - 1 > 0:
            return True

        return False

    def track(
            self, variant: object, variants: list or np.ndarray, givens: dict,
            model_name: str, variants_ranked_and_track_runners_up: bool,
            timestamp: object = None, message_id: str = None):
        """
        Track that variant is causal in the system


        Parameters
        ----------
        variant: object
            chosen variant
        variants: list or np.ndarray
            collection of tracked variants
        givens: dict:
            givens for variants
        model_name: str
            name of model which made the decision (?) / to which observations
            will be assigned
        variants_ranked_and_track_runners_up: bool
            are the variants ranked and runners up should be tracked
        timestamp: object
            when was decision tracked


        Returns
        -------
        str
            message id of sent improve request

        """

        if not self.track_url:
            return None

        if model_name is None:
            warnings.warn('`model_name` must not be None in order to be tracked')
            return None

        if not re.search(XGBChooser.MODEL_NAME_REGEXP, model_name):
            warnings.warn(f'`model_name` must comply with {XGBChooser.MODEL_NAME_REGEXP} regexp')
            return None

        if isinstance(variants, np.ndarray):
            variants = variants.tolist()

        body = {
            self.TYPE_KEY: self.DECISION_TYPE,
            self.MODEL_KEY: model_name,
            self.VARIANT_KEY: variant,
            self.VARIANTS_COUNT_KEY:
                len(variants) if variants is not None else 1,
            }

        if len(variants) == 2:
            # for 2 variants variants_ranked_and_track_runners_up should be True
            assert variants_ranked_and_track_runners_up
            if variant != variants[0]:
                variants = list(reversed(variants))

        # for variants_ranked_and_track_runners_up == false (max_runners_up == 0)
        # we skip this clause and runners_up == None
        runners_up = None
        if variants is not None and variants != [None] \
                and variants_ranked_and_track_runners_up:
            assert variant == variants[0]

            runners_up = self._top_runners_up(ranked_variants=variants)

            if runners_up is not None:
                body[self.RUNNERS_UP_KEY] = \
                    self._top_runners_up(ranked_variants=variants)

        # If runners_up == None and len(variants) == 2 -> sample should be extracted
        if self._is_sample_available(variants=variants, runners_up=runners_up):
            sample = \
                self.get_sample(
                    variant=variant, variants=variants,
                    track_runners_up=variants_ranked_and_track_runners_up)
            body[self.SAMPLE_KEY] = sample

        if givens is not None:
            body[self.GIVENS_KEY] = givens

        return self.post_improve_request(
            body_values=body,
            block=lambda result, error: (
                warn("Improve.track error: {}".format(error))
                if error else 0, 0), timestamp=timestamp, message_id=message_id)

    def add_reward(self, reward: float or int, model_name: str, decision_id: str):
        """
        Adds provided reward for a given decision id made by a given model.


        Parameters
        ----------
        reward: float or int
            reward to be assigned to a given decision
        model_name: str
            name of a model which made rewarded decision
        decision_id: str
            ksuid of rewarded decision


        Returns
        -------
        None
            None

        """

        assert model_name is not None and decision_id is not None
        assert re.search(XGBChooser.MODEL_NAME_REGEXP, model_name) is not None
        assert isinstance(reward, float) or isinstance(reward, int)
        assert reward is not None
        assert not np.isnan(reward)
        assert not np.isinf(reward)

        body = {
            self.TYPE_KEY: self.REWARD_TYPE,
            self.MODEL_KEY: model_name,
            self.REWARD_KEY: reward,
            self.DECISION_ID_KEY: decision_id}

        self.post_improve_request(
            body_values=body,
            block=lambda result, error: (
                warn("Improve.track error: {}".format(error)) if error else 0, 0))

    def get_sample(self, variant: object, variants: list, track_runners_up: bool):
        """
        Gets sample from ranked_variants. Takes runenrs up into account


        Parameters
        ----------
        variant: object
            selected variant (first of ranked variants)
        variants: list or np.ndarray
            list of ranked variants
        track_runners_up: bool
            should runners up be tracked ?


        Returns
        -------
        object
            sample

        """

        assert isinstance(track_runners_up, bool)

        if not (isinstance(variants, list) or isinstance(variants, tuple) or isinstance(variants, np.ndarray)):
            raise TypeError(
                'Provided variants are of a wrong type: {}. Only list type is '
                'allowed'.format(type(variants)))

        assert len(variants) > 1
        if len(variants) == 2:
            assert self.max_runners_up == 0

        # If there are no runners up, then sample is a random sample
        # from variants with just best excluded.
        if not track_runners_up:
            variant_idx = variants.index(variant)
            while True:
                sample_idx = np.random.randint(0, len(variants))
                if variant_idx != sample_idx:
                    break

            return variants[sample_idx]

        assert variant == variants[0]
        # If there are no remaining variants after best and runners up,
        # then there is no sample.
        last_runner_up_idx = min(len(variants), self.max_runners_up + 1)
        assert last_runner_up_idx < len(variants)

        # If there are runners up, then sample is a random sample from
        # variants with best and runners up excluded.
        sample = \
            variants[np.random.randint(last_runner_up_idx, len(variants))]
        return sample

    def _is_valid_message_id(self, message_id: str or None) -> bool:
        """
        Check if message_id is  valid


        Parameters
        ----------
        message_id: str
            checked message_id


        Returns
        -------
        bool
            True if message_id otherwise False

        """
        try:
            assert isinstance(message_id, str) or message_id is None
            Ksuid.from_base62(message_id)
            return True
        except:
            return False

    def post_improve_request(
            self, body_values: Dict[str, object], block: callable,
            message_id: str = None, timestamp: object = None):
        """
        Posts request to tracker endpoint


        Parameters
        ----------
        body_values: dict
            dict to be posted
        block: callable
            callable to execute on error
        timestamp: object
            timestamp of request


        Returns
        -------
        str
            message id of sent improve request

        """

        headers = {'Content-Type': 'application/json'}

        if self.api_key:
            headers[self.API_KEY_HEADER] = self.api_key

        assert self._is_valid_message_id(message_id=message_id)

        body = {
            self.TIMESTAMP_KEY:
                timestamp if timestamp else str(np.datetime_as_string(
                    np.datetime64(datetime.now()), unit='ms', timezone='UTC')),
            self.MESSAGE_ID_KEY: message_id if message_id is not None else str(Ksuid())}

        body.update(body_values)

        # serialization is a must-have for this requests
        try:
            payload_json = orjson.dumps(body).decode('utf-8')

        except Exception as exc:
            warn("Data serialization error: {}\nbody: {}".format(exc, body))
            return None

        error = None
        resp = None

        try:
            resp = \
                rq.post(url=self.track_url, data=payload_json, headers=headers)

        except Exception as exc:
            error = exc

        if not block:
            return None

        if not error and isinstance(resp, rq.models.Response):

            if resp.status_code >= 400:
                user_info = dict(deepcopy(resp.headers))
                user_info[self.REQUEST_ERROR_CODE_KEY] = str(resp.status_code)
                content = orjson.dumps(body).decode('utf-8')

                if content:
                    user_info[self.PAYLOAD_FOR_ERROR_KEY] = content \
                        if len(content) <= 1000 else content[:100]

                error = str(error) if not isinstance(error, str) else error
                error += \
                    ' | when attempting to post to improve.ai endpoint got an error with ' \
                    'code {} and user info: {}' \
                    .format(str(resp.status_code), orjson.dumps(user_info).decode('utf-8'))

        json_object = None
        if not error:
            json_object = orjson.dumps(body).decode('utf-8')

        if error:
            print('error')
            print(error)
            block(None, error)
        elif json_object:
            block(json_object, None)
        else:
            raise NotImplementedError(
                'Both error and payload objects are None / empty - this should '
                'not happen (?)')

        return body[self.MESSAGE_ID_KEY]
