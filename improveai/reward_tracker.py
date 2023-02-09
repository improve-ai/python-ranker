from copy import deepcopy
import numpy as np
import orjson
import re
import requests as rq
from typing import Dict
from warnings import warn

from ksuid import Ksuid

import improveai
from improveai.chooser import XGBChooser
from improveai.utils.general_purpose_tools import check_items, is_valid_ksuid


class DecisionTracker:

    @property
    def MODEL_KEY(self) -> str:
        """
        Track request body key storing model name

        Returns
        -------
        str
            Track request body key storing model name

        """
        return "model"

    @property
    def MESSAGE_ID_KEY(self) -> str:
        """
        Track request body key storing message ID

        Returns
        -------
        str
            Track request body key storing message ID

        """
        return "message_id"

    @property
    def TYPE_KEY(self) -> str:
        """
        Track request body key storing request type (e.g. decision is a request type as well as reward)

        Returns
        -------
        str
            Track request body key storing request type

        """
        return "type"

    @property
    def ITEM_KEY(self) -> str:
        """
        Track request body key storing best variant

        Returns
        -------
        str
            Track request body key storing best variant

        """
        return "variant"

    @property
    def CONTEXT_KEY(self) -> str:
        """
        Track request body key storing givens

        Returns
        -------
        str
            Track request body key storing givens

        """
        return "givens"

    @property
    def REWARD_TYPE(self) -> str:
        """
        If request is a reward this should be provided as `<request body>[<TYPE_KEY>]`

        Returns
        -------
        str
            type to be appended to request body in case of reward request

        """
        return 'reward'

    @property
    def REWARD_KEY(self) -> str:
        """
        Track request body key storing reward value

        Returns
        -------
        str
            Track request body key storing reward value

        """
        return 'reward'

    @property
    def DECISION_TYPE(self) -> str:
        """
        If a request is a decision this should be provided as `<request body>[<TYPE_KEY>]`

        Returns
        -------
        str
            type to be appended to request body in case of decision request

        """

        return "decision"

    @property
    def REWARD_ID_KEY(self) -> str:
        """
        Track request body key storing decision ID

        Returns
        -------
        str
            Track request body key storing decision ID

        """
        return "decision_id"

    @property
    def API_KEY_HEADER(self) -> str:
        """
        Track request headers key storing `API key`

        Returns
        -------
        str
            Track request headers key storing `API key`

        """

        return "x-api-key"

    @property
    def PAYLOAD_FOR_ERROR_KEY(self) -> str:
        """
        user info dict key storing track request body which caused an error

        Returns
        -------
        str
            user info dict key storing track request body which caused an error

        """
        return 'ERROR_WITH_PAYLOAD'

    @property
    def REQUEST_ERROR_CODE_KEY(self) -> str:
        """
        user info dict key storing track request error code

        Returns
        -------
        str
            user info dict key storing track request error code

        """
        return 'REQUEST_ERROR_CODE'

    @property
    def ITEMS_COUNT_KEY(self) -> str:
        """
        Track request headers key storing variants count

        Returns
        -------
        str
            Track request headers key storing variants count

        """
        return 'count'

    @property
    def SAMPLE_KEY(self) -> str:
        """
        Track request headers key storing sample

        Returns
        -------
        str
            Track request headers key storing sample

        """
        return 'sample'

    @property
    def model_name(self) -> str:
        """
        Name of tracked model. Model name must pass "^[a-zA-Z0-9][\w\-.]{0,63}$"
        regex

        Returns
        -------
        str
            Name of the tracked model

        """
        return self._model_name

    @model_name.setter
    def model_name(self, value: str):
        # make sure model_name is not None
        assert value is not None
        # make sure model name is valid
        assert re.search(XGBChooser.MODEL_NAME_REGEXP, value) is not None, \
            f'`model_name` must comply with {XGBChooser.MODEL_NAME_REGEXP} regexp'
        self._model_name = value

    @property
    def track_url(self) -> str:
        """
        Improve AI track endpoint URL

        Returns
        -------
        str
            Improve AI track endpoint URL

        """
        return self._track_url

    @track_url.setter
    def track_url(self, new_val: str):
        assert new_val is not None
        self._track_url = new_val

    @property
    def api_key(self) -> str:
        """
        track endpoint API key (if applicable); Can be None

        Returns
        -------
        str
            track endpoint API key

        """
        return self._api_key

    @api_key.setter
    def api_key(self, new_val: str):
        self._api_key = new_val

    def __init__(self, model_name: str, track_url: str, track_api_key: str = None):
        """
        Init with params

        Parameters
        ----------
        track_url: str
            Improve AI track endpoint URL
        track_api_key: str
            Improve AI track endpoint API key (nullable)
        """

        self.model_name = model_name
        self.track_url = track_url
        self.api_key = track_api_key

    def _get_track_body(
            self, item: object, num_candidates: int, context: object, sample: object):
        """
        Helper method to create track body. used by RewardTracker's track()

        Parameters
        ----------
        item: object
            tracked variant
        num_candidates: int
            number of variants
        context: dict
            givens for this decision
        sample: object
            sample to be tracked

        Returns
        -------
        dict
            body for _post_improve_request()

        """

        body = {
            self.TYPE_KEY: self.DECISION_TYPE,
            self.MODEL_KEY: self.model_name,
            self.ITEM_KEY: item,
            self.ITEMS_COUNT_KEY: num_candidates}

        if sample is not None:
            body[self.SAMPLE_KEY] = sample

        if context is not None:
            body[self.CONTEXT_KEY] = context

        return body

    def _check_candidates(
            self, candidates: list or tuple or np.ndarray, num_candidates: int, item: object):
        """
        Check if provided candidates and num_candidates have valid values and
        item is in candidates

        Parameters
        ----------
        candidates: list or tuple or np.ndarray
            collection of candidates
        num_candidates: int
            number of candidates
        item: object
            item chosen from candidates

        Returns
        -------
        None
            None

        """
        if candidates is not None:
            # num candidates must be None
            assert num_candidates is None
            # candidates must be at least of length 2 (item + 1 candidate)
            assert len(candidates) > 1
            # item must be in candidates
            assert item in candidates
        elif candidates is None and num_candidates is not None:
            # num candidates must be at least 2 (item + 1 candidate)
            assert num_candidates > 1
        else:
            # this is the case when both candidates and num_candidates are None
            raise ValueError('Either `candidates` or `num_candidates` must be provided')

    def _get_items_count(self, candidates: list or tuple or np.ndarray, num_candidates: int):
        """
        Extracs items count based on candidates and num_candidates values

        Parameters
        ----------
        candidates: list
            list of candidates
        num_candidates: int
            number of candidates

        Returns
        -------
        int
            number of candidates to be tracked

        """

        if candidates is not None:
            return len(candidates)
        elif num_candidates is not None:
            return num_candidates
        else:
            raise ValueError('Both `candidates` and `num_candidates` must not be provided at once')

    def _check_sample(self, sample: object, item: object, candidates: list or tuple or np.ndarray):
        """
        Checks if sample is None.
        if sample is not None, checks if sample is different from item and is in
        candidates.

        Parameters
        ----------
        sample: object
            JSON encodable object included in candidates or None
        item: object
            the best of candidates
        candidates: list or tuple or np.ndarray
            collection of candidates

        Returns
        -------
        None
            None

        """
        if sample is not None:
            # make sure sample is different from item
            # TODO determine if duplicates are allowed -> if so id()s should be compared (?)
            assert id(item) != id(sample)
            # make sure sample is in candidates
            assert sample in candidates

    def track(self, item: object, candidates: list or tuple or np.ndarray = None,
              num_candidates: int = None, context: object = None, sample: object = None) -> str or None:
        """
        Track that variant is causal in the system

        Parameters
        ----------
        item: object
            any JSON encodable object chosen as best from candidates
        candidates: list or tuple or np.ndarray
            collection of items from which best is chosen
        num_candidates: int
            number of items from which best is chosen; if provided, candidates must be None
        context: object
            any JSON encodable object representing context
        sample: object
            any JSON encodable object randomly selected from candidates
            (different from `item`)

        Returns
        -------
        str or None
            message id of sent improve request or None if an error happened

        """
        assert self.track_url is not None

        # this will raise an assertion error if candidates are bad
        self._check_candidates(candidates=candidates, num_candidates=num_candidates, item=item)
        # this will raise an assertion error if sample is bad
        self._check_sample(sample=sample, item=item, candidates=candidates)

        body = self._get_track_body(
            item=item, num_candidates=self._get_items_count(candidates, num_candidates),
            context=context, sample=sample)

        return self.post_improve_request(body_values=body)

    def add_reward(self, reward: float or int, reward_id: str):
        """
        Adds provided reward for a given decision id made by a given model.


        Parameters
        ----------
        reward: float or int
            reward to be assigned to a given decision
        reward_id: str
            ksuid of the reward

        Returns
        -------
        str
            message ID

        """

        assert is_valid_ksuid(reward_id)
        assert isinstance(reward, float) or isinstance(reward, int)
        assert reward is not None
        assert not np.isnan(reward)
        assert not np.isinf(reward)

        body = {
            self.TYPE_KEY: self.REWARD_TYPE,
            self.MODEL_KEY: self.model_name,
            self.REWARD_KEY: reward,
            self.REWARD_ID_KEY: reward_id}

        self.post_improve_request(body_values=body)

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

    def do_post_improve_request(self, payload_json: str, headers: dict):
        """
        Execute improve POST request with provided payload and headers

        Parameters
        ----------
        payload_json: str
            improveai body dumped to JSON
        headers: dict
            request headers

        Returns
        -------
        None
            None

        """
        try:
            response = rq.post(url=self.track_url, data=payload_json, headers=headers)
            if response.status_code >= 400:
                user_info = dict(deepcopy(response.headers))
                user_info[self.REQUEST_ERROR_CODE_KEY] = str(response.status_code)

                if payload_json:
                    user_info[self.PAYLOAD_FOR_ERROR_KEY] = payload_json \
                        if len(payload_json) <= 1000 else payload_json[:100]

                # TODO test this path within thread but how?
                warn('When attempting to post to improve.ai endpoint got an error with code {} and user info: {}'
                     .format(str(response.status_code), orjson.dumps(user_info).decode('utf-8')))

        except Exception as exc:
            print('Following error occurred:')
            print(exc)

    def post_improve_request(self, body_values: Dict[str, object], message_id: str = None) -> str or None:
        """
        Posts request to tracker endpoint
        Parameters
        ----------
        body_values: dict
            dict to be posted
        message_id: str or None
            ksuid of a given request
        Returns
        -------
        str
            message id of sent improve request
        """

        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers[self.API_KEY_HEADER] = self.api_key

        assert self._is_valid_message_id(message_id=message_id)
        body = {self.MESSAGE_ID_KEY: message_id if message_id is not None else str(Ksuid())}
        body.update(body_values)

        # serialization is a must-have for this requests
        try:
            payload_json = orjson.dumps(body).decode('utf-8')
        except Exception as exc:
            warn("Data serialization error: {}\nbody: {}".format(exc, body))
            return None

        improveai.track_improve_executor.submit(self.do_post_improve_request, *[payload_json, headers])

        return body[self.MESSAGE_ID_KEY]
