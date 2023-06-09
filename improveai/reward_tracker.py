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
from improveai.utils.general_purpose_tools import check_candidates, \
    is_valid_ksuid, deepcopy_args


class RewardTracker:
    """
    Tracks items and rewards for training updated scoring models. When an item
    becomes causal, pass it to the track() function, which will return a `reward_id`.
    Use the `reward_id` to track future rewards associated with that item.
    """

    @property
    def MODEL_KEY(self) -> str:
        """
        Key under which model name is stored in track request dict / body

        Returns
        -------
        str
            "model" -> track request body key storing model name
        """
        return "model"

    @property
    def MESSAGE_ID_KEY(self) -> str:
        """
        Key under which message ID is stored in track request dict / body

        Returns
        -------
        str
            "message_id" -> track request body key storing message ID
        """
        return "message_id"

    @property
    def ITEM_KEY(self) -> str:
        """
        Key under which item is stored in track request dict / body

        Returns
        -------
        str
            "item" -> track request body key storing best variant
        """
        return "item"

    @property
    def CONTEXT_KEY(self) -> str:
        """
        Key under which context is stored in track request dict / body

        Returns
        -------
        str
            "context" -> track request body key storing givens
        """
        return "context"

    @property
    def REWARD_KEY(self) -> str:
        """
        Key under which reward is stored in reward request dict / body

        Returns
        -------
        str
            "reward" -> track request body key storing reward value
        """
        return 'reward'

    @property
    def REWARD_ID_KEY(self) -> str:
        """
        Key under which reward ID is stored in reward request dict / body

        Returns
        -------
        str
            "decision_id" -> track request body key storing decision ID
        """
        return "decision_id"

    @property
    def API_KEY_HEADER(self) -> str:
        """
        Key for request headers storing an API key

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
        Key under which candidates / items count is stored in reward request dict / body

        Returns
        -------
        str
            Track request headers key storing variants count
        """
        return 'count'

    @property
    def SAMPLE_KEY(self) -> str:
        """
        Key under which sample is stored in reward request dict / body

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
        Improve AI AWS track endpoint URL

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
        AWS track endpoint API key (if applicable); Can be None

        Returns
        -------
        str
            track endpoint API key
        """
        return self._api_key

    @api_key.setter
    def api_key(self, new_val: str):
        self._api_key = new_val

    @property
    def threaded_requests(self) -> bool:
        """
        Boolean flag indicating whether requests should be executed with ThreadPoolExecutor.
        If True requests are non-blocking and executed within threads. If False
        requests are blocking and executed consecutively.

        Returns
        -------
        bool
            should requests to AWS track endpoint be non-blockng
        """
        return self.__threaded_requests

    def __init__(self, model_name: str, track_url: str, track_api_key: str = None, _threaded_requests: bool = True):
        """
        Create a RewardTracker for a specific model.

        Parameters
        ----------
        model_name: str
            Name of the model, such as "songs" or "discounts", which either makes
            the decisions or which decisions are being rewarded
        track_url: str
            The track endpoint URL that all tracked data will be sent to.
        track_api_key: str
            track endpoint API key (if applicable); Can be None
        _threaded_requests: bool
            flag indicating whether requests to AWS track endpoint should be
            non-blockng / executed within sub-threads. True by default
        """

        self.model_name = model_name
        self.track_url = track_url
        self.api_key = track_api_key

        # bool flag indicating whether thread pool executor will be used for requests
        self.__threaded_requests = _threaded_requests

    def _get_track_body(self, item: object, num_candidates: int, context: object, sample: object):
        """
        Helper method to create track body. Used by RewardTracker's `track()`
        and `track_with_sample()`

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
            self.MODEL_KEY: self.model_name,
            self.ITEM_KEY: item,
            self.ITEMS_COUNT_KEY: num_candidates}

        # as long as the num_candidates is > 1, we can assume that the sample
        # is present even if it is None
        if num_candidates > 1:
            body[self.SAMPLE_KEY] = sample

        if context is not None:
            body[self.CONTEXT_KEY] = context

        # deepcopy to avoid effects of in-place modifications
        return deepcopy(body)

    # TODO check / add unittests
    def _get_sample(self, item: object, candidates: list or tuple or np.ndarray) -> object:
        """
        Randomly selects a sample from `candidates` excluding `item`.

        Parameters
        ----------
        item: object
            the best of candidates
        candidates: list or tuple or np.ndarray
            collection of candidates

        Raises
        -------
        AssertionError
            if `len(candidates) == 1` it is nto possible to select a sample

        Returns
        -------
        object
            sample from candidates
        """
        assert len(candidates) > 1, \
            'candidates must have at least 2 items in order to draw sample'

        if not isinstance(candidates, list):
            candidates = list(candidates)

        item_index = candidates.index(item)
        sample_index = item_index
        while sample_index == item_index:
            sample_index = np.random.randint(len(candidates))
        return candidates[sample_index]

    def track(self, item: object, candidates: list or tuple or np.ndarray = None,
              context: object = None) -> str or None:
        """
        Tracks the item selected from candidates and a random sample from the remaining items.
        If `len(candidates) == 1` there is no sample.

        Parameters
        ----------
        item: object
            any JSON encodable object chosen as best from candidates
        candidates: list or tuple or np.ndarray
            collection of items from which best is chosen
        context: object
            any JSON encodable extra context info that was used with each of the
            item to get its score

        Returns
        -------
        str or None
            reward_id of this track request or None if an error happened
        """
        # this will raise an assertion error if candidates are bad
        check_candidates(candidates)
        item, candidates, context = deepcopy_args(*[item, candidates, context])
        if isinstance(candidates, np.ndarray):
            candidates = candidates.tolist()
        # item must be in candidates
        assert item in candidates

        body = self._get_track_body(
            item=item, num_candidates=len(candidates), context=context,
            sample=self._get_sample(item, candidates) if len(candidates) > 1 else None)

        return self.post_improve_request(body_values=body)

    def track_with_sample(
            self, item: object, num_candidates: int = None, context: object = None,
            sample: object = None) -> str or None:
        """
        Tracks the item selected and a specific sample.. Provided sample is
        appended to track request (in contrary to `track(...)` where sample is
        randomly selected from candidates).

        Parameters
        ----------
        item: object
            any JSON encodable object chosen as best from candidates
        num_candidates: int
            total number of candidates, including the selected item
        context: object
            any JSON encodable extra context info that was used with each of the
            item to get its score
        sample: object
            a random sample from the candidates

        Returns
        -------
        str or None
            reward_id of this track request or None if an error happened
        """

        item, num_candidates, context, sample = \
            deepcopy_args(*[item, num_candidates, context, sample])

        assert num_candidates > 0
        body = self._get_track_body(
            item=item, num_candidates=num_candidates, context=context, sample=sample)
        return self.post_improve_request(body_values=body)

    def add_reward(self, reward: float or int, reward_id: str):
        """
        Add reward for the provided reward_id

        Parameters
        ----------
        reward: float or int
            the reward to add; must be numeric (float, int ro bool), must not be
             `None`, `np.nan` or +-`inf`
        reward_id: str
            the id that was returned from the track(...) / track_with_sample(...) methods

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
            self.MODEL_KEY: self.model_name,
            self.REWARD_KEY: reward,
            self.REWARD_ID_KEY: reward_id}

        self.post_improve_request(body_values=body)

    def _is_valid_message_id(self, message_id: str or None) -> bool:
        """
        Check if message_id is a valid Ksuid


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
        Execute Improve POST to `track_url` with provided payload and headers

        Parameters
        ----------
        payload_json: str
            improveai body dumped to JSON
        headers: dict
            request headers
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
        Posts request to AWS tracker endpoint. In case of an exception during
        POST process does not break the main thread, just prints a warning with
        an error message.

        Parameters
        ----------
        body_values: dict
            dict containing request body. Bodies differ for `track()` and `add_reward()`
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

        if self.threaded_requests:
            improveai.track_improve_executor.submit(self.do_post_improve_request, *[payload_json, headers])
        else:
            self.do_post_improve_request(payload_json, headers)

        return body[self.MESSAGE_ID_KEY]
