import re
import warnings
from copy import deepcopy
import numpy as np
import orjson
import requests as rq
from typing import Dict
from warnings import warn

from ksuid import Ksuid

from improveai.chooser import XGBChooser
from improveai.utils.general_purpose_tools import constant, check_variants


class DecisionTracker:

    @constant
    def MODEL_KEY() -> str:
        """
        Track request body key storing model name

        Returns
        -------
        str
            Track request body key storing model name

        """
        return "model"

    @constant
    def MESSAGE_ID_KEY() -> str:
        """
        Track request body key storing message ID

        Returns
        -------
        str
            Track request body key storing message ID

        """
        return "message_id"

    @constant
    def TYPE_KEY() -> str:
        """
        Track request body key storing request type (e.g. decision is a request type as well as reward)

        Returns
        -------
        str
            Track request body key storing request type

        """
        return "type"

    @constant
    def VARIANT_KEY() -> str:
        """
        Track request body key storing best variant

        Returns
        -------
        str
            Track request body key storing best variant

        """
        return "variant"

    @constant
    def GIVENS_KEY() -> str:
        """
        Track request body key storing givens

        Returns
        -------
        str
            Track request body key storing givens

        """
        return "givens"

    @constant
    def VARIANTS_COUNT_KEY() -> str:
        """
        Track request body key storing variants count (from how many variants best was chosen)

        Returns
        -------
        str
            Track request body key storing variants count

        """
        return "count"

    @constant
    def REWARD_TYPE() -> str:
        """
        If request is a reward this should be provided as `<request body>[<TYPE_KEY>]`

        Returns
        -------
        str
            type to be appended to request body in case of reward request

        """
        return 'reward'

    @constant
    def REWARD_KEY():
        """
        Track request body key storing reward value

        Returns
        -------
        str
            Track request body key storing reward value

        """
        return 'reward'

    @constant
    def DECISION_TYPE() -> str:
        """
        If a request is a decision this should be provided as `<request body>[<TYPE_KEY>]`

        Returns
        -------
        str
            type to be appended to request body in case of decision request

        """

        return "decision"

    @constant
    def DECISION_ID_KEY() -> str:
        """
        Track request body key storing decision ID

        Returns
        -------
        str
            Track request body key storing decision ID

        """
        return "decision_id"

    @constant
    def API_KEY_HEADER() -> str:
        """
        Track request headers key storing `API key`

        Returns
        -------
        str
            Track request headers key storing `API key`

        """

        return "x-api-key"

    @constant
    def PAYLOAD_FOR_ERROR_KEY() -> str:
        """
        user info dict key storing track request body which caused an error

        Returns
        -------
        str
            user info dict key storing track request body which caused an error

        """
        return 'ERROR_WITH_PAYLOAD'

    @constant
    def REQUEST_ERROR_CODE_KEY() -> str:
        """
        user info dict key storing track request error code

        Returns
        -------
        str
            user info dict key storing track request error code

        """
        return 'REQUEST_ERROR_CODE'

    @constant
    def RUNNERS_UP_KEY() -> str:
        """
        Track request headers key storing runners up

        Returns
        -------
        str
            Track request headers key storing runners up

        """
        return 'runners_up'

    @constant
    def VARIANTS_COUNT_KEY() -> str:
        """
        Track request headers key storing variants count

        Returns
        -------
        str
            Track request headers key storing variants count

        """
        return 'count'

    @constant
    def SAMPLE_KEY() -> str:
        """
        Track request headers key storing sample

        Returns
        -------
        str
            Track request headers key storing sample

        """
        return 'sample'

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

    @property
    def max_runners_up(self):
        """
        maximum number of runners up to be included in the Improve AI request

        Returns
        -------
        int
            maximum number of runners up to be included in the Improve AI request

        """
        return self._max_runners_up

    @max_runners_up.setter
    def max_runners_up(self, new_val):
        self._max_runners_up = new_val

    def __init__(self, track_url: str, track_api_key: str = None):
        """
        Init with params

        Parameters
        ----------
        track_url: str
            Improve AI track endpoint URL
        track_api_key: str
            Improve AI track endpoint API key (nullable)
        """

        self.track_url = track_url
        self.api_key = track_api_key
        # defaults to 50
        self.max_runners_up = 50

    def _should_track_runners_up(self, variants_count: int) -> bool:
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
            return np.random.rand() < 1 / min(variants_count - 1, self.max_runners_up)

    def _top_runners_up(
            self, ranked_variants: list or tuple or np.ndarray) -> list or tuple or np.ndarray or None:
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
        top_runners_up = ranked_variants[1:min(len(ranked_variants), self.max_runners_up + 1)]\
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

    def _is_sample_available(self, ranked_variants: list or None, runners_up: list or None) -> bool:
        """
        Returns True / False flag indicating whether sample is available


        Parameters
        ----------
        ranked_variants: list or None
            collection of evaluated variants
        runners_up: list or None
            tracked runners up


        Returns
        -------
        bool
            True if sample is available False otherwise

        """

        variants_count = len(ranked_variants)
        runners_up_count = len(runners_up) if runners_up else 0

        if variants_count - runners_up_count - 1 > 0:
            return True

        return False

    def track(self, ranked_variants: list or np.ndarray, givens: dict, model_name: str) -> str or None:
        """
        Track that variant is causal in the system


        Parameters
        ----------
        ranked_variants: list or np.ndarray
            collection of tracked variants
        givens: dict:
            givens for variants
        model_name: str
            name of model which made the decision (?) / to which observations
            will be assigned

        Returns
        -------
        str or None
            message id of sent improve request or None if an error happened

        """

        if not self.track_url:
            return None

        if model_name is None:
            warnings.warn('`model_name` must not be None in order to be tracked')
            return None

        if not re.search(XGBChooser.MODEL_NAME_REGEXP, model_name):
            warnings.warn(f'`model_name` must comply with {XGBChooser.MODEL_NAME_REGEXP} regexp')
            return None

        if isinstance(ranked_variants, np.ndarray):
            ranked_variants = ranked_variants.tolist()

        body = {
            self.TYPE_KEY: self.DECISION_TYPE,
            self.MODEL_KEY: model_name,
            self.VARIANT_KEY: ranked_variants[0],
            self.VARIANTS_COUNT_KEY: len(ranked_variants) if ranked_variants is not None else 1
        }

        track_runners_up = self._should_track_runners_up(variants_count=body[self.VARIANTS_COUNT_KEY])

        if len(ranked_variants) == 2:
            # for 2 variants runners_up should be True
            assert track_runners_up

        # for track_runners_up == false (max_runners_up == 0)
        # we skip this clause and runners_up == None
        runners_up = None
        if track_runners_up:
            runners_up = self._top_runners_up(ranked_variants=ranked_variants)
            if runners_up is not None:
                body[self.RUNNERS_UP_KEY] = runners_up

        # If runners_up == None and len(variants) == 2 -> sample should be extracted
        if self._is_sample_available(ranked_variants=ranked_variants, runners_up=runners_up):
            body[self.SAMPLE_KEY] = self.get_sample(ranked_variants=ranked_variants, track_runners_up=track_runners_up)

        if givens is not None:
            body[self.GIVENS_KEY] = givens

        return self.post_improve_request(
            body_values=body,
            block=lambda result, error: (warn("Improve.track error: {}".format(error)) if error else 0, 0))

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
        str
            message ID

        """

        assert self.track_url is not None, '`track_url` is None - please provide valid `track_url`'
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

    def get_sample(self, ranked_variants: list, track_runners_up: bool) -> object:
        """
        Gets sample from ranked_variants. Takes runenrs up into account


        Parameters
        ----------
        ranked_variants: list or np.ndarray
            list of ranked variants
        track_runners_up: bool
            should runners up be tracked ?


        Returns
        -------
        object
            sample

        """

        assert isinstance(track_runners_up, bool)
        # TODO maybe this check is pointless
        # check_variants(ranked_variants)

        assert len(ranked_variants) > 1
        if len(ranked_variants) == 2:
            assert self.max_runners_up == 0

        # If there are no runners up, then sample is a random sample
        # from variants with just best excluded.

        # randomly select variants from all minus best
        if not track_runners_up:
            return ranked_variants[np.random.randint(1, len(ranked_variants))]

        # If there are no remaining variants after best and runners up,
        # then there is no sample.
        # below is just a check
        last_runner_up_idx = min(len(ranked_variants), self.max_runners_up + 1)
        assert last_runner_up_idx < len(ranked_variants)

        # If there are runners up, then sample is a random sample from
        # variants with best and runners up excluded.
        return ranked_variants[np.random.randint(last_runner_up_idx, len(ranked_variants))]

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

    def post_improve_request(self, body_values: Dict[str, object], block: callable) -> str:
        """
        Posts request to tracker endpoint


        Parameters
        ----------
        body_values: dict
            dict to be posted
        block: callable
            callable to execute on error


        Returns
        -------
        str
            message id of sent improve request

        """

        headers = {'Content-Type': 'application/json'}

        if self.api_key:
            headers[self.API_KEY_HEADER] = self.api_key

        # body = {self.MESSAGE_ID_KEY: message_id if message_id is not None else str(Ksuid())}
        body = {self.MESSAGE_ID_KEY: str(Ksuid())}
        assert self._is_valid_message_id(message_id=body[self.MESSAGE_ID_KEY])

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
            resp = rq.post(url=self.track_url, data=payload_json, headers=headers)

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
