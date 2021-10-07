from collections.abc import Iterable
from copy import deepcopy
from datetime import datetime
import json
import numpy as np
import requests as rq
from typing import Dict
from uuid import uuid4
from warnings import warn

from improveai.utils.general_purpose_tools import constant


class DecisionTracker:

    @constant
    def MODEL_KEY() -> str:
        return "model"

    @constant
    def HISTORY_ID_KEY() -> str:
        return "history_id"

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
    def REWARDS_KEY() -> str:
        return "rewards"

    @constant
    def VARIANTS_COUNT_KEY() -> str:
        return "count"

    @constant
    def VARIANTS_KEY() -> str:
        return "variants"

    @constant
    def SAMPLE_VARIANT_KEY() -> str:
        return "sample_variant"

    @constant
    def REWARD_KEY_KEY() -> str:
        return "reward_key"

    @constant
    def EVENT_KEY() -> str:
        return "event"

    @constant
    def PROPERTIES_KEY() -> str:
        return "properties"

    @constant
    def DECISION_TYPE() -> str:
        return "decision"
# TODO verify if this is obsolete
    # @constant
    # def REWARDS_TYPE() -> str:
    #     return "rewards"

    @constant
    def EVENT_TYPE() -> str:
        return "event"

    @constant
    def API_KEY_HEADER() -> str:
        return "x-api-key"

    @constant
    def HISTORY_ID_DEFAULTS_KEY() -> str:
        return "ai.improve.history_id"

    @constant
    def PAYLOAD_FOR_ERROR_KEY():
        return 'ERROR_WITH_PAYLOAD'

    @constant
    def ERROR_CODE_KEY():
        return 'PY_ERROR_MESSAGE'

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

    @property
    def history_id(self) -> str:
        return self._history_id

    @history_id.setter
    def history_id(self, new_val: str):
        self._history_id = new_val

    def __init__(
            self, track_url: str, api_key: str = None,
            max_runners_up: int = 50, history_id: str = None):

        self.track_url = track_url
        self.api_key = api_key

        # TODO determined whether it should be set once per tracker`s life or
        #  with each track request (?)
        self.max_runners_up = max_runners_up
        self.history_id = history_id

        # self.track_runners_up = None

    def should_track_runners_up(self, variants_count: int):

        if variants_count == 1 or self.max_runners_up == 0:
            return False
        elif variants_count == 2:
            return True
        else:
            return np.random.rand() < 1 / min(
                variants_count - 1, self.max_runners_up)

    def top_runners_up(self, ranked_variants: list) -> Iterable or None:

        # TODO ask if max_runners_up should indicate index or a count -
        #  I would assume count

        # TODO should this method return [] instead of None ?

        # len(ranked_variants) - 1 -> this will not include last element of
        # collection

        top_runners_up = ranked_variants[
             1:min(len(ranked_variants), self.max_runners_up + 1)]\
            if ranked_variants is not None else None
        # If `top_runners_up` == [] return None
        if not top_runners_up:
            return None

        returned_top_runners_up = top_runners_up
        if isinstance(top_runners_up, np.ndarray):
            returned_top_runners_up = top_runners_up.tolist()

        return returned_top_runners_up

    def _is_sample_available(self, variants: list or None, runners_up: list):

        variants_count = len(variants)
        runners_up_count = len(runners_up) if runners_up else 0

        if (variants_count - runners_up_count - 1) > 0 and variants_count > 2:
            return True

        return False

    def track(
            self, variant: object, variants: list or np.ndarray, givens: dict,
            model_name: str, variants_ranked_and_track_runners_up: bool,
            timestamp: object = None):
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

        """

        if not self.track_url:
            return None

        if isinstance(variants, np.ndarray):
            variants = variants.tolist()

        body = {
            self.TYPE_KEY: self.DECISION_TYPE,
            self.MODEL_KEY: model_name,
            self.VARIANT_KEY: variant,
            self.VARIANTS_COUNT_KEY:
                len(variants) if variants is not None else 1}

        if len(variants) == 2:
            if variant != variants[0]:
                variants = list(reversed(variants))
            variants_ranked_and_track_runners_up = True

        # TODO unittest / validate that when
        #  variants_ranked_and_track_runners_up == False runners_up are always
        #  empty
        runners_up = None
        if variants is not None and variants != [None] \
                and variants_ranked_and_track_runners_up:

            # TODO verify if this should stay
            assert variant == variants[0]

            runners_up = self.top_runners_up(ranked_variants=variants)

            if runners_up is not None:
                body[self.RUNNERS_UP_KEY] = \
                    self.top_runners_up(ranked_variants=variants)

        if self._is_sample_available(variants=variants, runners_up=runners_up):

            sample = \
                self.get_sample(
                    variant=variant, variants=variants,
                    track_runners_up=variants_ranked_and_track_runners_up)
            body[self.SAMPLE_KEY] = sample

        if givens is not None:
            body[self.GIVENS_KEY] = givens

        # TODO determine what sort of completion_block should be used
        #  For now completion_block() set to 0
        return self.post_improve_request(
            body_values=body,
            block=lambda result, error: (
                warn("Improve.track error: {}".format(error))
                if error else 0, 0), timestamp=timestamp)

    def track_event(
            self, event_name: str, properties: Dict[str, object] = None,
            timestamp: object = None):
        """
        Executes post_improve_request constructing body from input params:
        event, properties and givens

        Parameters
        ----------
        event_name: str
            name of event
        properties: Dict[str, object]
           part fo payload (?)
        timestamp: object
            timestamp of tracking request

        Returns
        -------
        None
            None

        """

        body = {self.TYPE_KEY: self.EVENT_TYPE}

        if properties:
            if not isinstance(properties, dict):
                raise TypeError('`properties must be of a dict type`')

        for key, val in zip(
                [self.EVENT_KEY, self.PROPERTIES_KEY], [event_name, properties]):
            if val:
                body[key] = val

        # TODO determine what sort of completion_block should be used
        #  For now completion_block() set to 0
        return self.post_improve_request(
            body_values=body,
            block=lambda result, error: (
                warn("Improve.track error: {}".format(error))
                if error else 0, 0), timestamp=timestamp)

    # @staticmethod
    # def _get_non_numpy_type_sample(sample: object):
    #     """
    #     Gets numeric object of non-numpy type (e.g. gets int object from uint8)
    #
    #     Parameters
    #     ----------
    #     sample: object
    #         object from which basic python type will be extracted
    #
    #     Returns
    #     -------
    #     object
    #         basic python typed object
    #
    #     """
    #     if hasattr(sample, 'item'):
    #         return sample.item()
    #     return sample

    def get_sample(
            self, variant: object, variants: list, track_runners_up: bool):
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

        if not isinstance(variants, list):
            raise TypeError(
                'Provided variants are of a wrong type: {}. Only list type is '
                'allowed'.format(type(variants)))

        if len(variants) <= 2:
            raise ValueError(
                'Can`t sample from 2 or less ({}) variants'.format(len(variants)))

        # TODO If there are no runners up, then sample is a random sample
        #  from variants with just best excluded.
        if not track_runners_up:
            variant_idx = variants.index(variant)
            while True:
                sample_idx = np.random.randint(0, len(variants))
                if variant_idx != sample_idx:
                    break

            return variants[sample_idx]

        assert variant == variants[0]
        # TODO If there are no remaining variants after best and runners up,
        #  then there is no sample.
        last_runner_up_idx = min(len(variants), self.max_runners_up + 1)
        assert last_runner_up_idx < len(variants)

        # TODO If there are runners up, then sample is a random sample from
        #  variants with best and runners up excluded.
        sample = \
            variants[np.random.randint(last_runner_up_idx, len(variants))]
        return sample

    def post_improve_request(
            self, body_values: Dict[str, object], block: callable,
            timestamp: object = None):
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

        """

        if not self.history_id:
            warn("historyId cannot be nil")
            # TODO make sure what this clause in iOS-SDK does
            return None

        headers = {'Content-Type': 'application/json'}

        if self.api_key:
            headers[self.API_KEY_HEADER] = self.api_key

        body = {
            # was checked -> this convention seems to be accurate
            self.TIMESTAMP_KEY:
                timestamp if timestamp else str(np.datetime_as_string(
                    np.datetime64(datetime.now()), unit='ms', timezone='UTC')),
            self.HISTORY_ID_KEY: self.history_id,
            # TODO check if this is the desired uuid
            self.MESSAGE_ID_KEY: str(uuid4())}

        body.update(body_values)

        # serialization is a must have for this requests
        try:
            payload_json = json.dumps(body)
            # print(payload_json)

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
                content = json.dumps(body)

                if content:
                    user_info[self.PAYLOAD_FOR_ERROR_KEY] = content \
                        if len(content) <= 1000 else content[:100]

                error = str(error) if not isinstance(error, str) else error
                error += \
                    ' | when attempting to post to ai.improve got error with ' \
                    'code {} and user info: {}' \
                    .format(str(resp.status_code), json.dumps(user_info))

        json_object = None
        if not error:
            # TODO body might contain objects which are not JSON-encodable
            json_object = json.dumps(body)

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

        return resp


if __name__ == '__main__':
    track_url = 'https://gxlnlafa6e.execute-api.us-east-2.amazonaws.com/track'

    dt = DecisionTracker(track_url=track_url, history_id='dummy-history-id-1')

    # resp = decision_tracker.track(
    #     variant=variant,
    #     variants=np.array(variants),
    #     givens=givens, model_name=self.dummy_model_name,
    #     variants_ranked_and_track_runners_up=True,
    #     message_id=self.dummy_message_id,
    #     history_id=self.dummy_history_id,
    #     timestamp=self.dummy_timestamp)

    variants = [el for el in range(100)]
    # variants[0] = ''.join(['x' for _ in range(int(10110000/10))])

    # with open('dummy.json', 'w') as dj:
    #     q = json.dumps(variants[0])
    #     dj.write(q)

    # np.random.shuffle(variants)

    import base64


    def base64len(s):
        encoded_str = base64.b64decode(s)
        return len(encoded_str)
    # large_variant = ''.join(['x' for _ in range(1011100)])

    import time

    for _ in range(80):

        givens = {}

        if np.random.rand() > 0.8:
            givens = {
                'g1': 0,
                'g2': 1}

        resp = dt.track(
            variant=variants[0],
            variants=variants[:1],
            givens=givens, model_name='appconfig',
            variants_ranked_and_track_runners_up=False,
            timestamp=str(np.datetime_as_string(
                        np.datetime64(datetime.now()), unit='ms', timezone='UTC')))

        print(resp.status_code)
        time.sleep(0.1)
