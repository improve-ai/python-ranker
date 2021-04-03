from copy import deepcopy
from datetime import datetime
import json
import numpy as np
import requests as rq
from typing import Dict, List
from uuid import uuid4
from warnings import warn

from decisions.v6 import Decision
from models.decision_models import DecisionModel
from utils.general_purpose_utils import constant


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
    def CONTEXT_KEY() -> str:
        return "context"

    @constant
    def REWARDS_KEY() -> str:
        return "rewards"

    @constant
    def VARIANTS_COUNT_KEY() -> str:
        return "variants_count"

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
        return "decisions"

    @constant
    def REWARDS_TYPE() -> str:
        return "rewards"

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
    def debug(self) -> bool:
        return self._debug

    @debug.setter
    def debug(self, new_val: bool):
        self._debug = new_val

    @property
    def top_runners_up(self) -> List[Dict[str, object]]:
        return self._top_runners_up

    @top_runners_up.setter
    def top_runners_up(self, new_val: List[Dict[str, object]]):
        self._top_runners_up = new_val

    @property
    def history_id(self) -> str:
        return self._history_id

    @history_id.setter
    def history_id(self, new_val: str):
        self._history_id = new_val

    def __init__(
            self, track_url: str, api_key: str = None, history_id: str = None,
            debug: bool = False):

        self.track_url = track_url
        self.api_key = api_key
        # TODO determined whether it should be set once per tracker`s life or
        #  with each track request (?)
        self.history_id = history_id
        self.debug = debug

    # TODO determine type of timestamp (str, int or datetime)
    def track_using_best_from(
            self, decision: Decision, message_id: str, history_id: str,
            timestamp: object, completion_handler: callable = None, **kwargs):

        top_runners_up = \
            decision.top_runners_up() if decision.track_runners_up else None

        body_keys = \
            [self.TYPE_KEY, self.MODEL_KEY, self.VARIANT_KEY,
             self.RUNNERS_UP_KEY, self.VARIANTS_COUNT_KEY, self.SAMPLE_KEY,
             self.CONTEXT_KEY]

        body_vals = \
            [self.DECISION_TYPE, decision.model_name, decision.best(),
             top_runners_up, len(decision.variants),
             self._get_sample(decision=decision), decision.context]

        self.post_improve_request(
            body_values=dict(
                [(k, v) for k, v in zip(body_keys, body_vals) if v]),
            block=lambda result, error: (
                warn("Improve.track error: {}".format(error))
                if error and self.debug else 0,
                completion_handler(error) if completion_handler else 0),
            message_id=message_id, history_id=history_id, timestamp=timestamp)

        return decision.best()

    def _get_sample(self, decision: Decision, sample_size: int = 1):

        dec_variants = decision.variants
        # TODO check if best() should be called here - I don't think so
        dec_best = decision.memoized_best

        # TODO If there is only one variant, which is the best, then there is
        #  no sample.
        if len(dec_variants) == 1:
            if dec_variants[0] == dec_best:
                return None
            else:
                # TODO Is this even possible? What to do in such case
                raise NotImplementedError(
                    'This is case where decisions has a single variant ant it is'
                    ' not equal to best - best is None which means best() has '
                    'not yet been called')

        # TODO If there are no runners up, then sample is a random sample
        #  from variants with just best excluded.
        if not decision.track_runners_up:
            variants_to_be_sampled = \
                np.array([v for v in dec_variants if v != dec_best])
            sample = \
                list(np.random.choice(variants_to_be_sampled, size=sample_size))
            return sample

        # TODO If there are no remaining variants after best and runners up,
        #  then there is no sample.

        sample_excluded_variants = list(decision.top_runners_up())
        sample_excluded_variants.append(dec_best)

        variants_to_be_sampled = \
            np.array(
                [v for v in dec_variants if v not in sample_excluded_variants])

        if variants_to_be_sampled.size == 0:
            return None

        # TODO If there are runners up, then sample is a random sample from
        #  variants with best and runners up excluded.
        sample = \
            list(np.random.choice(variants_to_be_sampled, size=sample_size))
        return sample

    def track_using(
            self, variant: dict, model_name: str, context: dict = None,
            message_id: str = None, history_id: str = None,
            timestamp: object = None, completion_handler: callable = None,
            **kwargs):

        body_keys = \
            [self.TYPE_KEY, self.MODEL_KEY, self.VARIANT_KEY, self.CONTEXT_KEY]
        body_vals = [self.DECISION_TYPE, model_name, variant, context]

        # # TODO check if message ID should be added to the body
        # if message_id:
        #     body[self.MESSAGE_ID_KEY] = message_id

        self.post_improve_request(
            body_values=dict(
                [(k, v) for k, v in zip(body_keys, body_vals) if v]),
            block=lambda result, error: (
                warn("Improve.track error: {}".format(error))
                if error and self.debug else 0,
                completion_handler(error) if completion_handler else 0),
            message_id=message_id, history_id=history_id, timestamp=timestamp)

        return variant

    def track(
            self, body: Dict[str, object], completion_block: callable = None,
            message_id: str = None, history_id: str = None,
            timestamp: object = None, **kwargs):
        """
        Executes post_improve_request with provided body

        Parameters
        ----------
        body: Dict[str, object]
            payload to send
        completion_block: callable
            callable to be executed on completion
        history_id: str
            <not sure yet>
        kwargs

        Returns
        -------
        None
            None

        """

        if not self.track_url:
            return None

        self.post_improve_request(
            body_values=body,
            block=lambda result, error: (
                warn("Improve.track error: {}".format(error))
                if error and self.debug else 0,
                completion_block(error) if completion_block else 0),
            message_id=message_id, history_id=history_id, timestamp=timestamp)

    def track_event(
            self, event_name: str, properties: Dict[str, object],
            context: Dict[str, object] = None, message_id: str = None,
            history_id: str = None, timestamp: object = None, **kwargs):
        """
        Executes post_improve_request constructing body from input params:
        event, properties and context

        Parameters
        ----------
        event_name: str
            name of event
        properties: Dict[str, object]
           part fo payload (?)
        context: Dict[str, object]
            context dict for a given event
        history_id: str
            TBD
        kwargs

        Returns
        -------
        None
            None

        """

        body = {self.TYPE_KEY: self.EVENT_TYPE}

        for key, val in zip(
                [self.EVENT_KEY, self.PROPERTIES_KEY, self.CONTEXT_KEY],
                [event_name, properties, context]):
            if val:
                body[key] = val

        self.track(
            body=body, message_id=message_id, history_id=history_id,
            timestamp=timestamp)

    def add_reward(
            self, reward: float, reward_key: str, message_id: str = None,
            history_id: str = None, timestamp: object = None, **kwargs):
        """
        Adds provided reward for a input reward_key (model?)

        Parameters
        ----------
        reward: float
            rewards to be awarded to reward_key / model
        reward_key: str
            reward target
        history_id: str
            <not sure what this represents yet>

        Returns
        -------
        None
            None

        """
        single_rewards = {reward_key: reward}
        self.add_rewards(
            rewards=single_rewards, completion_handler=None,
            message_id=message_id, history_id=history_id, timestamp=timestamp)

    def add_rewards(
            self, rewards: Dict[str, float], completion_handler: callable = None,
            message_id: str = None, history_id: str = None,
            timestamp: object = None, **kwargs):
        """
        Adds multiple rewards

        Parameters
        ----------
        rewards: Dict[str, float]
            dict with reward_key -> reward mapping
        completion_handler: callable
            callable to be executed on completion (?)
        history_id: str
            <not sure yet>
        kwargs

        Returns
        -------
        None
            None

        """

        if rewards:
            if self.debug:
                print("Tracking rewards: {}".format(rewards))
            track_body = {
                self.TYPE_KEY: self.REWARDS_TYPE,
                self.REWARDS_KEY: rewards}

            self.track(
                body=track_body,
                completion_block=lambda error:
                    completion_handler(error) if completion_handler else 0,
                message_id=message_id, history_id=history_id,
                timestamp=timestamp)
        else:
            if self.debug:
                print('Skipping trackRewards for nil rewards')
            completion_handler(None) if completion_handler else None

    def track_analytics_event(
            self, event: str, properties: Dict[str, object],
            context: Dict[str, object] = None, history_id: str = None, **kwargs):
        """
        Executes post_improve_request constructing body from input params:
        event, properties and context

        Parameters
        ----------
        event: str
            JSON serialied event (?)
        properties: Dict[str, object]
           part fo payload (?)
        context: Dict[str, object]
            context dict for a given event
        history_id: str
            TBD
        kwargs

        Returns
        -------
        None
            None

        """

        body = {self.TYPE_KEY: self.EVENT_TYPE}

        for key, val in zip(
                [self.EVENT_KEY, self.PROPERTIES_KEY, self.CONTEXT_KEY],
                [event, properties, context]):
            if val:
                body[key] = val

        self.track(body=body, history_id=history_id)

    def post_improve_request(
            self, body_values: Dict[str, object], block: callable,
            timestamp: object = None, message_id: str = None,
            history_id: str = None, **kwargs):

        if not history_id:
            if self.debug:
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
            self.HISTORY_ID_KEY: history_id if history_id else self.history_id,
            # TODO check if this is the desired uuid
            self.MESSAGE_ID_KEY: message_id if message_id else str(uuid4())}

        body.update(body_values)

        # serialization is a must have for this requests
        try:
            payload_json = json.dumps(body)
        except Exception as exc:
            if self.debug:
                warn("Data serialization error: {}\nbody: {}".format(exc, body))
            return None

        error = None
        resp = None
        try:
            resp = \
                rq.post(url=self.track_url, data=payload_json, headers=headers)

            print(resp.status_code)

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
                    user_info[self.PAYLOAD_FOR_ERROR_KEY] = content

                error = str(error) if not isinstance(error, str) else error
                error += \
                    ' | when attempting to post to ai.improve got error with ' \
                    'code {} and user info: {}' \
                    .format(str(resp.status_code), json.dumps(user_info))

        json_object = None
        if not error:
            json_object = json.dumps(body)

        if error:
            block(None, error)
        elif json_object:
            block(json_object, None)
        else:
            raise NotImplementedError(
                'Both error and payload objects are None / empty - this should '
                'not happen (?)')


if __name__ == '__main__':

    model_kind = 'xgb_native'

    xgb_model_pth = '../artifacts/models/12_11_2020_verses_conv.xgb'

    dm = DecisionModel(model_kind=model_kind, model_pth=xgb_model_pth)

    with open('../artifacts/test_artifacts/sorting_context.json', 'r') as cjson:
        read_str = ''.join(cjson.readlines())
        context = json.loads(read_str)

    with open('../artifacts/data/real/meditations.json') as mjson:
        read_str = ''.join(mjson.readlines())
        variants = json.loads(read_str)

    decision = Decision(variants=variants[:500], model=dm, context=context)

    track_url = \
        'https://u0cxvugtmi.execute-api.us-west-2.amazonaws.com/test/track'
    api_key = 'xScYgcHJ3Y2hwx7oh5x02NcCTwqBonnumTeRHThI'

    dt = DecisionTracker(track_url=track_url, api_key=api_key, debug=True)

    message_id = str(uuid4())
    history_id = str(uuid4())

    dt.track_using_best_from(
        decision=decision, message_id=message_id, history_id=history_id,
        timestamp=str(datetime.now()))
