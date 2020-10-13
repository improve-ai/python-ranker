from copy import deepcopy
from datetime import datetime
import json
import numpy as np
import requests as rq
from typing import Dict, List
from uuid import uuid4
from warnings import warn

from utils.gen_purp_utils import constant


class ImproveTracker:

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
        return "decision"

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

    def __init__(
            self, tracker_url: str, api_key: str = None, debug: bool = False):
        """
        Init w params

        Parameters
        ----------
        tracker_url: str
            tracker endpoint's ulr
        api_key: str
            api key for tracker app
        """

        self.track_url = tracker_url
        self.api_key = api_key
        self.debug = debug

    def track_decision(
            self, variant: Dict[str, str], variants: List[Dict[str, str]],
            model_name: str, history_id: str, context: dict = None,
            reward_key: str = None, completion_handler: callable = None, **kwargs):
        """
        Track that a variant was chosen in order to train the system to learn
        what rewards it receives.

        Parameters
        ----------
        variant: dict
            The JSON encodeable chosen variant to track
        variants: list
            variants from which variant has been chosen
        model_name: str
            name of model which chose variant
        history_id: str
            history id
        context: dict
            The JSON encodeable context that the chosen variant is being used in
             and should be rewarded against.  It is okay for this to be
             different from the context that was used during choose or sort.
        reward_key: str
            The rewardKey used to assign rewards to the chosen variant. If nil,
            rewardKey is set to the namespace. track_rewards() must also use
            this key to assign rewards to this chosen variant.
        completion_handler: callable
            Called after sending the decision to the server.
            <not sure yet>
        kwargs

        Returns
        -------

        """

        if not self.track_url:
            return None

        if not variant and self.debug:
            warn(
                "Skipping trackDecision for nil variant. To track null values "
                "use [NSNull null]")

        usd_reward_key = reward_key
        if not usd_reward_key:
            if self.debug:
                warn("Using model name as rewardKey: {}".format(model_name))
            usd_reward_key = model_name

        body = {
            self.TYPE_KEY: self.DECISION_TYPE,
            self.VARIANT_KEY: variant,
            self.MODEL_KEY: model_name,
            self.REWARD_KEY_KEY: usd_reward_key}

        if context:
            body[self.CONTEXT_KEY] = context

        if variants and len(variants) > 0:
            body[self.VARIANTS_COUNT_KEY] = len(variants)
            if np.random.rand() > 1.0 / float(len(variants)):
                body[self.SAMPLE_VARIANT_KEY] = np.random.choice(variants)
            else:
                body[self.VARIANTS_KEY] = variants

        self.post_improve_request(
            body_values=body,
            block=lambda result, error: (
                warn("Improve.track error: {}".format(error))
                if error and self.debug else 0,
                completion_handler(error) if completion_handler else 0),
            history_id=history_id)

    def add_reward(
            self, reward: float, reward_key: str, history_id: str = None):
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
            history_id=history_id)

    def add_rewards(
            self, rewards: Dict[str, float], completion_handler: callable = None,
            history_id: str = None, **kwargs):
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
                history_id=history_id)
        else:
            if self.debug:
                print('Skipping trackRewards for nil rewards')
            completion_handler(None) if completion_handler else None

    def track(
            self, body: Dict[str, object], completion_block: callable = None,
            history_id: str = None, **kwargs):
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
            history_id=history_id)

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
            history_id: str = None, **kwargs):

        if not history_id:
            if self.debug:
                warn("historyId cannot be nil")
            # TODO make sure what this clase in iOS-SDK does
            return None

        headers = {'Content-Type': 'application/json'}

        if self.api_key:
            headers[self.API_KEY_HEADER] = self.api_key

        body = {
            # was checked -> this convention seems to be accurate
            self.TIMESTAMP_KEY:
                str(np.datetime_as_string(
                    np.datetime64(datetime.now()), unit='ms', timezone='UTC')),
            self.HISTORY_ID_KEY: history_id,
            # TODO check if this is the desired uuid
            self.MESSAGE_ID_KEY: str(uuid4())}

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

    tracker_url = \
        'https://u0cxvugtmi.execute-api.us-west-2.amazonaws.com/test/track'
    api_key = 'xScYgcHJ3Y2hwx7oh5x02NcCTwqBonnumTeRHThI'

    it = ImproveTracker(tracker_url=tracker_url, api_key=api_key)
    example_history_id = str(uuid4())

    with open('../test_artifacts/meditations.json', 'r') as medf:
        variants = json.loads(''.join(medf.readlines()))

    variant = variants[0]
    model_name = 'py_test_post'

    with open('../test_artifacts/context.json') as cf:
        context = json.loads(''.join(cf.readlines()))

    # sanity check print
    # print('################')
    # print('variants')
    # print('################')
    # print(variants)
    # print('################')
    # print('################\n')
    #
    # print('################')
    # print('variants[0]')
    # print('################')
    # print(variants[0])
    # print('################')
    # print('################\n')
    #
    # print('################')
    # print('context')
    # print('################')
    # print(context)
    # print('################')
    # print('################\n')
    #
    # print('################')
    # print('example_history_id')
    # print('################')
    # print(example_history_id)
    # print('################')
    # print('################\n')

    it.track_decision(
        variant=variant, variants=variants, model_name=model_name,
        reward_key=model_name, history_id=example_history_id, context=context)
    # reward: float, reward_key: str, history_id:
    it.add_reward(
        reward=1.0, reward_key='py_add_reward_test',
        history_id=example_history_id)

    # rewards: Dict[str, float], completion_handler: callable = None,
    # history_id: str = None
    it.add_rewards(
        rewards={'py_add_reward_test1': 1.0, 'py_add_reward_test2': 2.0},
        history_id=example_history_id)

    # event: str, properties: Dict[str, object],
    # context: Dict[str, object] = None, history_id: str = None
    it.track_analytics_event(
        event='py_test_track_analytics_event',
        properties={'prop_str': 1, 'prop_float': 1.0}, context=context,
        history_id=example_history_id)
