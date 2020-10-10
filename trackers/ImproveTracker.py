from datetime import datetime
import json
import numpy as np
from typing import Dict, List
from uuid import uuid4
from warnings import warn

from utils.gen_purp_utils import constant


class ImproveTracker:

    @constant
    def model_key() -> str:
        return "model"

    @constant
    def history_id_key() -> str: 
        return "history_id"

    @constant
    def timestamp_key() -> str:
        return "timestamp"

    @constant
    def message_id_key() -> str:
        return "message_id"

    @constant
    def type_key() -> str:
        return "type"

    @constant
    def variant_key() -> str:
        return "variant"

    @constant
    def context_key() -> str:
        return "context"

    @constant
    def rewards_key() -> str:
        return "rewards"

    @constant
    def variants_count_key() -> str:
        return "variants_count"

    @constant
    def variants_key() -> str:
        return "variants"

    @constant
    def sample_variant_key() -> str:
        return "sample_variant"

    @constant
    def reward_key_key() -> str:
        return "reward_key"

    @constant
    def event_key() -> str:
        return "event"

    @constant
    def properties_key() -> str:
        return "properties"

    @constant    
    def decision_type() -> str:
        return "decision"

    @constant
    def rewards_type() -> str:
        return "rewards"

    @constant
    def event_type() -> str:
        return "event"

    @constant
    def api_key_header() -> str:
        return "x-api-key"

    @constant
    def history_id_defaults_key() -> str:
        return "ai.improve.history_id"

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

    def __init__(self, tracker_url: str, api_key: str):
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

        if not variant:
            warn(
                "Skipping trackDecision for nil variant. To track null values "
                "use [NSNull null]")

        usd_reward_key = reward_key
        if not usd_reward_key:
            warn("Using model name as rewardKey: {}".format(model_name))
            usd_reward_key = model_name

        body = {
            self.type_key: self.decision_type,
            self.variant_key: variant,
            self.model_key: model_name,
            self.rewards_key: usd_reward_key}

        if context:
            body[self.context_key] = context

        if variants and len(variants) > 0:
            body[self.variants_count_key] = len(variants)
            if np.random.rand() > 1.0 / float(len(variants)):
                body[self.sample_variant_key] = np.random.choice(variants)
            else:
                body[self.variants_key] = variants

        # def semi_block(result, error):
        #     if error:
        #         warn("Improve.track error: {}".format(error))
        #     if completion_handler:
        #         completion_handler(error)

        self.post_improve_request(
            body_values=body,
            block=lambda result, error: (
                warn("Improve.track error: {}".format(error)) if error else 0,
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
            print("Tracking rewards: {}".format(rewards))
            track_body = {
                self.type_key: self.rewards_type,
                self.rewards_key: rewards}

            self.track(
                body=track_body,
                completion_block=lambda error:
                    completion_handler(error) if completion_handler else 0,
                history_id=history_id)
        else:
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
                warn("Improve.track error: {}".format(error)) if error else 0,
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

        body = {self.type_key: self.event_type}

        for key, val in zip(
                [self.event_key, self.properties_key, self.context_key],
                [event, properties, context]):
            if val:
                body[key] = val

        self.track(body=body, history_id=history_id)

    def post_improve_request(
            self, body_values: Dict[str, object], block: callable,
            history_id: str = None, **kwargs):

        if not history_id:
            warn("historyId cannot be nil")
            # TODO make sure what this clase in iOS-SDK does
            return None

        headers = {'Content-Type': 'application/json'}

        if self.api_key:
            headers[self.api_key_header] = self.api_key

        body = {
            # TODO check if proper time format is returned -> implement
            #  timestamp_from_date myby (?)
            self.timestamp_key:
                np.datetime_as_string(
                    np.datetime64(datetime.now()), unit='ms', timezone='UTC'),
            self.history_id_key: history_id,
            # TODO check if this is the desired uuid
            self.message_id_key: uuid4()}

        body.update(body_values)

        try:
            payload_json = json.dumps(body)
        except Exception as exc:
            warn("Data serialization error: {}\nbody: {}".format(exc, body))
            return None
        # TODO finish up
