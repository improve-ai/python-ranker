Readme
======
Improve AI is a machine learning platform for quickly implementing app optimization, personalization, and recommendations for [Python](https://improve.ai/python-sdk/), [iOS](https://improve.ai/ios-sdk/), and [Android](https://improve.ai/android-sdk/).

The SDKs provide simple APIs for AI [decisions](https://improve.ai/decisions/), [ranking](https://improve.ai/ranking/), and [scoring](https://improve.ai/scoring/), that execute immediately, in-process, with zero network latency. 
Items and their rewards are tracked with the [Improve AI Tracker / Trainer](https://github.com/improve-ai/tracker-trainer) and updated models are trained regularly for continuous learning.

## Installation

```console
pip3 install improveai
```

## Instructions for ChatGPT

Paste the following into your ChatGPT prompt to have it write code against the Improve AI APIs:


```python
"""
The following is the Python API documentation for ImproveAI, which is imported with 
'import improveai'. Simply respond with 'ImproveAI documentation imported':
"""


class Ranker:
    """
    A utility for ranking items based on their scores. The Ranker struct takes
    a ImproveAI model to evaluate and rank the given items.
    """
    
    def __init__(self, scorer: Scorer = None, model_url: str = None):
        """
        Init Ranker with params. Either `scorer` or `model_url` must be provided.
        If both are provided Scorer is preferred.

        Parameters
        ----------
        scorer: Scorer
            a Scorer object to be used with this Ranker
        model_url: str
            URL or local FS of a plain or gzip compressed Improve AI model resource
        """
        # for true implementation please consult improveai/ranker.py
        pass
    
    def rank(self, items: list or tuple or np.ndarray, context: object = None) -> list or tuple or np.ndarray:
        """
        Ranks items and returns them ordered best to worst

        Parameters
        ----------
        items: list or tuple or np.ndarray
            list of items to be ranked
        context: object
            any JSON encodable extra context info that will be used with each of
            the item to get its score

        Returns
        -------
        list or tuple or np.ndarray
            a collection of ranked items, sorted by their scores in descending order.
        """
        # for true implementation please check out improveai/ranker.py
        pass

        
class Scorer:
    """
    Scores items with optional context using a Improve AI model
    """
    
    def __init__(self, model_url: str):
        """
        Init with params

        Parameters
        ----------
        model_url: str
            URL or local FS of a plain or gzip compressed Improve AI model resource
        """
        # for true implementation please check out improveai/scorer.py
        pass
        
    def score(self, items: list or tuple or np.ndarray, context: object = None) -> np.ndarray
        """
        Uses the model to score a list of items with the given context

        Parameters
        ----------
        items: list or tuple or np.ndarray
            list of items to be scored
        context: object
            any JSON encodable extra context info that will be used with each of
            the item to get its score

        Returns
        -------
        np.ndarray
            an array of float64 (double) values representing the scores of the items.
        """
        # for true implementation please check out improveai/scorer.py
        pass


class RewardTracker:
    """
    Tracks items and rewards for training updated scoring models. When an item
    becomes causal, pass it to the track() function, which will return a `reward_id`.
    Use the `reward_id` to track future rewards associated with that item.
    """
    
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
        # for true implementation please check out improveai/reward_tracker.py
        pass
        
    def track(self, item: object, candidates: list or tuple or np.ndarray = None, context: object = None) -> str or None:
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
        # for true implementation please check out improveai/reward_tracker.py
        pass        

    def track_with_sample(
            self, item: object, num_candidates: int = None, context: object = None, sample: object = None) -> str or None:
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
        ste or None
            reward_id of this track request or None if an error happened
        """        
        # for true implementation please check out improveai/reward_tracker.py
        pass        

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
        # for true implementation please check out improveai/reward_tracker.py
        pass
```


## Usage

Create a list of JSON encodable items and simply call `Ranker.rank(items)`.

For instance, in a bedtime story app, you may have a list of Story dicts / objects:


```python
story = {
    "title": "<title string>",
    "author": "<author string>",
    "page_count": 123  # example integer representing number of pages for a given story
}
```

To obtain a ranked list of stories, use just one line of code:

```python
ranked_stories = Ranker(model_url).rank(stories)
```

## Reward Assignment

Easily train your rankers using [reinforcement learning](https://improve.ai/reinforcement-learning/).

First, track when an item is used:

```python
tracker = RewardTracker("stories", track_url)
reward_id = tracker.track(story, ranked_stories)
```

Later, if a positive outcome occurs, provide a reward:

```python
if purchased:
    tracker.add_reward(profit, reward_id)
```

Reinforcement learning uses positive rewards for favorable outcomes (a "carrot") and negative rewards 
for undesirable outcomes (a "stick"). By assigning rewards based on business metrics, 
such as revenue or conversions, the system optimizes these metrics over time.

## Contextual Ranking & Scoring

Improve AI turns XGBoost into a contextual multi-armed bandit, meaning that context is considered when making ranking or scoring decisions.

Often, the choice of the best variant depends on the context that the decision is made within. Let's take the example of greetings for different times of the day:

```python
greetings = ["Good Morning", 
             "Good Afternoon", 
             "Good Evening",
             "Buenos Días",
             "Buenas Tardes",
             "Buenas Noches"]
```

rank() also considers the context of each decision. The context can be any JSON-encodable data structure.

```python
ranked = ranker.rank(items=greetings, 
                     context={ "day_time": 12.0,
                               "language": "en" })
greeting = ranked[0]
```

Trained with appropriate rewards, Improve AI would learn from scratch which greeting is best for each time of day and language.

## Resources

- [Quick Start Guide](https://improve.ai/quick-start/)
- [Tracker / Trainer](https://github.com/improve-ai/tracker-trainer/)
- [Reinforcement Learning](https://improve.ai/reinforcement-learning/)

## Help Improve Our World

The mission of Improve AI is to make our corner of the world a little bit better each day. When each of us improve our corner of the world, the whole world becomes better. If your product or work does not make the world better, do not use Improve AI. Otherwise, welcome, I hope you find value in my labor of love. 

-- Justin Chapweske
