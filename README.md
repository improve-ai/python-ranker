# AI Decisions, Ranking, Scoring & Multivariate Optimization for Python

Improve AI is a machine learning platform for quickly implementing app optimization, personalization, and recommendations for [Python](https://improve.ai/python-sdk/), [iOS](https://improve.ai/ios-sdk/), and [Android](https://improve.ai/android-sdk/).

The SDKs provide simple APIs for AI [decisions](https://improve.ai/decisions/), [ranking](https://improve.ai/ranking/), [scoring](https://improve.ai/scoring/), and [multivariate optimization](https://improve.ai/multivariate-optimization/) that execute immediately, on-device, with zero network latency. Decisions and rewards are tracked in the cloud with the [Improve AI Gym](https://github.com/improve-ai/gym/) and updated models are trained regularly on AWS SageMaker.

## Installation

```console
pip3 install improveai
```

## Initialization

```python
import improveai
```

```python
# track and model urls are obtained from your Improve AI Gym configuration
track_url = 'https://xxxx.lambda-url.us-east-1.on.aws/'
model_url = 'https://xxxx.s3.amazonaws.com/models/latest/greetings.xgb.gz'

greetings_model = improveai.load_model(model_url, track_url)
```

## Usage

The heart of Improve AI is the *which()* statement. *which()* is like an AI *if/then* statement.

```python
greeting, decision_id = greetings_model.which('Hello', 'Howdy', 'Hola')
```

*which()* takes a list of *variants* and returns the best - the "best" being the variant that provides the highest expected reward given the current conditions.

Decision models are easily trained with [reinforcement learning](https://improve.ai/reinforcement-learning/):

```python
if success:
   greetings_model.add_reward(1.0, decision_id)
```

With reinforcement learning, positive rewards are assigned for positive outcomes (a "carrot") and negative rewards are assigned for undesirable outcomes (a "stick").

*which()* automatically tracks it's decision with the [Improve AI Gym](https://github.com/improve-ai/gym/).

## Contextual Decisions

Unlike A/B testing or feature flags, Improve AI uses *context* to make the best decision. 

Context can be provided via *given()*:

```python

greeting, decision_id = greetings_model.given({"language": "cowboy"}) \
                                       .which("Hello", "Howdy", "Hola")
```

Given the language is *cowboy*, the variant with the highest expected reward should be *"Howdy"* and the model would learn to make that choice.


## Ranking

[Ranking](https://improve.ai/ranking/) is a fundamental task in recommender systems, search engines, and social media feeds. Fast ranking can be performed on-device in a single line of code:

```python
ranked_wines = sommelier_model.given(entree).rank(wines)
```

**Note**: Decisions are not tracked when calling *rank()*. *which()* or *decide()* must be used to train models for ranking.

## Scoring

[Scoring](https://improve.ai/scoring/) makes it easy to turn any database table into a recommendation engine.

Simply add a *score* column to the database and update the score for each row.

```python
scores = conversion_rate_model.score(rows)
```

At query time, sort the query results descending by the *score* column and the first results will be the top recommendations.

*score()* is also useful for crafting custom optimization algorithms or providing supplemental metrics in a multi-stage recommendation system.

**Note**: Decisions are not tracked when calling *score()*. *which()*, *decide()*, or *optimize()* must be used to train models for scoring.

## Multivariate Optimization

[Multivariate optimization](https://improve.ai/multivariate-optimization/) is the joint optimization of multiple variables simultaneously. This is often useful for app configuration and performance tuning.

```swift
config, decision_id = config_model.optimize({"buffer_size": [1024, 2048, 4096, 8192],
                                             "video_bitrate": [256000, 384000, 512000]})
```

This example decides multiple variables simultaneously.  Notice that instead of a single list of variants, a dictionary mapping keys to lists of variants is provided. This multi-variate mode jointly optimizes all variables for the highest expected reward.  

*optimize()* automatically tracks it's decision with the [Improve AI Gym](https://github.com/improve-ai/gym/). Rewards are credited to the most recent decision made by the model, including from a previous app session.

## Variant Types

Variants can be any JSON encodeable data structure of arbitrary complexity, including nested dicts, lists, strings, numbers, and None. Object properties and nested items within collections are automatically encoded as machine learning features to assist in the decision making process.

The following are all valid:

```python
greeting, decision_id = greetings_model.which('Hello', 'Howdy', 'Hola')

discount, decision_id = discounts_model.which(0.1, 0.2, 0.3)

enabled, decision_id = feature_flag_model.which(True, False)

item, decision_id = filter_model.which(item, None)

themes = [{"font": "Helvetica", "size": 12, "color": "#000000"},
          {"font": "Comic Sans", "size": 16, "color": "#F0F0F0"}]

theme, decision_id = themes_model.which(themes)
```

## Privacy
  
It is strongly recommended to never include Personally Identifiable Information (PII) in variants or givens so that it is never tracked, persisted, or used as training data.

## Resources

- [Quick Start Guide](https://improve.ai/quick-start/)
- [Python SDK API Docs](https://improve.ai/python-sdk/)
- [Improve AI Gym](https://github.com/improve-ai/gym/)
- [Improve AI Trainer (FREE)](https://aws.amazon.com/marketplace/pp/prodview-pyqrpf5j6xv6g)
- [Improve AI Trainer (PRO)](https://aws.amazon.com/marketplace/pp/prodview-adchtrf2zyvow)
- [Reinforcement Learning](https://improve.ai/reinforcement-learning/)
- [Decisions](https://improve.ai/multivariate-optimization/)
- [Ranking](https://improve.ai/ranking/)
- [Scoring](https://improve.ai/scoring/)
- [Multivariate optimization](https://improve.ai/multivariate-optimization/)

## Help Improve Our World

The mission of Improve AI is to make our corner of the world a little bit better each day. When each of us improve our corner of the world, the whole world becomes better. If your product or work does not make the world better, do not use Improve AI. Otherwise, welcome, I hope you find value in my labor of love. 

-- Justin Chapweske
