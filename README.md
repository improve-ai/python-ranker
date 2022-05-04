# Improve.ai for python (3.7+)


## Fast AI Decisions for Python

Improve AI provides quick on-device AI decisions that get smarter over time. It's like an AI *if/then* statement. Replace guesses in your app's configuration with AI decisions to increase your app's revenue, user retention, or any other metric automatically.

## Installation


### Installation prerequisites
 - python 3.7+
 - in order to take advantage of fast feature encoding please install `gcc` and python headers: `python3-dev` or `python3-devel` (`sudo apt install gcc python3-dev` for apt and `sudo yum install gcc python3-devel` for yum or dnf)
 - for macOS it might be necessary to [build xgboost from sources](https://xgboost.readthedocs.io/en/stable/build.html) (otherwise `pip3 install xgboost` might fail)
 - if possible virtual environment (e.g. venv) usage is strongly encouraged
 - upgrading pip, wheel and packages is also a good idea:

    `pip3 install --upgrade pip wheel build`



### Install with pip

To install from pip simply use:

`pip3 install improveai`

#### pip's cache
Fog big packages and small amount of RAM (e.g., 1 GB) pip's caching mechanism might cause Out Of Memory error resulting in 
"Killed" error message on e.g. xgboost installation attempt. 
To avoid this either purge pip's cache:

`pip3 cache purge`

or use `--no-cache-dir` flag




### Build and install from cloned git repo

To install from cloned repo:     
 1. clone repo: git clone https://github.com/improve-ai/python-sdk    
 2. make sure you are in the cloned folder (python-sdk)    
 3. activate your virtualenv (if you are using one, if not you can skip this step; using venv is advised)    
 4. purge pip's cache:
    
    `pip3 cache purge`

 5. install wheel and cmake:    
    
    `pip3 install --upgrade pip build wheel cmake --no-cache-dir`

 6. install requirements:

    `pip3 install -r requirements.txt --no-cache-dir`

 7. to build package wheel call:

    `python3 -m build`

 8. install built wheel with pip:

    `pip3 install dist/improveai-7.0.1*.whl`

    where `*` represents system specific part of wheel name

## Initialization

General initialization can be done with simple import:

```python
import improveai
```

[Gym](https://github.com/improve-ai/gym) needs new data (decisions and rewards) to train increasingly accurate models. 
Initializing `DecisionModel()` with `track_url` allows DecisionModel() to send decisions and rewards directly to [gym's](https://github.com/improve-ai/gym)
track endpoint. Easiest way to get started with deciding and rewarding is to import `DecisionModel` from `improveai`:

```python
from improveai import DecisionModel
```

Possible `DecisionModel` initializations:
- `model_name != None` and `track_url != None` &#8594; decisions of `'grettings'` model are tracked and rewarded


```python
track_url = 'https://x5fvx48stc.execute-api.<region>.amazonaws.com/track'
decision_model = DecisionModel(model_name='greetings', track_url=track_url)
```

- `model_name != None` and `track_url == None` &#8594; decisions are not tracked nor rewarded

```python
# by default track_url = None
decision_model = DecisionModel(model_name='greetings')
```

- `model_name == None` and `track_url != None` &#8594; decisions are not tracked nor rewarded 
(`model_name` must not be None for a valid decision / reward)

```python
track_url = 'https://x5fvx48stc.execute-api.<region>.amazonaws.com/track'
# model_name is an obligatory parameter of DecisionModel's constructor
decision_model = DecisionModel(model_name=None, track_url=track_url)
```

- `model_name == None` and `track_url == None` &#8594; decisions are not tracked and rewarded
```python
# model_name is an obligatory parameter of DecisionModel's constructor
decision_model = DecisionModel(model_name=None)
```

Once model is initialized an existing XGBoost Improve AI model can be loaded. 
If `DecisionModel` was initialized with `model_name = None` then the `model_name` cached 
in the loaded booster will be set to `model_name` attribute of `DecisionModel`.

```python
decision_model.load(model_url='<URL or FS path to booster>')
```


## Usage

### Greeting

Improve AI makes quick on-device AI decisions that get smarter over time.

The heart of Improve AI is the which statement. `which()` is like an AI if/then statement.

```python
from improveai import DecisionModel


# initialize model with model name  and track URL
greetings_model = DecisionModel(model_name='greetings', track_url='<gym`s track url>')
# load desired booster from URL or filesystem
greetings_model.load('<greetings model URL or filesystem path>')
# choose best variant
greeting, decision_id = greetings_model.which('Hello', 'Howdy', 'Hola')
```

`which()` makes decisions using a decision model. Decision models are easily trained by assigning rewards for positive outcomes.

```python
# add reward to the Decision; without valid track_url rewards will not be added to Decision
greetings_model.add_reward(reward=1.0, decision_id=decision_id)
```

Rewards are credited to the specific decisions - you can add rewards with `add_reward()` call specifying desired reward and `decision_id`.
`DecisionModel().which(*variants)` will make the decision that provides the highest expected reward. 
When the rewards are business metrics, such as revenue or user retention, 
the decisions will optimize to automatically improve those metrics over time.

That's like A/B testing on steroids.


### Numbers Too

What discount should we offer?

```python
from improveai import DecisionModel


discounts_model = DecisionModel(model_name='discounts', track_url='<gym`s track url>')
discounts_model.load(model_url='<discounts model URL>')
discount, _ = discounts_model.which(0.1, 0.2, 0.3)
```

## Booleans

Dynamically enable feature flags for best performance...

```python
from improveai import DecisionModel


# example decision attributes
example_attributes = {'string attribute': 'string attribute value',
                      'float attribute': 123.132,  # float attribute value
                      'bool attribute': True}  # bool attribute value

features_model = DecisionModel(model_name='feature_flags', track_url=track_url)
features_model.load(model_url='<features model URL>')

# choose best feature flag considering example attributes
feature_flag, _ = features_model.given(givens=example_attributes).which(True, False)
```


### Complex Objects


```python
from improveai import DecisionModel


theme_variants = [
    {"textColor": "#000000", "backgroundColor": "#ffffff" },
    { "textColor": "#F0F0F0", "backgroundColor": "#aaaaaa" }]

themes_model = DecisionModel(model_name='themes', track_url='<gym`s track url>')
themes_model.load('<themes model URL>')

# lists of variants should be passed to which() as pythonic *args
theme, _ = themes_model.which(*theme_variants)
```
`DecisionModel.which()` accepts pythonic `*args`. 
`DecisionModel.which(*variants)` will unpack each element of `varaints` as a separate variant  
while `DecisionModel.which(variants)` will interpret `variants` as a single variant of a list type.

Improve learns to use the attributes of each key and value in a complex variant to make the optimal decision.

Variants can be any JSON encodeable data structure of arbitrary complexity, including nested dictionaries, arrays, strings, numbers, nulls, and booleans.

### Decisions are Contextual

Unlike A/B testing or feature flags, Improve AI uses *context* to make the best decision for each user.
On iOS, the following *context* is automatically included:

- $country - two letter country code
- $lang - two letter language code
- $tz - numeric GMT offset
- $carrier - cellular network
- $device - string portion of device model
- $devicev - device version
- $os - string portion of OS name
- $osv - OS version
- $pixels - screen width x screen height
- $app - app name
- $appv - app version
- $sdkv - Improve AI SDK version
- $weekday - (ISO 8601, monday==1.0, sunday==7.0) plus fractional part of day
- $time - fractional day since midnight
- $runtime - fractional days since session start
- $day - fractional days since born
- $d - the number of decisions for this model
- $r - total rewards for this model
- $r/d - total rewards/decisions
- $d/day - decisions/$day



Using the context, on a Spanish speaker's device we expect our greetings model to learn to choose *"Hola"*.


Custom context can be provided via given():


```python
from improveai import DecisionModel

cowboy_givens = {"language": "cowboy"}

greetings_model = DecisionModel(model_name='greetings', track_url=track_url)
greetings_model.load('<trained greetings model url>')
greeting, _ = greetings_model.given(givens=cowboy_givens).which("Hello", "Howdy", "Hola")
```

Given the language is *cowboy*, the variant with the highest expected reward should be *"Howdy"* and the model would learn to make that choice.

### Example: Optimizing an Upsell Offer

Improve AI is powerful and flexible. Variants can be any JSON encodeable data structure including **strings**, 
**numbers**, **booleans**, **lists**, and **dictionaries**.

For a dungeon crawler game, say the user was purchasing an item using an In App Purchase. 
We can use Improve AI to choose an additional product to display as an upsell offer during checkout. 
With a few lines of code, we can train a model that will learn to optimize the upsell offer given the original product being purchased.

```python
from improveai import DecisionModel


# create DecisionModel object
upsell_model = DecisionModel(model_name='upsell_model', track_url=track_url)

# load model from a path / url
upsell_model_url = '<upsell model url>'
upsell_model.load(model_url=upsell_model_url)

product = {'name': 'red sword', 'price': 4.99}
# create upsell Decision object
upsell, upsell_decision_id = \
    upsell_model.given(givens=product).\
        which(*[{ "name": "gold", "quantity": 100, "price": 1.99 },
                { "name": "diamonds", "quantity": 10, "price": 2.99 },
                { "name": "red scabbard", "price": 0.99 }])
```

The product to be purchased is the red sword. Notice that the variants are dictionaries with a mix of string and numeric values.

The rewards in this case might be any additional revenue from the upsell.

```python
upsell_purchased = True  # flag indicating if decision was correct

if upsell_purchased:
    # assign reward
    upsell_model.add_reward(upsell['price'], decision_id=upsell_decision_id)
```

While it is reasonable to hypothesize that the red scabbard might be the best upsell offer to pair with the red sword, it is still a guess. Any time a guess is made on the value of a variable, instead use Improve AI to decide.

*Replace guesses with AI decisions.*


### Example: Performance Tuning

In the 2000s I was writing a lot of video streaming code. 
The initial motivation for Improve AI came out of my frustrations with attempting to 
tune video streaming clients across heterogeneous networks.

I was forced to make guesses on performance sensitive configuration defaults through 
slow trial and error. My client configuration code maybe looked something like this:

```python
config = {"bufferSize": 2048,
          "videoBitrate": 384000}
```

This is the code I wish I could have written:

```python
from itertools import product


buffer_sizes = [1024, 2048, 4096, 8192]
video_bit_rates = [256000, 384000, 512000]

config, config_decision_id = \
    config_model.which(*[{"bufferSize": bs, "videoBitrate": br} for bs, br in product(*[buffer_sizes, video_bit_rates])])
```

This example decides multiple variables simultaneously.
This multi-variate mode jointly optimizes both variables for the highest expected reward.

[//]: # (Notice that instead of a single list of variants, a dictionary mapping keys to lists of variants is provided to which. )

The rewards in this case might be negative to penalize any stalls during video playback.


```python
if video_stalled:
    config_model.add_reward(-0.001, decision_id=config_decision_id)
```

Improve AI frees us from having to overthink our configuration values during development. 
We simply give it some reasonable variants and let it learn from real world usage.

Look for places where you're relying on guesses or an executive decision and consider 
instead directly optimizing for the outcomes you desire.

## Privacy
  
It is strongly recommended to never include Personally Identifiable Information (PII) in variants or givens so that it is never tracked, persisted, or used as training data.

## Help Improve Our World

The mission of Improve AI is to make our corner of the world a little bit better each day. When each of us improve our corner of the world, the whole world becomes better. If your product or work does not make the world better, do not use Improve AI. Otherwise, welcome, I hope you find value in my labor of love. - Justin Chapweske
