# Improve.ai for python (3.7+)


## Fast AI Decisions for Python

Improve AI provides quick on-device AI decisions that get smarter over time. It's like an AI *if/then* statement. Replace guesses in your app's configuration with AI decisions to increase your app's revenue, user retention, or any other metric automatically.

## Installation


### Installation prerequisites
 - python 3.7+
 - `gcc` and python headers: `python3-dev` or `python3-devel` (`sudo apt install gcc python3-dev` for apt and `sudo yum install gcc python3-devel` for yum or dnf)
 - for macOS you might also need to [build xgboost from sources](https://xgboost.readthedocs.io/en/stable/build.html) (otherwise `pip3 install xgboost` might fail)
 - if possible virtual environment (e.g. venv) usage is strongly encouraged


### Install with pip

To install from pypi sources simply use pip:

`pip3 install improveai`


### Install from cloned git repo

To install from cloned repo:     
 1. clone repo: git clone https://github.com/improve-ai/python-sdk    
 2. make sure you are in the cloned folder (python-sdk)    
 3. activate your virtualenv (if you are using one, if not you can skip this step; using venv is advised)    
 4. install wheel and cmake:    
    `pip3 install wheel cmake --no-cache-dir`
 5. install requirements:    
    `pip3 install -r requirements.txt --no-cache-dir`
 6. to build package wheel call:
    `python3 -m build`
 7. install built wheel with pip: 
    `pip3 install dist/improveai-7.0.1*.whl`
    where `*` represents system specific part of wheel name


## Usage

### Greeting

Improve AI makes quick on-device AI decisions that get smarter over time.

The heart of Improve AI is the which statement. which is like an AI if/then statement.

```python
from improveai import DecisionModel

greeting = DecisionModel(model_name='greetings').which('Hello', 'Howdy', 'Hola')
```

`which()` makes decisions on-device using a decision model. Decision models are easily trained by assigning rewards for positive outcomes.

```python
from improveai import DecisionModel

# create an instance of Decision object
decision = DecisionModel(model_name='greetings').choose_from(variants=['Hello', 'Howdy', 'Hola'])
# choose best greeting with get()
best_greeting = decision.get()

# add reward to the Decision
decision.add_reward(reward=1.0)
```

Rewards are credited to the decisions - you can add them to the existing decision object with an `add_reward()` call. 
`Decision.get()` will make the decision that provides the highest expected reward. 
When the rewards are business metrics, such as revenue or user retention, 
the decisions will optimize to automatically improve those metrics over time.

That's like A/B testing on steroids.


### Numbers Too

What discount should we offer?

```python
from improveai import DecisionModel

discount = DecisionModel(model_name='discounts').which(0.1, 0.2, 0.3)
```

## Booleans

Dynamically enable feature flags for best performance...

```python
from improveai import DecisionModel

# example decision attributes
givens = {'string attribute': 'string attribute value',
          'float attribute': 123.132,  # float attribute value
          'bool attribute': True}  # bool attribute value

# choose best feature flag considering example attributes
feature_flag = DecisionModel(model_name='feature_flags').given(givens=givens).which(True, False)
```


### Complex Objects


```python
from improveai import DecisionModel

theme_variants = [
    {"textColor": "#000000", "backgroundColor": "#ffffff" },
    { "textColor": "#F0F0F0", "backgroundColor": "#aaaaaa" }]

# lists of variants should be passed to which() as pythonic *args
theme = DecisionModel(model_name='themes').which(*theme_variants)
```

Passing list of `variants` as pythonic `*args` will make `which()` interpret it as a list of variants
`DecisionModel.which()` accepts pythonic `*args`. 
This means that `DecisionModel.which(*variants)` will interpret each element of `variants` list as a separate variant
while `DecisionModel.which(variants)` will interpret `variants` as a single variant of a list type.

Improve learns to use the attributes of each key and value in a complex variant to make the optimal decision.

Variants can be any JSON encodeable data structure of arbitrary complexity, including nested dictionaries, arrays, strings, numbers, nulls, and booleans.

### Decisions are Contextual

Unlike A/B testing or feature flags, Improve AI uses context to make the best decision for each user.
Custom context can be provided via given().

```python
from improveai import DecisionModel

cowboy_givens = {"language": "cowboy"}

# create model object
greetings_model = DecisionModel(model_name='greetings')
# load greetings model 
greetings_model.load('<trained greetings model url>')

greeting = greetings_model.given(givens=cowboy_givens).which("Hello", "Howdy", "Hola")
```

Given the language is cowboy and `greetings_model` was successfully loaded, 
the variant with the highest expected reward should be *"Howdy"* and the model would learn to make that choice.

### Example: Optimizing an Upsell Offer

Improve AI is powerful and flexible. Variants can be any JSON encodeable data structure including strings, 
numbers, booleans, lists, and dictionaries.

For a dungeon crawler game, say the user was purchasing an item using an In App Purchase. 
We can use Improve AI to choose an additional product to display as an upsell offer during checkout. 
With a few lines of code, we can train a model that will learn to optimize the upsell offer given the original product being purchased.

```python
from improveai import DecisionModel


# create DecisionModel object
upsell_model = DecisionModel(model_name='upsell_model')

# load model from a path / url
upsell_model_url = '<example upsell model url>'
upsell_model.load(model_url=upsell_model_url)

product = {'name': 'red sword', 'price': 4.99}
# create upsell Decision object
upsell_decision = \
    upsell_model.given(givens=product).\
        choose_from(
        [{ "name": "gold", "quantity": 100, "price": 1.99 },
         { "name": "diamonds", "quantity": 10, "price": 2.99 },
         { "name": "red scabbard", "price": 0.99 }])

# get best upsell
upsell = upsell_decision.get()
```

The product to be purchased is the red sword. Notice that the variants are dictionaries with a mix of string and numeric values.

The rewards in this case might be any additional revenue from the upsell.

```python
upsell_purchased = True  # flag indicating if decision was correct

if upsell_purchased:
    upsell_decision.add_reward(upsell['price'])
```

While it is reasonable to hypothesize that the red scabbord might be the best upsell offer to pair with the red sword, it is still a guess. Any time a guess is made on the value of a variable, instead use Improve AI to decide.

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
config = configModel.which({"bufferSize": [1024, 2048, 4096, 8192],
                            "videoBitrate": [256000, 384000, 512000]})
```

Improve AI frees us from having to overthink our configuration values during development. 
We simply give it some reasonable variants and let it learn from real world usage.

Look for places where you're relying on guesses or an executive decision and consider 
instead directly optimizing for the outcomes you desire.

## Privacy
  
It is strongly recommended to never include Personally Identifiable Information (PII) in variants or givens so that it is never tracked, persisted, or used as training data.

## Help Improve Our World

Thank you so much for enjoying my labor of love. Please onlyThe mission of Improve AI is to make our corner of the world a little bit better each day. When each of us improve our corner of the world, the whole world becomes better. If your product or work does not make the world better, do not use Improve AI. Otherwise, welcome, I hope you find value in my labor of love. - Justin Chapweske
