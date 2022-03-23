# Improve.ai for python (3.7.9)


## Fast AI Decisions for Python

[![License](https://img.shields.io/cocoapods/l/Improve.svg?style=flat)](http://cocoapods.org/pods/Improve)

It's like an AI *if/then* statement. Quickly make decisions that configure your app to maximize revenue, performance, user retention, or any other metric.


## Installation

Python-SDK for Improve.ai is currently available only on github.

To install python-sdk for Improve.ai:

 0. install prerequisites:
    - Fedora:
      - sudo yum groupinstall "Development Tools"
      - sudo yum install python3-devel python3-Bottleneck python3-numpy
    - Amazon Linux 2:
      - sudo yum update
      - yum groupinstall "Development Tools"
      - sudo yum install python3-devel numpy
    - Ubuntu (18.04 and 20.04):
      - sudo apt install build-essential make gcc python3-dev python3-numpy python3-venv
    - macOS:
      - pyenv (allows ot install python 3.7.9)
      - xgboost should be built
      - venv usage is strongly encouraged
     
 1. clone repo: git clone https://github.com/improve-ai/python-sdk
    
 2. make sure you are in the cloned (python-sdk) folder
    
 3. activate your virtualenv (if you are using one, if not you can skip this step; using venv is advised)
    
 4. install wheel and cmake:

    
    pip3 install wheel cmake --no-cache-dir

    
 5. install requirements:

    
    pip3 install -r requirements.txt --no-cache-dir


 6. run the following commands to install improveai python package:

    
    python3 setup.py install


### Hello World!

What is the best greeting?

```python
from improveai import DecisionModel

# prepare JSON encodable variants to choose from:
hello_variants = [
    {'text': "Hello World!"},
    {'text': "Hi World!"},
    {'text': "Howdy World!"}]

givens = {'language': 'cowboy'}

# Get the best greeting
greeting = \
    DecisionModel(model_name="greetings").given(givens=givens)\
    .choose_from(variants=hello_variants).get()
```

*greeting* should result in *Howdy World* assuming it performs best when *language* is *cowboy*.


### Numbers Too

What discount should we offer?

```python
discount = DecisionModel(model_name='discounts').choose_from(variants=[0.1, 0.2, 0.3]).get()
```

## Booleans

Dynamically enable feature flags for best performance...

```python
feature_flag = DecisionModel(model_name='feature_flags').choose_from(variants=[True, False]).get()
```


### Complex Objects


```python
theme_variants = [
    {"textColor": "#000000", "backgroundColor": "#ffffff" },
    { "textColor": "#F0F0F0", "backgroundColor": "#aaaaaa" }]

theme = DecisionModel(model_name='themes').choose_from(variants=theme_variants).get()
```

Improve learns to use the attributes of each key and value in a complex variant to make the optimal decision.

Variants can be any JSON encodeable data structure of arbitrary complexity, including nested dictionaries, arrays, strings, numbers, nulls, and booleans.


## Models

A *DecisionModel* contains the AI decision logic, analogous to a large number of *if/then* statements.

Models are thread-safe and a single model can be used for multiple decisions.

### Synchronous Model Loading

```python
product = DecisionModel(model_name=None).load(model_url=model_url).choose_from(["clutch", "dress", "jacket"]).get()
```

Models can be loaded from the app bundle or from https URLs.

### Asynchronous Model Loading

Asynchronous model loading allows decisions to be made at any point, even before the model is loaded.  If the model isn't yet loaded or fails to load, the first variant will be returned as the decision.

```python
from improveai import DecisionModel, DecisionTracker

track_url = 'http://your.track.url'
api_key = '<tracker API key>'
history_id = '<history_id to be used by tracker>'

tracker = DecisionTracker(track_url=track_url, api_key=api_key, history_id=history_id)

model_url = '/ model/ path / or / url'

model = DecisionModel(model_name="greetings")
model.track_with(tracker=tracker)
model.load_async(model_url=model_url)

# It is very unlikely that the model will be loaded by the time this is called, 
# so "Hello World" would be returned and tracked as the decision
greeting = model.choose_from(variants=['Hello World', 'Howdy World', 'Yo World']).get()
```

## Tracking & Training Models

The magic of Improve AI is it's learning process, whereby models continuously improve by training on past decisions. To accomplish this, decisions and events are tracked to your deployment of the Improve AI Gym.

### Tracking Decisions

Set a *DecisionTracker* on the *DecisionModel* to automatically track decisions and enable learning.  A single *DecisionTracker* instance can be shared by multiple models.

```python
tracker = DecisionTracker(track_url=track_url)  # trackUrl is obtained from your Gym configuration

font_size = \
    DecisionModel(model_name=None).load(model_url=model_url).track_with(tracker=tracker).chooseFrom([12, 16, 20]).get()
```

The decision is lazily evaluated and then automatically tracked as being causal upon calling *get()*.

For this reason, wait to call *get()* until the decision will actually be used.

### Tracking Events

Events are the mechanism by which decisions are rewarded or penalized.  In most cases these will mirror the normal analytics events that your app tracks and can be integrated with any event tracking singletons in your app.

```python
tracker.track_event(event_name="Purchased", properties={"product_id": 8, "value": 19.99})
```

Like most analytics packages, *track* takes an *event* name and an optional *properties* dictionary.  The only property with special significance is *value*, which indicates a reward value for decisions prior to that event.  

If *value* is ommitted then the default reward value of an event is *0.001*.

By default, each decision is rewarded the total value of all events that occur within 48 hours of the decision.

Assuming a typical app where user retention and engagement are valuable, we recommend tracking all of your analytics events with the *DecisionTracker*.  You can customize the rewards assignment logic later in the Improve AI Gym.

## Privacy
  
It is strongly recommended to never include Personally Identifiable Information (PII) in variants or givens so that it is never tracked, persisted, or used as training data.

## An Ask

Thank you so much for enjoying my labor of love. Please only use it to create things that are good, true, and beautiful. - Justin

## License

Improve AI is copyright Mind Blown Apps, LLC. All rights reserved.  May not be used without a license.


[comment]: <> (---)

[comment]: <> (### ImproveModel CLI)

[comment]: <> (The prepared CLI takes as input:)

[comment]: <> ( - one of supported method names to execute:)

[comment]: <> (    - score - calculates predictions on all provided input data)

[comment]: <> (    - sort - scores input and returns it ordered descendingly)

[comment]: <> (    - choose - scores input and returns best choice along with score)

[comment]: <> ( - desired model type to use out of 2 supported mdoel types:)

[comment]: <> (    - *.mlmodel)

[comment]: <> (    - *.xgb &#40;xgboost native format&#41;)

[comment]: <> ( - path to desired model)

[comment]: <> ( - JSON string with input data encapsulated with '' -> '<json string>')

[comment]: <> ( - JSON string with context encapsulated with '' -> '<json string>')

[comment]: <> (In order to use prepared ImproveModel CLI:)

[comment]: <> ( - make sure to change directory to python-sdk folder)

[comment]: <> ( - call improve_model_cli.py in the following fashion <br>)

[comment]: <> ( python3.7 improve_model_cli.py [desired method name] [desired model type] [path to deired model] --variant [input JSON string] --context [context JSON string] --model_metadata [metadata JSON string])
 
[comment]: <> (To see example results please call: <br>)

[comment]: <> (python3.7 improve_model_cli.py score xgb_native test_artifacts/model.xgb)

[comment]: <> (To use CLI with files &#40;i.e. for variants/context/model metadata/results&#41; please use:)

[comment]: <> (python3.7 improve_model_cli.py score xgb_native artifacts/models/improve-messages-2.0.xgb --variants_pth artifacts/data/real/meditations.json --context_pth artifacts/test_artifacts/sorting_context.json --results_pth artifacts/results/20_11_2020_meditations_sanity_check.json --prettify_json )

[comment]: <> (### Results)

[comment]: <> (Currently supported objectives:)

[comment]: <> ( - regression - returns [input JSON string, score value, 0] for each observation)

[comment]: <> ( - binary classification - returns [input JSON string, class 1 probability, class 1 label] for each observation)

[comment]: <> ( - multiple classification - returns [input JSON string, highest class probability, most probable class label] for each observation)

[comment]: <> (Results are always returned as a JSON strings: <br>)

[comment]: <> ([[input JSON string, value, label], ...])

[comment]: <> ( - score method returns list of all inputs scored)

[comment]: <> ( - sort method returns list of all inputs scored &#40;if multiclass classification is the case then the list is sorted for each class from highest to lowest scores&#41;)

[comment]: <> ( - choose method returns best highest scored variant info &#40;[input JSON string, value, label]&#41;. For binary classification best scores for class 1 are returned. For multiple classification best choices in each class are returned. Ties are broken randomly. )


[comment]: <> (### Model Conversion)

[comment]: <> (To convert **xgboost** model to **mlmodel** please use:)

[comment]: <> (python3.7 transformers/mlmodels_generators.py --src_model_pth test_artifacts/model.xgb --model_metadata_pth test_artifacts/model.json --trgt_model_pth test_artifacts/conv_model.mlmodel)

[comment]: <> (### XGBoost model appending)

[comment]: <> (To append **xgboost** model with desired model metadata please use:)

[comment]: <> (python3.7 transformers/xgb_model_generators.py --src_model_pth test_artifacts/model.xgb --model_metadata_pth test_artifacts/model.json --trgt_model_pth test_artifacts/conv_model.mlmodel )


[comment]: <> (## Cython compatibility issues fix -> symlink numpy)

[comment]: <> (sudo ln -s /usr/lib/python3.9/dist-packages/numpy/core/include/numpy /usr/include/numpy)