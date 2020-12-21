# Improve.ai for python (3.7.9)

## An AI library for Making Great Choices

[![License](https://img.shields.io/cocoapods/l/Improve.svg?style=flat)](http://cocoapods.org/pods/Improve)

Quickly choose and sort objects to maximize user retention, performance, revenue, or any other metric. It's like an AI if / then statement.

Improve.ai performs fast machine learning on any `JSON` encodable data structure including dictionaries, arrays, strings, integers, floats, and null values.

## Installation

Python-SDK for Improve.ai is currently available only on github.

To install python-sdk for Improve.ai:
 1. clone repo: git clone https://github.com/improve-ai/python-sdk
 2. make sure you are in the cloned (python-sdk) folder 
 3. activate your virtualenv (if you are using one, if not you can skip this step; using venv is advised)
 4. install requirements:
    
    pip3 install -r requirements.txt    

 5. run the following commands to install improve_ai python package:
    
    python3 setup.py build_ext --inplace && python3 setup.py install


## Import and initialize the SDK.

```python
from improveai.trackers.decision_trackers import DecisionTracker

endpoint_url = '<desired tracker endpoint url>'

# Call with no API key
tracker = DecisionTracker(track_url=endpoint_url)

# Call with API key
api_key = '<API key for endpoint_url>'
tracker = DecisionTracker(track_url=endpoint_url, api_key=api_key)
```

To obtain the model bundle URL and api key, first deploy an Improve Model Gateway (link).

Currently SDK supports:
 - loading models from a file (also a gziped one)
 - downloading models as raw or gziped streams from specified URL
 - 2 types of models:
   - mlmodel (coremltools type)
   - native xgboost type
    
To load desired model / model from desired endpoint (Improve.ai trains new model daily).

```python
from improveai.models.decision_models import DecisionModel

model_kind = 'xgb_native'
model_pth = '<path / or / URL / to / model>'

dm = DecisionModel(model_pth=model_pth, model_kind=model_kind)
```

### Hello World!

What is the best greeting?

```python
from improveai.models.decision_models import DecisionModel
from improveai.decisions.v6 import Decision 

model_kind = 'xgb_native'
model_pth = '<path / or / URL / to / model>'

dm = DecisionModel(model_pth=model_pth, model_kind=model_kind)

# prepare JSON encodable variants to choose from:
# [@"Hello World!", @"Hi World!", @"Howdy World!"]
variants = [
    {'text': "Hello World!"},
    {'text': "Hi World!"},
    {'text': "Howdy World!"}]

d = Decision(variants=variants, model=dm, model_name='greetings')

# Get the best greeting
best_hello_world = d.best()

# Train model using decision
tracker.track_using_best_from(
    decision=d, message_id='<unique msg id>', history_id='<unique history id>', 
    timestamp='<datetime64 timestamp or None>')

# ... later when the `best_hello_world` is chosen, give the decision a reward
tracker.add_reward(
    reward=1.0, reward_key='greetings', message_id='<unique msg id>', 
    history_id='<unique history id>', timestamp='<datetime64 timestamp or None>')
```

Improve quickly learns to choose the greeting with the highest chance of button tap.

```'greeting'``` in this example is the namespace for the type of variant being chosen. Namespaces ensure that multiple uses of Improve in the same project are decided and trained separately. 
A namespace can be a simple string like "discount" or "song" or can be more complicated like "SubscriptionViewController.buttonText". 
Namespace strings are opaque and can be any format you wish.


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

[comment]: <> (sudo ln -s /usr/lib/python3.7/dist-packages/numpy/core/include/numpy /usr/include/numpy)