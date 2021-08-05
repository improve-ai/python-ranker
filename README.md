# Improve.ai for python (3.7.9)

## An AI library for Making Great Choices

[![License](https://img.shields.io/cocoapods/l/Improve.svg?style=flat)](http://cocoapods.org/pods/Improve)

Quickly choose and sort objects to maximize user retention, performance, revenue, or any other metric. It's like an AI if / then statement.

Improve.ai performs fast machine learning on any `JSON` encodable data structure including dictionaries, arrays, strings, integers, floats, and null values.

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
     
 1. clone repo: git clone https://github.com/improve-ai/python-sdk
    
 2. make sure you are in the cloned (python-sdk) folder
    
 3. activate your virtualenv (if you are using one, if not you can skip this step; using venv is advised)
    
 4. install wheel and cmake:

    
    pip3 install wheel cmake --no-cache-dir

    
 5. install requirements:

    
    pip3 install -r requirements.txt --no-cache-dir


 6. run the following commands to install improve_ai python package:

    
    python3 setup.py build_ext --inplace && python3 setup.py install

## Import and initialize the SDK.

```python
from improveai import DecisionTracker

tracker_url = '<desired tracker endpoint url>'
history_id = '<desired history id>'

# Call with no API key
tracker = DecisionTracker(track_url=tracker_url, history_id=history_id)

# Call with API key
api_key = '<API key for endpoint_url>'
tracker = \
    DecisionTracker(
        track_url=tracker_url, api_key=api_key, history_id=history_id)
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
from improveai import DecisionModel

 
model_name = '<desired model name>'
model_url = '<path / or / URL / to / model>'

dm = DecisionModel(model_name=model_name).load(model_url=model_url)
```

### Hello World!

What is the best greeting?

```python
from improveai import Decision, DecisionModel


model_name = 'hello-world-model'
dm = DecisionModel(model_name=model_name) 

# If you already have a trained model you might want to use one 
i_have_model = False
if i_have_model:
    model_url = '<path / or / URL / to / model>'
    dm.load(model_url=model_url)

# prepare JSON encodable variants to choose from:
hello_variants = [
    {'text': "Hello World!"},
    {'text': "Hi World!"},
    {'text': "Howdy World!"}]

givens = {
    '<metadata_key_0>': '<metadata_value_0>',
    '<metadata_key_1>': '<metadata_value_1>'}

# Get the best greeting
best_hello_world = \
    Decision(decision_model=dm).choose_from(variants=hello_variants)\
    .given(givens=givens).get()

# Train model using decision
tracker.track(
    variant=best_hello_world,
    variants=hello_variants,
    givens=givens, model_name=model_name,
    timestamp='<datetime64 timestamp or None>')

```

Improve quickly learns to choose the greeting with the highest chance of button tap.

```'greeting'``` in this example is the namespace for the type of variant being chosen. Namespaces ensure that multiple uses of Improve in the same project are decided and trained separately. 
A namespace can be a simple string like "discount" or "song" or can be more complicated like "SubscriptionViewController.buttonText". 
Namespace strings are opaque and can be any format you wish.


### Numbers Too

How many bonus gems should we offer on our In App Purchase?

```python
from improveai import Decision, DecisionModel


model_name = 'gems-model'
dm = DecisionModel(model_name=model_name)

# prepare JSON encodable variants to choose from:
gems_variants = [{'number': 1000}, {'number': 2000}, {'number': 3000}]

givens = {
    '<metadata_key_0>': '<metadata_value_0>',
    '<metadata_key_1>': '<metadata_value_1>'}

best_gems_count = \
    Decision(decision_model=dm).choose_from(variants=gems_variants)\
    .givens(givens=givens).get()

# train the model using the decision
tracker.track(
    variant=best_gems_count,
    variants=gems_variants,
    givens=givens, model_name=model_name,
    timestamp='<datetime64 timestamp or None>')


```

### Complex Objects

```python
from improveai import Decision, DecisionModel


dm = None
# If you already have a trained model you might want to use one
model_name = 'complex-objects-model'
i_have_model = False
if i_have_model:
    model_ulr = '<path / or / URL / to / model>'
    
    dm = DecisionModel(model_name=model_name).load(model_url=model_ulr)

# prepare JSON encodable variants to choose from:
complex_variants = [
    {"textColor": "#000000", "backgroundColor": "#ffffff" },
    { "textColor": "#F0F0F0", "backgroundColor": "#aaaaaa" }]

best_complex_variant = \
    Decision(decision_model=dm).choose_from(variants=complex_variants).get()

```

Improve learns to use the attributes of each key and value in a dictionary variant to make the optimal decision.  

Variants can be any JSON encodeable object of arbitrary complexity.

### Howdy World (Context for Cowboys)

If language is "cowboy", which greeting is best?

```python
from improveai import Decision, DecisionModel
 

dm = None
# If you already have a trained model you might want to use one
model_name = 'greeting-model'

i_have_model = False
if i_have_model:
    model_url = '<path / or / URL / to / model>'
    
    dm = DecisionModel(model_name=model_name).load(model_url=model_url)

# prepare JSON encodable variants to choose from:

cowboy_hello_variants = [
    {"text": "Hello World!"},
    {"text": "Hi World!"},
    {"text": "Howdy World!"}]

cowboy_greeting_context = {"language": "cowboy"}

best_complex_variant = \ 
    Decision(decision_model=dm).choose_from(variants=cowboy_hello_variants)\
    .get()

```

Improve can optimize decisions for a given ```givens``` of arbitrary complexity. We might imagine that "Howdy World!" would produce the highest rewards for { language: cowboy }, while another greeting might be best for other contexts.

You can think of contexts like: If ```<givens>``` then ```<variant>```.

### Learning from Specific Types of Rewards
Instead of tracking rewards for every separate decision namespace, we can assign a custom rewardKey during track for that specific decision to be trained on.

```python
variants = \
    [{'song': "Hey Jude"}, {'song': "Lucy in the sky with diamond"}, 
     {'song': "Yellow submarine"}]
tracked_variant = {'song': "Hey Jude"}
tracked_variant_givens = {}  # some context for a given variant

# train the model using the decision
tracker.track(
    variant=tracked_variant,
    variants=variants,
    givens=tracked_variant_givens, model_name=model_name,
    timestamp='<datetime64 timestamp or None>')

# track_rewards() is called in iOS API here but is not implemented
 ```

 ### Learning Rewards for a Specific Variant
 
 Instead of applying rewards to general categories of decisions, they can be scoped to specific variants by specifying a custom rewardKey for each variant.


```python
from improveai import Decision, DecisionModel
 

dm = None
# If you already have a trained model you might want to use one

model_name = 'videos-model'
dm = DecisionModel(model_name=model_name)

i_have_model = False
if i_have_model:
    model_url = '<path / or / URL / to / model>'
    dm.load(model_url=model_url)

viral_video_variants = [{'video': 'video1'}, {'video': 'video2'}]
viral_video_givens = {}  # context for videos

best_video = \
    Decision(decision_model=dm).choose_from(variants=viral_video_variants)\
    .given(givens=viral_video_givens).get()

# Create a custom rewardKey specific to this variant
tracker.track(
    variant=best_video,
    variants=viral_video_variants,
    givens=viral_video_givens, model_name=model_name,
    timestamp='<datetime64 timestamp or None>')
```

[comment]: <> (```python)

[comment]: <> (video_reward_key = 'sample_video_key')

[comment]: <> (best_video_revenue = 1  # award / revenue assigned to the 'best' variant)

[comment]: <> (tracker.add_reward&#40;)

[comment]: <> (    reward=best_video_revenue, reward_key=video_reward_key, )

[comment]: <> (    message_id='<unique msg id>', history_id='<unique history id>', )

[comment]: <> (    timestamp='<datetime64 timestamp or None>'&#41;)

[comment]: <> (```)


 ### Server-Side Decision/Rewards Processing
 
 Some deployments may wish to handle all training and reward assignments on the server side. In this case, you may simply track generic app events to be parsed by your custom backend scripts and converted to decisions and rewards.
 
 ```python
# omit trackDecision and trackReward on the client and use custom code on the model gateway to do it instead

#...when the song is played
song_properties = {'song': 'example_song_object'}
tracker.track_event(event_name='song_played', properties=song_properties)
```

 ## Algorithm
 
The algorithm is a production tuned contextual multi-armed bandit algorithm related to Thompson Sampling.
 
 ## Security & Privacy
 
 Improve uses tracked variants, context, and rewards to continuously train statistical models.  If models will be distributed to unsecured clients, then the most conservative stance is to assume that what you put in the model you can get out.
 
 That said, all variant and context data is hashed (using a secure pseudorandom function once siphash is deployed) and never transmitted in models so if a sensitive information were accidentally included in tracked data, it is not exposed by the model.
 
It is strongly recommended to never include Personally Identifiable Information (PII) in an Improve variant or context if for no other reason than to ensure that it is not persisted in your Improve Model Gateway analytics records.
 
 The types of information that can be gleaned from an Improve model are the types of things it is designed for, such as the relative ranking and scoring of variants and contexts, and the relative frequency of variants and contexts.  Future versions will normalize rewards to zero, so absolute information about rewards will not be transmitted at that time.
 
 Additional security measures such as white-listing specific variants and context or obfuscating rewards can be implemented by custom scripts on the back end.
 
 For truly sensitive model information, you may wish to only use those Improve models within a secure server environment.
 
 ## Additional Caveats
 
 Use of rapidly changing data in variants and contexts is discouraged.  This includes timestamps, counters, random numbers, message ids, or unique identifiers.  These will be treated as statistical noise and may slow down model learning or performance.  If models become bloated you may filter such nuisance data on the server side during training time.
 
 Numbers with limited range, such as ratios, are okay as long as they are encoded as NSNumbers.
 
 In addition to the previous noise issue, linear time based data is generally discouraged because decisions will always being made in a time interval ahead of the training data.  If time based context must be used then ensure that it is cyclical such as the day of the week or hour of the day without reference to an absolute time.

## License

Improve.ai is copyright Mind Blown Apps, LLC. All rights reserved.  May not be used without a license.


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