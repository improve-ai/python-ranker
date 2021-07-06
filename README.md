# Improve.ai for python (3.7.9)

## An AI library for Making Great Choices

[![License](https://img.shields.io/cocoapods/l/Improve.svg?style=flat)](http://cocoapods.org/pods/Improve)

Quickly choose and sort objects to maximize user retention, performance, revenue, or any other metric. It's like an AI if / then statement.

Improve.ai performs fast machine learning on any `JSON` encodable data structure including dictionaries, arrays, strings, integers, floats, and null values.

## Installation

Python-SDK for Improve.ai is currently available only on github.

o install python-sdk for Improve.ai:

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

endpoint_url = '<desired tracker endpoint url>'
history_id = '<desired history id>'

# Call with no API key
tracker = DecisionTracker(track_url=endpoint_url, history_id=history_id)

# Call with API key
api_key = '<API key for endpoint_url>'
tracker = \
        DecisionTracker(
            track_url=endpoint_url, api_key=api_key, history_id=history_id)
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


dm = None
# If you already have a trained model you might want to use one 
i_have_model = False
if i_have_model:
    model_url = '<path / or / URL / to / model>'
    
    dm = DecisionModel(model_url=model_url)

# prepare JSON encodable variants to choose from:
hello_variants = [
    {'text': "Hello World!"},
    {'text': "Hi World!"},
    {'text': "Howdy World!"}]

d = Decision(model=dm)

# Get the best greeting
best_hello_world = d.best()

# Train model using decision
tracker.track(decision=d, timestamp='<datetime64 timestamp or None>')

best_hello_world_revenue = 1  # award / revenue assigned to the 'best' variant

# ... later when the `best_hello_world` is chosen, give the decision a reward
tracker.add_reward(
    reward=best_hello_world_revenue, reward_key='greetings', 
    message_id='<unique msg id>', history_id='<unique history id>', 
    timestamp='<datetime64 timestamp or None>')
```

Improve quickly learns to choose the greeting with the highest chance of button tap.

```'greeting'``` in this example is the namespace for the type of variant being chosen. Namespaces ensure that multiple uses of Improve in the same project are decided and trained separately. 
A namespace can be a simple string like "discount" or "song" or can be more complicated like "SubscriptionViewController.buttonText". 
Namespace strings are opaque and can be any format you wish.


### Numbers Too

How many bonus gems should we offer on our In App Purchase?

```python
from improveai import Decision


# prepare JSON encodable variants to choose from:
gems_variants = [{'number': 1000}, {'number': 2000}, {'number': 3000}]

d = Decision(variants=gems_variants, model_name='bonusGems')

best_gems_count = d.best()

# train the model using the decision
tracker.track_using_best_from(
    decision=d, message_id='<unique msg id>', history_id='<unique history id>', 
    timestamp='<datetime64 timestamp or None>')

# ... later when the user makes a purchase, give the decision a reward
best_gems_count_revenue = 1  # award / revenue assigned to the 'best' variant
tracker.add_reward(
    reward=best_gems_count_revenue, reward_key='bonusGems', 
    message_id='<unique msg id>', history_id='<unique history id>', 
    timestamp='<datetime64 timestamp or None>')
```

### Complex Objects

```python
from improveai import Decision, DecisionModel


dm = None
# If you already have a trained model you might want to use one
i_have_model = False
if i_have_model:
    model_kind = 'xgb_native'
    model_pth = '<path / or / URL / to / model>'
    
    dm = DecisionModel(model_pth=model_pth, model_kind=model_kind)

# prepare JSON encodable variants to choose from:
complex_variants = [
    {"textColor": "#000000", "backgroundColor": "#ffffff" },
    { "textColor": "#F0F0F0", "backgroundColor": "#aaaaaa" }]

d = Decision(variants=complex_variants, model=dm, model_name='theme')

bset_coplex_variant = d.best()
```

Improve learns to use the attributes of each key and value in a dictionary variant to make the optimal decision.  

Variants can be any JSON encodeable object of arbitrary complexity.

### Howdy World (Context for Cowboys)

If language is "cowboy", which greeting is best?

```python
from improveai import Decision, DecisionModel
 

dm = None
# If you already have a trained model you might want to use one
i_have_model = False
if i_have_model:
    model_kind = 'xgb_native'
    model_pth = '<path / or / URL / to / model>'
    
    dm = DecisionModel(model_pth=model_pth, model_kind=model_kind)

# prepare JSON encodable variants to choose from:

cowboy_hello_variants = [
    {"text": "Hello World!"},
    {"text": "Hi World!"},
    {"text": "Howdy World!"}]

cowboy_greeting_context = {"language": "cowboy"}

d = Decision(
    variants=complex_variants, model=dm, model_name='greetings', 
    context=cowboy_greeting_context)

bset_coplex_variant = d.best()
```

Improve can optimize decisions for a given context of arbitrary complexity. We might imagine that "Howdy World!" would produce the highest rewards for { language: cowboy }, while another greeting might be best for other contexts.

You can think of contexts like: If `<context>` then `<variant>`.

### Learning from Specific Types of Rewards
Instead of tracking rewards for every seperate decision namespace, we can assign a custom rewardKey during trackDecision for that specific decision to be trained on.

```python
tracked_variant = {'song': "Hey Jude"}
tracked_variant_context = {}  # some context for a given variant

# train the model using the decision
tracker.track_using(
    variant=tracked_variant, model_name='songs', message_id='<unique msg id>', 
    history_id='<unique history id>', timestamp='<datetime64 timestamp or None>')

# track_rewards() is called in iOS API here but is not implemented
 ```

 ### Learning Rewards for a Specific Variant
 
 Instead of applying rewards to general categories of decisions, they can be scoped to specific variants by specifying a custom rewardKey for each variant.


```python
from improveai import Decision, DecisionModel
 

dm = None
# If you already have a trained model you might want to use one
i_have_model = False
if i_have_model:
    model_kind = 'xgb_native'
    model_pth = '<path / or / URL / to / model>'
    
    dm = DecisionModel(model_pth=model_pth, model_kind=model_kind)


viral_video_variants = [{'video': 'video1'}, {'video': 'video2'}]
viral_video_context = {}  # context for videos

d = Decision(
    variants=viral_video_variants, model=dm, context=viral_video_context)

best_video = d.best()

# Create a custom rewardKey specific to this variant
tracker.track_using_best_from(
    decision=d, message_id='<unique msg id>', history_id='<unique history id>',
    timestamp='<datetime64 timestamp or None>')

video_reward_key = 'sample_video_key'
best_video_revenue = 1  # award / revenue assigned to the 'best' variant
tracker.add_reward(
    reward=best_video_revenue, reward_key=video_reward_key, 
    message_id='<unique msg id>', history_id='<unique history id>', 
    timestamp='<datetime64 timestamp or None>')
```

 ### Sort Stuff

```python
from improveai import Decision, DecisionModel
 

dm = None
# If you already have a trained model you might want to use one
i_have_model = False
if i_have_model:
    model_kind = 'xgb_native'
    model_pth = '<path / or / URL / to / model>'
    
    dm = DecisionModel(model_pth=model_pth, model_kind=model_kind)

dogs_variants = [
    {'breed': 'German Shepard'}, 
    {'breed': "Border Collie"}, 
    {'breed': "Labrador Retriever"}]
dogs_context = {}  # dogs context dict
dogs_model_name = 'dog'

d = Decision(
    variants=dogs_variants, model=dm, model_name=dogs_model_name, 
    context=dogs_context)

# No human could ever make this decision, but math can.
sorted_breeds = d.ranked()
top_dog = sorted_breeds[0] 

# With ranked, training is done just as before, on one individual variant at a time.
tracker.track_using_best_from(
    decision=d, message_id='<unique msg id>', history_id='<unique history id>',
    timestamp='<datetime64 timestamp or None>')

top_dog_revenue = 1000  # award / revenue assigned to the 'best' variant
tracker.add_reward(
    reward=top_dog_revenue, reward_key=dogs_model_name, 
    message_id='<unique msg id>', history_id='<unique history id>', 
    timestamp='<datetime64 timestamp or None>')

```

 ### Server-Side Decision/Rewards Processing
 
 Some deployments may wish to handle all training and reward assignments on the server side. In this case, you may simply track generic app events to be parsed by your custom backend scripts and converted to decisions and rewards.
 
 ```python
# omit trackDecision and trackReward on the client and use custom code on the model gateway to do it instead

#...when the song is played
song_properties = {'song': 'example_song_object'}
tracker.track_analytics_event(event='Song Played', properties=song_properties)
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