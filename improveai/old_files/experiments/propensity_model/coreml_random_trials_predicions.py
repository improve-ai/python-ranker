"""
Calculates predictions for randomly generated trials using the model.
Creates 'trials.json' output which may be used for testing.

Args:
1) .mlmodel model path
2) path to .py file where the FeatureEncoder class is defined

python3
"""

import coremltools as cml
import os
import sys
import json
import importlib
import random
import string
import numpy as np


def random_key():
    length = random.randint(2, 5)
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def random_variant():
    simple = random.random() < 0.8
    if simple:
        if random.random() < 0.9:
            type = random.choice(['numb', 'str'])
        else:
            type = 'null'
    else:
        type = random.choice(['array', 'dict'])

    if type == 'numb':
        return random.uniform(100, 1000000)
    elif type == 'str':
        return random_key()
    elif type == 'null':
        return None
    elif type == 'array':
        n = random.randint(2, 5)
        return [random_variant() for i in range(n)]
    elif type == 'dict':
        n = random.randint(1, 4)
        return dict([(random_key(), random_variant()) for i in range(n)])


# Apply this transform to CoreML prediction to get "binary:logistic" output similar to XGBoost
def sigmfix(x):
  return 1 / (1 + np.exp(0.5 - x))


trials = [random_variant() for i in range(40)]
context = {
    "device_manufacturer":"Apple","version_name":"4.3","os_name":"ios","share_ratio":0.05000000074505806,"day":80,"shared":{"This moment is a fresh beginning.":1,"You will die some day. How are you living today?":1,"Accept this moment as it is and you will find peace.":1,"Be the witness to the story, rather than the actor in it.":1},"language":"English","page":80,"os_version":"12.1.2","country":"United States","carrier":"AT&T","device_model":"iPhone 7 Plus"
}

model = cml.models.MLModel(sys.argv[1])
metadata = json.loads(model.user_defined_metadata["json"])
lookup_table = metadata['table']
n_features = len(lookup_table[1])
feature_name_prefix = "f"

encoder_folder, encoder_filename = os.path.split(sys.argv[2])
sys.path.append(os.path.abspath(encoder_folder))
FeatureEncoder = importlib.import_module(os.path.splitext(encoder_filename)[0]).FeatureEncoder
encoder = FeatureEncoder(lookup_table, metadata["model_seed"])
encoded_context = encoder.encode_features({"context": context})

predictions = []
for trial in trials:
    encoded_trial = encoder.encode_features({"variant": trial}, encoded_context)

    # Ensure that all features are presented to prevent errors by filling missing
    # features with 'nan'.
    # Make names to be the same as the model feture names (f0, f1, etc).
    features = {feature_name_prefix + str(i): float('nan') for i in range(n_features) }
    for intKey, val in encoded_trial.items():
        features[feature_name_prefix + str(intKey)] = val

    print(features)
    prediction = sigmfix(model.predict(features)["target"])
    print(prediction)
    predictions.append(prediction)

output = {"trials": trials, "context": context, "predictions": predictions}
with open('trials.json', 'w') as outfile:
    json.dump(output, outfile, indent=2)

