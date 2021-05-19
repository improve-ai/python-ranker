# V6 FeatureEncoder test suite description.

### Test Case

All v6 FeatureEncoder test cases are placed in a `test_suite` folder.
Each test case is a JSON file which looks as follows:

```python

{
    "test_case": {
        "variant": -2147483647
    },
    "test_output": {
        "5e9f9c59": -2147498275.6207075
    },
    "model_seed": 1,
    "noise": 0.8928601514360016,
    "variant_seed": 2675988294294598568,
    "value_seed": 6818340268807889528,
    "context_seed": 5164679660109946987
}

```

`"test_case"` entry of a test case JSON always contains tested input (sometimes it might have only
`"variant"` key, but also may have both - `"variant"` and `"context"`)


`"test_output"` entry always contains the result of complete encoding of the input (encoding of a 
`"context"` + encoding of the `"variant"`). If there are any collisions in dictionaries obtained by 
encoding `"variant"` and `"context"` values stored under colliding keys should be added, e.g.:

```python

example_complete_test_case = {
   "test_case": {"variant": {...}, "givens": {...}},
   "test_output": {...},
   "model_seed": ...,
   "noise": ...,
   "variant_seed": ...,
   "value_seed": ...,
   "context_seed": ...
}
# example_complete_test_case.get("model_seed") 
# will fetch value for "model_seed" key from example_complete_test_case 
fe = FeatureEncoder(model_seed=example_complete_test_case.get("model_seed"))

test_input = example_complete_test_case.get("test_case")

noise = example_complete_test_case.get("noise")

encoded_variant = fe.encode_variant(variant=test_input.get("variant"),
                                    noise=noise)
# lets assume encoded_variant = {'a': 1, 'b': 2}

encoded_context = fe.encode_context(context=test_input.get("givens"),
                                    noise=noise)
# lets assume encoded_context = {'b': 1, 'c': 2}

# in such case the following code:
fully_encoded_variant = {}
for k in set(encoded_context) | set(encoded_variant):
   fully_encoded_variant[k] =
   encoded_context.get(k, 0) + encoded_variant.get(k, 0)
# should result in completely encoded test case
# fully_encoded_variant = \ 
#   {'a': 1, #  'a' comes from encoded_variant
#    'b': 2 + 1,  #  'b' is a sum of values for key 'b' from both encoded_variant and encoded_context
#    'c': 2 #  'c' comes from encoded_context
#    }


```

Attributes: `variant_seed`, `value_seed` and `context_seed` are placed in a test case for verification only.
When a FeatureEncoder's constructor is called with a proper seed (in this case `model_seed`=1) values from
a test case JSON should match those calculated on FeatureEncoder's instantiation.

### Remarks

It is important to note that (for a fixed `model_seed` and `noise` value):
  - complete encoding of ```{"test_case": {"variant": <primitive / non-dict type>}}``` and 
   with  ```{"test_case": {"variant": {"$value": <primitive / non-dict type>}}}``` should yield
   identical results
 - context encoding method (encode_context() for python) should raise TypeError on an attempt to 
   encode non-dict type
 - 0, 0.0 and false should all yield identical encoding results
 - 1, 1.0, and true should all yield identical encoding results
 - empty list, empty dict / map and null / NaN should all encode to empty dict / map
 - `dict_foo_bar.json` test case might be tricky - I had problems loading json from a 
   string with a single escape character (`\`) so I hardcoded this test case into tests 

### Python specific test cases

There are python-specific test cases placed in `python_specific` folder. Those are test cases
for a batch encoding and missing features filling method methods. Maybe they'll be of use to you. 
