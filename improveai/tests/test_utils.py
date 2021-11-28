from numbers import Number
import numpy as np


def convert_values_to_float32(val: object):
    if isinstance(val, float):
        # return float(str(np.float32(val)))
        return np.float32(val)

    if isinstance(val, dict):
        conv_val = {}
        for k, v in val.items():
            conv_val[k] = convert_values_to_float32(v)

        return conv_val

    if isinstance(val, list):
        return [convert_values_to_float32(el) for el in val]

    if isinstance(val, np.ndarray):
        return np.array([convert_values_to_float32(el) for el in val])

    return val


def assert_dicts_identical(expected, calculated):
    for expected_key in expected.keys():
        expected_value = expected[expected_key]
        calculated_value = calculated[expected_key]
        assert expected_value == calculated_value
