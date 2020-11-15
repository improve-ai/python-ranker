#!python
#cython: language_level=3

# print('Hello World!')

import numpy as np
cimport numpy as np

from utils.cython_experiments.external_cython_dep import ExampleDep

def sample_array_oper(object[:] a, long[:] out=None):

    # edc = ExampleDep()
    # loop_in = edc.some_example_dep(w=a)

    # str example
    # cdef

    cdef res = np.empty_like(a).astype(str)

    for el_ixd in range(len(a)):
        res[el_ixd] = 'f' + str(a[el_ixd])

    return res
