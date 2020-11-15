%%cython
cimport numpy as np
import numba as nb
import numpy as np


@nb.jit()
def fast_string_appender():
    a = np.arange(1000).astype(str)
    f = np.array('f', dtype=str)

    for el in a:
        a[el] = f + a

    return np.arange(10)


def do_list_comp():
    return ['f{}'.format(el) for el in range(1000)]


if __name__ == '__main__':

    fast_string_appender()

    do_list_comp()
