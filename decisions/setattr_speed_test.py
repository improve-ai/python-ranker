import inspect
import time
from warnings import warn


class WithCustomSetattr(object):

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.example_list = [el for el in range(1000)]

    def __setattr__(self, key, value):
        caller_class_details = \
            inspect.stack()[1][0].f_locals  # ['self'].__class__.__name__
        # print('__setattr__ call')
        # pprint(caller_class_details)
        # TODO finish this up

        if 'self' not in caller_class_details.keys():
            warn('Setting from outside of Decision class is not allowed')
            return

        if not id(caller_class_details['self']) == id(self):
            warn('Setting from outside of Decision class is not allowed')
            return

        super(WithCustomSetattr, self).__setattr__(key, value)

    def get_gen(self, size=100):
        return (el for el in range(size))

    def get_iter(self, size=100):
        return iter(el for el in range(size))

    def get_list(self, size=100):
        return list(el for el in range(size))

    def time_ret_list(self):
        return self.example_list

    def time_ret_gen(self):
        return (el for el in self.example_list)


class NoCustomSetattr:
    def __init__(self, a, b):
        self.a = a
        self.b = b


if __name__ == '__main__':

    # start_time = time.time()
    # for _ in range(100):
    #     q = WithCustomSetattr(100, 'ssasasfsdf')
    #
    # print((time.time() - start_time) / 100)
    #
    # start_time = time.time()
    # for _ in range(100):
    #     q = NoCustomSetattr(100, 'ssasasfsdf')
    #
    # print((time.time() - start_time) / 100)

    # q = WithCustomSetattr(100, 'ssasasfsdf')
    #
    # p = q.get_gen(size=1000)
    #
    # del q
    #
    # for el in list(p)[:10]:
    #     print(el)

    # gen_time = 0
    # iter_time = 0
    #
    # for _ in range(1000):
    #     start_time = time.time()
    #     for _ in range(10000):
    #         p = q.time_ret_list()
    #     gen_time += time.time() - start_time
    #     # print((time.time() - start_time) / 100)
    #     # print(type(p))
    #
    # # q = WithCustomSetattr(100, 'ssasasfsdf')
    # # start_time = time.time()
    # # for _ in range(10000):
    # #     p = q.get_list(10000)
    # # print((time.time() - start_time) / 100)
    # # print(type(p))
    #
    # for _ in range(1000):
    #     start_time = time.time()
    #     for _ in range(10000):
    #         p = q.time_ret_gen()
    #     iter_time += time.time() - start_time
    #     # print((time.time() - start_time) / 100)
    #     # print(type(p))
    #
    # print(gen_time / 1000)
    # print(iter_time / 1000)

    import numpy as np

    a = [{'test': 'value'} for el in range(10)]
    b = [el for el in range(10)]

    create_from_list = 0
    create_from_full_and_fill = 0

    for _ in range(1000):
        start_time = time.time()
        for _ in range(100):
            p = np.array([a, b])
        create_from_list += time.time() - start_time

    for _ in range(1000):
        start_time = time.time()
        for _ in range(100):
            w = np.full((2, len(a)), '')
            w[0] = a
            w[1] = b
            # ww = w.T
        create_from_full_and_fill += time.time() - start_time

    print(create_from_list / 1000)
    print(create_from_full_and_fill / 1000)
