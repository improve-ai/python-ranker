class ExampleDep:
    def __init__(self):
        pass

    def some_example_dep(self, w):
        return [el + 1 for el in w]
