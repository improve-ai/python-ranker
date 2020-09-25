def constant(f):
    def fset(self, value):
        raise AttributeError

    def fget(self):
        return f()

    return property(fget, fset)