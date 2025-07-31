from collections.abc import Iterable

class StopNesting(Exception):
    def __init__(self, object):
        self.object = object

def IsIterable(obj):
    return isinstance(obj, Iterable) and not isinstance(obj, str)
