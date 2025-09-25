from collections.abc import Iterable

def IsIterable(obj):
    return isinstance(obj, Iterable) and not isinstance(obj, str)
