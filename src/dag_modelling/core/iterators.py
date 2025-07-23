from nested_mapping import NestedMapping

from .iter import IsIterable, StopNesting


def get_proper_iterator(obj, methodname, onerror, **kwargs):
    if methodname:
        if method := getattr(obj, methodname, None):
            return method(**kwargs)
    if isinstance(obj, NestedMapping):
        return obj.walkvalues()
    if IsIterable(obj):
        return obj
    raise RuntimeError(
        f"Do not know how to get an iterator for '{onerror}'! {obj=}, {type(obj)=}"
    )


def deep_iterate(obj, methodname, onerror, **kwargs):
    try:
        iterable = get_proper_iterator(obj, methodname, onerror, **kwargs)
        if isinstance(iterable, dict):
            raise StopNesting(iterable)
        for element in iterable:
            yield from deep_iterate(element, methodname, onerror, **kwargs)
    except StopNesting as sn:
        yield sn.object


def iter_inputs(inputs, disconnected_only=False):
    return deep_iterate(
        inputs,
        "deep_iter_inputs",
        "inputs",
        disconnected_only=disconnected_only,
    )


def iter_outputs(outputs):
    return deep_iterate(outputs, "deep_iter_outputs", "outputs")


def iter_child_outputs(inputs):
    return deep_iterate(inputs, "deep_iter_child_outputs", "child_outputs")
