from .copy import Copy


def _copy_class(cls: type, name: str, **attributes) -> type:
    """A copy of `cls` with other `attributes`, not a subclass: its methods can't use super()"""
    # the slot descriptors of cls would conflict with the new ones
    namespace = {key: value for key, value in vars(cls).items() if key not in cls.__slots__}
    return type(cls)(name, cls.__bases__, namespace | attributes)


Cache = _copy_class(Copy, "Cache", __module__=__name__, _cache_default=True)
