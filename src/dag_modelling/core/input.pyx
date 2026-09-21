# cython: language_level=3, boundscheck=False, wraparound=False
from ..core.labels import repr_pretty
from .input_methods import _InputMethods
from .edges import EdgeContainer


cdef class Input:
    # cdef public fields declared in input.pxd.

    def __init__(self, *args, **kwargs):
        _InputMethods.__init__(self, *args, **kwargs)

    def __str__(self):
        return _InputMethods.__str__(self)

    _repr_pretty_ = _InputMethods._repr_pretty_

    # Cold non-dunder API via class-attribute binding.
    set_own_data = _InputMethods.set_own_data
    set_child_output = _InputMethods.set_child_output
    _set_child_output = _InputMethods._set_child_output
    set_parent_output = _InputMethods.set_parent_output
    _set_parent_output = _InputMethods._set_parent_output
    touch = _InputMethods.touch
    allocate = _InputMethods.allocate
    to_dict = _InputMethods.to_dict
    # Not on the hot path: touched only by graph type propagation / repr.
    taint_type = _InputMethods.taint_type
    connected = _InputMethods.connected

    #
    # Properties - kept inline (not in _InputMethods) so Cython compiles
    # them into the tp_getset slot instead of a Python descriptor.
    #

    @property
    def own_data(self):
        return self._own_data

    @property
    def own_dd(self):
        return self._own_dd

    @property
    def owns_buffer(self):
        return self._owns_buffer

    @property
    def closed(self):
        return self._node.closed if self._node else False

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def node(self):
        return self._node

    @property
    def parent_node(self):
        return self._parent_output.node

    @property
    def logger(self):
        return self._node.logger

    @property
    def child_output(self):
        return self._child_output

    @property
    def invalid(self):
        return self._parent_output.invalid

    @invalid.setter
    def invalid(self, invalid):
        self._node.invalid = invalid

    @property
    def has_data(self):
        return self._own_data is not None

    @property
    def allocatable(self):
        return self._allocatable

    @property
    def debug(self):
        return self._debug

    @property
    def parent_output(self):
        return self._parent_output

    @property
    def data(self):
        return self._parent_output.data

    @property
    def _data(self):
        return self._parent_output._data

    @property
    def dd(self):
        return self._parent_output.dd

    @property
    def tainted(self):
        return self._parent_output.tainted

    #
    # Hot path
    #
    cpdef void touch_above_threshold(self, int taint_threshold):
        self._parent_output.touch_above_threshold(taint_threshold)

    cpdef void taint(self, bint force_taint=False, bint force_computation=False, object caller=None):
        self._node.taint(
            force_taint=force_taint,
            force_computation=force_computation,
            caller=self if caller is None else caller,
        )

    cpdef taint_below_threshold(
        self,
        int taint_threshold = 0,
        bint force_taint = False,
        bint force_computation = False,
        object caller = None
    ):
        self._node.taint_below_threshold(
            taint_threshold,
            force_taint=force_taint,
            force_computation=force_computation,
            caller=self if caller is None else caller,
        )

class Inputs(EdgeContainer):
    __slots__ = ()

    def __init__(self, iterable=None):
        super().__init__(iterable)
        self._dtype = Input

    def __str__(self):
        return f"→[{tuple(obj.name for obj in self)}]○"

    _repr_pretty_ = repr_pretty

    def touch(self):
        for input in self:
            input.touch()
