# cython: language_level=3, boundscheck=False, wraparound=False
from collections.abc import Generator, Mapping, Sequence

from ..core.labels import repr_pretty
from .edges import EdgeContainer
from .exception import (
    AllocationError,
    CalculationError,
    UnclosedGraphError,
)

from .input cimport Input
from .output_methods import _OutputMethods


cdef class Output:
    # cdef public fields declared in output.pxd.

    def __init__(self, *args, **kwargs):
        _OutputMethods.__init__(self, *args, **kwargs)

    def __str__(self):
        return _OutputMethods.__str__(self)

    def __rshift__(self, other):
        return _OutputMethods.__rshift__(self, other)

    _repr_pretty_ = _OutputMethods._repr_pretty_

    # Cold non-dunder API via class-attribute binding.
    to_dict = _OutputMethods.to_dict
    data_dense = _OutputMethods.data_dense
    data_sparse = _OutputMethods.data_sparse
    _set_data = _OutputMethods._set_data
    allocate = _OutputMethods.allocate
    _connect_to_input = _OutputMethods._connect_to_input
    _connect_to_node = _OutputMethods._connect_to_node
    # Reached only from Python (set/seti via the Parameter setter), so
    # compiling these as cpdef would give no dispatch speedup.
    set = _OutputMethods.set
    seti = _OutputMethods.seti
    _taint_children_implementation = _OutputMethods._taint_children_implementation
    taint_children_type = _OutputMethods.taint_children_type
    connected = _OutputMethods.connected

    #
    # Properties - kept inline (not in _OutputMethods) so Cython compiles
    # them into the tp_getset slot instead of a Python descriptor.
    #

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def allocatable(self):
        return self._allocatable

    @property
    def has_data(self):
        return self._data is not None

    @property
    def node(self):
        return self._node

    @property
    def labels(self):
        return self._labels if self._labels is not None else self._node and self._node.labels

    @labels.setter
    def labels(self, labels):
        self._labels = labels

    @property
    def child_inputs(self):
        return self._child_inputs

    @property
    def parent_input(self):
        return self._parent_input

    @parent_input.setter
    def parent_input(self, input):
        self._parent_input = input

    @property
    def logger(self):
        return self._node.logger

    @property
    def invalid(self):
        return self._node.invalid

    @invalid.setter
    def invalid(self, invalid):
        for input in self._child_inputs:
            input.invalid = invalid

    @property
    def data(self):
        if not self.closed:
            raise UnclosedGraphError(
                "Unable to get the output data from unclosed graph!",
                node=self._node,
                output=self,
            )
        try:
            self.touch()
            return self._data_ro
        except CalculationError as exc:
            raise CalculationError(
                "An exception occured while the node was touched!",
                node=self._node,
                output=self,
                args=exc.args,
            ) from exc

    @property
    def dd(self):
        return self._dd

    @property
    def owns_buffer(self):
        return self._owns_buffer

    @property
    def forbid_reallocation(self):
        return self._forbid_reallocation

    @property
    def closed(self):
        return self._node.closed if self._node else False

    @property
    def tainted(self):
        return self._node.tainted

    @property
    def debug(self):
        return self._debug

    #
    # Hot path
    #

    cpdef touch(self, bint force_computation=False):
        return self._node.touch(force_computation=force_computation)

    cpdef void touch_above_threshold(self, int taint_threshold, bint force_computation=False):
        self._node.touch_above_threshold(
            taint_threshold, force_computation=force_computation
        )

    cpdef void taint_children(self, bint force_taint=False, bint force_computation=False, object caller=None):
        cdef Input inp
        for inp in self._child_inputs:
            inp.taint(force_taint, force_computation, caller)

    cpdef void taint_children_below_threshold(
        self,
        int taint_threshold = 0,
        bint force_taint = False,
        bint force_computation = False,
        object caller = None,
    ):
        cdef Input child_input
        for child_input in self._child_inputs:
            child_input.taint_below_threshold(
                taint_threshold,
                force_taint=force_taint,
                force_computation=force_computation,
                caller=caller,
            )


class Outputs(EdgeContainer):
    __slots__ = ()

    def __init__(self, iterable=None):
        super().__init__(iterable)
        self._dtype = Output

    def __str__(self):
        return f"○[{tuple(obj.name for obj in self)}]→"

    _repr_pretty_ = repr_pretty
