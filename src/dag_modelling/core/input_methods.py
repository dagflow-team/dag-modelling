from __future__ import annotations

from ..core.labels import repr_pretty
from .data_descriptor import DataDescriptor, Layout
from .exception import AllocationError, ClosedGraphError, InitializationError, ReconnectionError


class _InputMethods:
    __slots__ = ()

    def __init__(
        self,
        name=None,
        node=None,
        *,
        child_output=None,
        parent_output=None,
        debug=None,
        allocatable=False,
        data=None,
        dtype=None,
        shape=None,
        axes_edges=None,
        axes_meshes=None,
    ):
        if data is not None and (allocatable or dtype is not None or shape is not None):
            raise InitializationError(input=self, node=node)

        self._name = name
        self._node = node
        self._allocatable = allocatable
        self._parent_output = parent_output

        self._child_output = None
        if child_output:
            self._set_child_output(child_output)

        if debug is not None:
            self._debug = debug
        elif node:
            self._debug = node.debug
        else:
            self._debug = False

        self._owns_buffer = False
        self._own_data = None
        self._own_dd = DataDescriptor(dtype, shape, axes_edges, axes_meshes)
        if data is not None:
            self.set_own_data(data, owns_buffer=True)

    def __str__(self):
        return self.connected() and f"→● {self._name}" or f"→○ {self._name}"

    _repr_pretty_ = repr_pretty

    #
    # Properties
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
        """Checks validity of the parent output data."""
        return self._parent_output.invalid

    @invalid.setter
    def invalid(self, invalid):
        """Sets the validity of the current node."""
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
    def data_dense(self):
        return self._parent_output.data_dense

    @property
    def data_sparse(self):
        return self._parent_output.data_sparse

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
    # Data setters
    #

    def set_own_data(
        self,
        data,
        *,
        owns_buffer=False,
        axes_edges=None,
        axes_meshes=None,
    ):
        if self.closed:
            raise ClosedGraphError("Unable to set input data.", node=self._node, input=self)
        if self._own_data is not None:
            raise AllocationError("Input already has data.", node=self._node, input=self)

        self._own_data = data
        self._owns_buffer = owns_buffer
        self._own_dd.dtype = data.dtype
        self._own_dd.shape = data.shape
        self._own_dd.layout = getattr(data, "layout", Layout.DENSE.value)
        self._own_dd.capacity = getattr(data, "capacity", None)
        self._own_dd.axes_edges = axes_edges or ()
        self._own_dd.axes_meshes = axes_meshes or ()

    def set_child_output(self, child_output, force_taint=False):
        if not self.closed:
            return self._set_child_output(child_output, force_taint)
        raise ClosedGraphError(input=self, node=self._node, output=child_output)

    def _set_child_output(self, child_output, force_taint=False):
        if self._child_output and not force_taint:
            raise ReconnectionError(output=self._child_output, node=self._node)
        self._child_output = child_output
        if child_output:
            child_output.parent_input = self

    def set_parent_output(self, parent_output, force_taint=False):
        if self.closed:
            raise ClosedGraphError(input=self, node=self._node, output=parent_output)
        return self._set_parent_output(parent_output, force_taint)

    def _set_parent_output(self, parent_output, force_taint=False):
        if self.connected() and not force_taint:
            raise ReconnectionError(output=self._parent_output, node=self._node)
        self._parent_output = parent_output

    #
    # Evaluation / allocation
    #

    def touch(self):
        return self._parent_output.touch()

    def taint_type(self, force_taint: bool = False) -> None:
        self._node.taint_type(force_taint=force_taint)

    def connected(self) -> bool:
        return self._parent_output is not None

    def allocate(self, **kwargs):
        """Returns True if data was reassigned."""
        if not self._allocatable or (
            self._own_data is not None and self._own_dd.consistent_with(self._own_data)
        ):
            return False

        if self._own_dd.shape is None or self._own_dd.dtype is None:
            raise AllocationError(
                "No shape/type information provided for the Input",
                node=self._node,
                output=self,
            )
        try:
            self._own_data = self._own_dd.allocate(**kwargs)
        except Exception as exc:
            raise AllocationError(f"Input: {exc.args[0]}", node=self._node, input=self) from exc

        return True

    def to_dict(self, *, label_from="text"):
        try:
            output = self._parent_output
        except AttributeError:
            return {"label": "input", "shape": "?"}
        else:
            shape = output.dd.shape
            return {
                "label": output.labels[label_from],
                "shape": shape[0] if len(shape) == 1 else shape,
            }
