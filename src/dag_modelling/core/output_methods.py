from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence

from ..core.labels import repr_pretty
from .csr_array import CSRArray
from .data_descriptor import DataDescriptor, Layout
from .exception import (
    AllocationError,
    CalculationError,
    ClosedGraphError,
    ConnectionError,
    DagModellingError,
    InitializationError,
    UnclosedGraphError,
)


class _OutputMethods:
    __slots__ = ()

    def __init__(
        self,
        name,
        node,
        *,
        debug=None,
        allocatable=None,
        data=None,
        owns_buffer=None,
        dtype=None,
        shape=None,
        axes_edges=None,
        axes_meshes=None,
        forbid_reallocation=False,
    ):
        self._labels = None
        self._data = None
        self._data_ro = None
        self._allocating_input = None

        self._name = name
        self._node = node
        self._child_inputs = []
        self._debug = debug if debug is not None else node.debug if node else False
        self._forbid_reallocation = forbid_reallocation

        self._dd = DataDescriptor(dtype, shape, axes_edges, axes_meshes)

        if data is None:
            self._allocatable = True if allocatable is None else allocatable
            self._parent_input = None
            self._owns_buffer = False
        else:
            if owns_buffer is None:
                owns_buffer = True
            self._allocatable = not owns_buffer
            self._set_data(data, owns_buffer=owns_buffer)

            if allocatable or dtype is not None or shape is not None:
                raise InitializationError(output=self, node=node)

    def __str__(self):
        return self.connected() and f"●→ {self._name}" or f"○→ {self._name}"

    _repr_pretty_ = repr_pretty

    #
    # Properties
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
        """Checks the validity of the current node"""
        return self._node.invalid

    @invalid.setter
    def invalid(self, invalid):
        """Sets the validity of the following nodes"""
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
    def data_dense(self):
        """The data as a dense array, a read-only copy for a sparse layout"""
        data = self.data
        if self._dd.layout == Layout.DENSE:
            return data
        dense = data.toarray()
        dense.flags.writeable = False
        return dense

    @property
    def data_sparse(self):
        """The sparse matrix, `None` for the dense layout"""
        return None if self._dd.layout == Layout.DENSE else self.data

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

    def to_dict(self, *, label_from="text"):
        shape = self._dd.shape
        size = self._dd.size
        ret = {
            "label": self.labels[label_from],
            "shape": shape[0] if shape and len(shape) == 1 else shape,
        }

        if size is not None:
            if size > 1:
                ret["value"] = "…"
            elif size == 1:
                try:
                    data = self.data_dense
                except DagModellingError:
                    ret["value"] = "???"
                else:
                    ret["value"] = float(data.ravel()[0])

        return ret

    #
    # Data setters
    #

    def _set_data(
        self,
        data,
        *,
        owns_buffer=False,
        override=False,
        forbid_reallocation=None,
    ):
        if self.closed:
            raise ClosedGraphError("Unable to set output data.", node=self._node, output=self)

        if owns_buffer:
            forbid_reallocation = True
        elif forbid_reallocation is None:
            forbid_reallocation = owns_buffer

        forbid_reallocation |= self._forbid_reallocation
        if forbid_reallocation and self._allocating_input:
            raise AllocationError(
                "Output is connected to allocating input, but reallocation is forbidden",
                node=self._node,
                output=self,
            )

        if not self._dd.consistent_with(data):
            self._node.taint_type()

        self._data = data
        if isinstance(data, CSRArray):
            self._data_ro = data.matrix
        else:
            self._data_ro = data.view()
            self._data_ro.flags.writeable = False

        self._dd.dtype = data.dtype
        self._dd.shape = data.shape
        self._dd.layout = getattr(data, "layout", Layout.DENSE.value)
        self._dd.capacity = getattr(data, "capacity", None)
        self._owns_buffer = owns_buffer
        self._forbid_reallocation = forbid_reallocation

    #
    # Allocation
    #

    def allocate(self, **kwargs):
        """returns True if data was reassigned"""
        if not self._allocatable:
            return False

        if self._allocating_input:
            _input = self._allocating_input
            _input.allocate(allocate_parents=False)
            if _input.has_data:
                idata = _input._own_data
                if not self._dd.consistent_with(idata):
                    raise AllocationError(
                        "Input's data shape/type is inconsistent",
                        node=self._node,
                        output=self,
                        input=_input,
                    )

                if self._data is not idata:
                    if self._data is not None:
                        idata[:] = self._data
                    self._set_data(idata, owns_buffer=False, override=True)
                return True

        if self._data is not None and self._dd.consistent_with(self._data):
            return False

        if self._dd.shape is None or self._dd.dtype is None:
            raise AllocationError(
                "No shape/type information provided for the Output",
                node=self._node,
                output=self,
            )

        try:
            self._set_data(self._dd.allocate(**kwargs), owns_buffer=True)
        except Exception as exc:
            raise AllocationError(f"Output: {exc.args[0]}", node=self._node, output=self) from exc

        return True

    #
    # Evaluation
    #

    def set(self, data, check_taint: bool = False, force_taint: bool = False) -> bool:
        if self._node.frozen and not force_taint:
            return False

        tainted = (self._data != data).any() if check_taint else True
        if tainted:
            self._data[:] = data
            self._taint_children_implementation()
        return tainted

    def seti(
        self,
        idx: int,
        value: float,
        check_taint: bool = False,
        force_taint: bool = False,
    ) -> bool:
        if self._node.frozen and not force_taint:
            return False

        tainted = self._data[idx] != value if check_taint else True
        if tainted:
            self._data[idx] = value
            self._taint_children_implementation()
        return tainted

    def _taint_children_implementation(self) -> None:
        self.taint_children()
        self._node.invalidate_parents()
        self._node._evaluator.tainted = False

    def taint_children_type(self, force_taint: bool = False) -> None:
        for input in self._child_inputs:
            input.taint_type(force_taint=force_taint)

    def connected(self) -> bool:
        return bool(self._child_inputs)

    #
    # Connection
    #

    def _connect_to_input(self, input):
        """Connect Output to Input with validation checks."""
        if self.closed:
            if input.allocatable:
                raise ConnectionError(
                    "Cannot connect a closed output to an allocatable input!",
                    node=self._node,
                    output=self,
                    input=input,
                )
        elif input.closed:
            raise ConnectionError(
                "Cannot connect an output to a closed input!",
                node=self._node,
                output=self,
                input=input,
            )

        if input.allocatable:
            if self._allocating_input:
                raise ConnectionError(
                    "Output has multiple allocatable/allocated child inputs",
                    node=self._node,
                    output=self,
                )
            if self._forbid_reallocation:
                raise ConnectionError(
                    "Output forbids reallocation and may not connect to allocating inputs",
                    node=self._node,
                    output=self,
                )
            self._allocating_input = input

        self._child_inputs.append(input)
        input.set_parent_output(self)

    def _connect_to_node(self, node, idx_scope=None, reassign_idx_scope=False):
        """Connect Output to Node, finding or creating an unconnected Input."""
        inp = None
        try:
            for inpt in node.inputs:
                if not inpt.connected():
                    inp = inpt
                    break
            if inp is None:
                raise IndexError()
        except IndexError:
            inp = node(idx_scope=idx_scope)
        if inp is None:
            raise ConnectionError(
                "Unable to find unconnected input or create a new one!",
                node=node,
                output=self,
            )
        self._connect_to_input(inp)
        if reassign_idx_scope:
            node.input_strategy._idx_scope = idx_scope

    def __rshift__(self, other):
        """`self >> other` — Output → Input | Node | MetaNode | Inputs | Sequence | Generator | Mapping | NestedMapping."""
        from nested_mapping import NestedMapping

        from .input import Input, Inputs
        from .meta_node import MetaNode
        from .node_base import NodeBase

        if isinstance(other, Input):
            self._connect_to_input(other)
        elif isinstance(other, MetaNode):
            try:
                idx_scope = other._input_strategy._idx_scope + 1
                self._connect_to_node(other, idx_scope=idx_scope, reassign_idx_scope=True)
            except Exception:
                try:
                    other.__rrshift__(self)
                except Exception as exc:
                    raise ConnectionError(
                        "Cannot connect an output to MetaNode: configure `input_strategy` "
                        f"or implement `__rrshift__` in the `{type(other).__name__}` class!",
                        node=other,
                        output=self,
                    ) from exc
        elif isinstance(other, NodeBase):
            idx_scope = other._input_strategy._idx_scope + 1
            self._connect_to_node(other, idx_scope=idx_scope, reassign_idx_scope=True)
        elif isinstance(other, (Sequence, Inputs, Generator)):
            for subother in other:
                self >> subother
        elif isinstance(other, NestedMapping):
            for subother in other.walkvalues():
                self >> subother
        elif isinstance(other, Mapping):
            self >> NestedMapping(dic=other)
        else:
            raise ConnectionError(
                f"Unable to connect the input to {other=}!",
                output=self,
            )
