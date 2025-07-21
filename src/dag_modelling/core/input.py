from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import zeros

from ..core.labels import repr_pretty
from .data_descriptor import DataDescriptor
from .edges import EdgeContainer
from .exception import AllocationError, ClosedGraphError, InitializationError, ReconnectionError

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray

    from .node_base import NodeBase
    from .output import Output
    from .types import EdgesLike, ShapeLike


class Input:
    __slots__ = (
        "_own_data",
        "_own_dd",
        "_node",
        "_name",
        "_parent_output",
        "_child_output",
        "_allocatable",
        "_owns_buffer",
        "_debug",
    )
    _own_data: NDArray | None
    _own_dd: DataDescriptor

    _node: NodeBase | None
    _name: str | None

    _parent_output: Output | None
    _child_output: Output | None

    _allocatable: bool
    _owns_buffer: bool

    _debug: bool

    def __init__(
        self,
        name: str | None = None,
        node: NodeBase | None = None,
        *,
        child_output: Output | None = None,
        parent_output: Output | None = None,
        debug: bool | None = None,
        allocatable: bool = False,
        data: NDArray | None = None,
        dtype: DTypeLike = None,
        shape: ShapeLike | None = None,
        axes_edges: EdgesLike | None = None,
        axes_meshes: EdgesLike | None = None,
    ):
        if data is not None and (allocatable or dtype is not None or shape is not None):
            raise InitializationError(input=input, node=node)

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

    def __str__(self) -> str:
        return self.connected() and f"→● {self._name}" or f"→○ {self._name}"

    _repr_pretty_ = repr_pretty

    @property
    def own_data(self) -> NDArray | None:
        return self._own_data

    @property
    def own_dd(self) -> DataDescriptor:
        return self._own_dd

    @property
    def owns_buffer(self) -> bool:
        return self._owns_buffer

    def set_own_data(
        self,
        data,
        *,
        owns_buffer: bool,
        axes_edges: EdgesLike | None = None,
        axes_meshes: EdgesLike | None = None,
    ):
        if self.closed:
            raise ClosedGraphError("Unable to set input data.", node=self._node, input=self)
        if self.own_data is not None:
            raise AllocationError("Input already has data.", node=self._node, input=self)

        self._own_data = data
        self._owns_buffer = owns_buffer
        self.own_dd.dtype = data.dtype
        self.own_dd.shape = data.shape
        self.own_dd.axes_edges = axes_edges or ()
        self.own_dd.axes_meshes = axes_meshes or ()

    @property
    def closed(self) -> bool:
        return self._node.closed if self.node else False

    def set_child_output(self, child_output: Output | None, force_taint: bool = False) -> None:
        if not self.closed:
            return self._set_child_output(child_output, force_taint)
        raise ClosedGraphError(input=self, node=self.node, output=child_output)

    def _set_child_output(self, child_output: Output, force_taint: bool = False) -> None:
        if self.child_output and not force_taint:
            raise ReconnectionError(output=self.child_output, node=self.node)
        self._child_output = child_output
        if child_output:
            child_output.parent_input = self

    def set_parent_output(self, parent_output: Output, force_taint: bool = False) -> None:
        if self.closed:
            raise ClosedGraphError(input=self, node=self.node, output=parent_output)

        return self._set_parent_output(parent_output, force_taint)

    def _set_parent_output(self, parent_output: Output, force_taint: bool = False) -> None:
        if self.connected() and not force_taint:
            raise ReconnectionError(output=self._parent_output, node=self.node)
        self._parent_output = parent_output

    @property
    def name(self) -> str | None:
        return self._name

    @name.setter
    def name(self, name) -> None:
        self._name = name

    @property
    def node(self) -> NodeBase | None:
        return self._node

    @property
    def parent_node(self) -> NodeBase | None:
        return self._parent_output.node

    @property
    def logger(self):
        return self._node.logger

    @property
    def child_output(self) -> Output | None:
        return self._child_output

    @property
    def invalid(self) -> bool:
        """Checks validity of the parent output data."""
        return self._parent_output.invalid

    @property
    def has_data(self) -> bool:
        return self._own_data is not None

    @property
    def allocatable(self) -> bool:
        return self._allocatable

    @property
    def debug(self) -> bool:
        return self._debug

    @invalid.setter
    def invalid(self, invalid) -> None:
        """Sets the validity of the current node."""
        self._node.invalid = invalid

    @property
    def parent_output(self) -> Output:
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
    def tainted(self) -> bool:
        return self._parent_output.tainted

    def touch(self):
        return self._parent_output.touch()

    def taint(
        self,
        *,
        force_taint: bool = False,
        force_computation: bool = False,
        caller: Input | None = None,
    ) -> None:
        self._node.taint(
            force_taint=force_taint,
            force_computation=force_computation,
            caller=self if caller is None else caller,
        )

    def taint_type(self, force_taint: bool = False) -> None:
        self._node.taint_type(force_taint=force_taint)

    def connected(self) -> bool:
        return self._parent_output is not None

    def allocate(self, **kwargs) -> bool:
        """Returns True if data was reassigned."""
        if not self._allocatable or (
            (data := self._own_data) is not None and self.own_dd.consistent_with(data)
        ):
            return False

        if self.own_dd.shape is None or self.own_dd.dtype is None:
            raise AllocationError(
                "No shape/type information provided for the Input",
                node=self._node,
                output=self,
            )
        try:
            self._own_data = zeros(self.own_dd.shape, self.own_dd.dtype, **kwargs)
        except Exception as exc:
            raise AllocationError(f"Input: {exc.args[0]}", node=self._node, input=self) from exc

        return True

    def to_dict(self, *, label_from: str = "text") -> dict:
        try:
            output = self.parent_output
        except AttributeError:
            return {"label": "input", "shape": "?"}
        else:
            shape = output.dd.shape
            return {
                "label": output.labels[label_from],
                "shape": shape[0] if len(shape) == 1 else shape,
            }


class Inputs(EdgeContainer):
    __slots__ = ()

    def __init__(self, iterable=None):
        super().__init__(iterable)
        self._dtype = Input

    def __str__(self):
        return f"→[{tuple(obj.name for obj in self)}]○"

    _repr_pretty_ = repr_pretty

    def touch(self) -> None:
        for input in self:
            input.touch()
