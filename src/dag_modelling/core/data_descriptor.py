from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from numpy import prod, zeros

from ..core.labels import repr_pretty
from .csr_array import CSRArray
from .exception import InitializationError

if TYPE_CHECKING:
    from typing import Literal

    from numpy.typing import DTypeLike, NDArray

    from .node_base import NodeBase
    from .types import EdgesLike, LayoutLike, MeshesLike, ShapeLike


class Layout(StrEnum):
    DENSE = "dense"
    SPARSE_CSR = "sparse-csr"


LAYOUTS = tuple(member.value for member in Layout)


def check_layout(layout: str, node: NodeBase) -> None:
    """Raise `InitializationError` if `layout` is not one of `LAYOUTS`"""
    if layout not in LAYOUTS:
        raise InitializationError(f"layout must be one of {LAYOUTS}, got {layout!r}", node=node)


class DataDescriptor:
    """
    The data descriptor class stores the data information:
        `dtype`: the data type
        `shape`: the shape of the array, or of the full matrix for a sparse layout
        `layout`: "dense" for a numpy array or "sparse-csr" for a Compressed Sparse Row matrix
        `capacity`: the number of entries to allocate for a sparse layout, the arrays may grow
        `axes_edges`, `axes_meshes`: outputs with the bin edges and meshes of each axis
    """

    __slots__ = (
        "dtype",
        "_shape",
        "axes_edges",
        "axes_meshes",
        "edges_inherited",
        "meshes_inherited",
        "layout",
        "capacity",
    )
    dtype: DTypeLike  # DTypeLike is already Optional
    _shape: ShapeLike | None
    axes_edges: EdgesLike
    axes_meshes: EdgesLike
    layout: LayoutLike
    capacity: int | None

    edges_inherited: bool
    meshes_inherited: bool

    def __init__(
        self,
        dtype: DTypeLike,  # DTypeLike is already Optional
        shape: ShapeLike | None,
        axes_edges: EdgesLike | None = None,
        axes_meshes: MeshesLike | None = None,
        layout: LayoutLike = Layout.DENSE.value,
        capacity: int | None = None,
    ) -> None:
        """
        Sets the attributes
        """
        self.dtype = dtype
        self.shape = shape
        self.axes_edges = axes_edges or ()
        self.axes_meshes = axes_meshes or ()
        self.layout = layout.value if isinstance(layout, Layout) else layout
        self.capacity = capacity

        self.edges_inherited = True
        self.meshes_inherited = True

    @property
    def shape(self) -> ShapeLike | None:
        return self._shape

    @shape.setter
    def shape(self, value: ShapeLike | None):
        if value is None:
            self._shape = None
            return

        self._shape = tuple(int(el) for el in value)

    def __str__(self):
        layout = f" [{self.layout}]" if self.layout != Layout.DENSE else ""
        return (
            f"{self.dtype} {self.shape}"
            f'{bool(self.axes_edges) and " [edges]" or ""}'
            f'{bool(self.axes_meshes) and " [meshes]" or ""}'
            f"{layout}"
        )

    _repr_pretty_ = repr_pretty

    @property
    def dim(self) -> int:
        """Return the dimension of the data"""
        return len(self.shape)

    @property
    def size(self) -> int:
        """Return the size of the data"""
        return prod(self.shape)

    @property
    def edges_arrays(self) -> tuple[NDArray, ...]:
        return tuple(o.data for o in self.axes_edges)

    @property
    def meshes_arrays(self) -> tuple[NDArray, ...]:
        return tuple(o.data for o in self.axes_meshes)

    def axis_label(
        self,
        axis: int = 0,
        axistype: Literal["any", "edges", "mesh"] = "any",
        *,
        root: bool = False,
    ) -> str | None:
        if self.axes_edges and axistype in {"any", "edges"}:
            try:
                if root:
                    return self.axes_edges[axis].labels.rootaxis_unit
                else:
                    return self.axes_edges[axis].labels.axis_unit
            except IndexError as e:
                raise RuntimeError(f"Invalid axis index {axis}") from e

        if self.axes_meshes and axistype in {"any", "mesh"}:
            try:
                if root:
                    return self.axes_meshes[axis].labels.rootaxis_unit
                else:
                    return self.axes_meshes[axis].labels.axis_unit
            except IndexError as e:
                raise RuntimeError(f"Invalid axis index {axis}") from e

        return None

    def allocate(self, **kwargs) -> NDArray | CSRArray:
        """Return zeroed data of this description"""
        if self.layout == Layout.SPARSE_CSR:
            return CSRArray.allocate(self._shape, self.dtype, self.capacity)
        return zeros(self._shape, self.dtype, **kwargs)

    def consistent_with(self, array: NDArray):
        return (
            array.shape == self._shape
            and array.dtype == self.dtype
            and getattr(array, "layout", Layout.DENSE.value) == self.layout
        )
