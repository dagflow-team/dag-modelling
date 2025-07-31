from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import prod

from ..core.labels import repr_pretty

if TYPE_CHECKING:
    from typing import Literal

    from numpy.typing import DTypeLike, NDArray

    from .types import EdgesLike, MeshesLike, ShapeLike


class DataDescriptor:
    """
    The data descriptor class stores `dtype`, `shape`,
    `axes_edges` and `axes_meshes` information.
    """

    __slots__ = (
        "dtype",
        "_shape",
        "axes_edges",
        "axes_meshes",
        "edges_inherited",
        "meshes_inherited",
    )
    dtype: DTypeLike  # DTypeLike is already Optional
    _shape: ShapeLike | None
    axes_edges: EdgesLike
    axes_meshes: EdgesLike

    edges_inherited: bool
    meshes_inherited: bool

    def __init__(
        self,
        dtype: DTypeLike,  # DTypeLike is already Optional
        shape: ShapeLike | None,
        axes_edges: EdgesLike | None = None,
        axes_meshes: MeshesLike | None = None,
    ) -> None:
        """
        Sets the attributes
        """
        self.dtype = dtype
        self.shape = shape
        self.axes_edges = axes_edges or ()
        self.axes_meshes = axes_meshes or ()

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
        return (
            f"{self.dtype} {self.shape}"
            f'{bool(self.axes_edges) and " [edges]" or ""}'
            f'{bool(self.axes_meshes) and " [meshes]" or ""}'
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

    def consistent_with(self, array: NDArray):
        return array.shape==self._shape and array.dtype==self.dtype

