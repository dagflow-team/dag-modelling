from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from numpy import array as nparray
from numpy import full

from nested_mapping import NestedMapping

from ...core.exception import InitializationError
from ...core.node import Node
from ...core.output import Output
from ...core.type_functions import check_dtype_of_edges, check_edges_consistency_with_array
from ...tools.iter import iter_sequence_not_string

if TYPE_CHECKING:
    from typing import Literal

    from numpy.typing import ArrayLike, DTypeLike, NDArray


class Array(Node):
    """Creates a node with a single data output with predefined array."""

    __slots__ = ("_mode", "_data", "_output")

    _mode: Literal["store", "store_weak", "fill"]
    _data: NDArray
    _output: Output

    def __init__(
        self,
        name: str,
        array: NDArray | list[float | int] | tuple[float | int, ...],
        *,
        mode: Literal["store", "store_weak", "fill"] = "store",
        outname: str = "array",
        dtype: DTypeLike = None,
        mark: str | None = None,
        edges: Output | Sequence[Output] | Node | None = None,
        meshes: Output | Sequence[Output] | Node | None = None,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self._mode = mode
        if mark is not None:
            self._labels.setdefault("mark", mark)
        elif edges:
            self._labels.setdefault("mark", "h⃗")
        elif meshes:
            self._labels.setdefault("mark", "y⃗")
        else:
            self._labels.setdefault("mark", "a⃗")
        self._data = nparray(array, copy=True, dtype=dtype)

        if mode == "store":
            self._output = self._add_output(outname, data=self._data)
        elif mode == "store_weak":
            self._output = self._add_output(outname, data=self._data, owns_buffer=False)
        elif mode == "fill":
            self._output = self._add_output(outname, dtype=self._data.dtype, shape=self._data.shape)
        else:
            raise InitializationError(f'Array: invalid mode "{mode}"', node=self)

        self._functions_dict.update(
            {
                "store": self._fcn_store,
                "store_weak": self._fcn_store,
                "fill": self._fcn_fill,
            }
        )
        self.function = self._functions_dict[self._mode]

        if edges:
            self.set_edges(edges)
        if meshes:
            self.set_mesh(meshes)

        if mode == "store":
            self.close()

    def _fcn_store(self):
        pass

    def _fcn_fill(self):
        self._output._data[:] = self._data

    @classmethod
    def from_value(
        cls,
        name,
        value: float | int,
        *,
        store: bool = False,
        edges: Output | Sequence[Output] | Node | None = None,
        shape: int | tuple[int, ...] | None = None,
        dtype: DTypeLike = None,
        **kwargs,
    ):
        if (shape is None) == (edges is None):
            raise RuntimeError("Array: should specify either shape or edges, but not both.")
        match edges:
            case Output():
                shape = (edges.dd.shape[0] - 1,)
            case Node():
                output = edges.outputs[0]
                shape = (output.dd.shape[0] - 1,)
            case Sequence():
                shape = tuple(output.dd.shape[0] - 1 for output in edges)
            case None:
                pass
            case _:
                raise RuntimeError("Invalid edges specification")
        array = full(shape, value, dtype=dtype)

        if store:
            return cls.replicate(name=name, array=array, edges=edges, **kwargs)

        return cls(name, array, edges=edges, **kwargs)

    @classmethod
    def from_storage(
        cls,
        path: str,
        storage: NestedMapping,
        *,
        edgesname: str | Sequence[str] = [],
        meshname: str | Sequence[str] = [],
        remove_processed_arrays: bool = False,
        **kwargs,
    ):
        localstorage = storage(path)
        tmpstorage = NestedMapping(sep=".")
        used_array_keys = set()
        for key, data in localstorage.walkitems():
            skey = ".".join((path,) + key)
            _, istorage = cls.replicate(name=skey, array=data, **kwargs)
            tmpstorage |= istorage
            used_array_keys.add(key)

        edges = [tmpstorage[f"nodes.{path}.{name}"] for name in iter_sequence_not_string(edgesname)]

        mesh = [tmpstorage[f"nodes.{path}.{name}"] for name in iter_sequence_not_string(meshname)]

        if edges or mesh:
            for node in tmpstorage("nodes").walkvalues():
                if node in edges or node in mesh:
                    continue
                node.set_mesh(mesh)
                node.set_edges(edges)

        if remove_processed_arrays:
            for key in used_array_keys:
                localstorage.delete_with_parents(key)
            if not localstorage:
                storage.delete_with_parents(path)

    def _type_function(self) -> None:
        check_dtype_of_edges(self, slice(None), "array")  # checks List[Output]
        check_edges_consistency_with_array(self, "array")  # checks dim and N+1 size

    def _post_allocate(self) -> None:
        if self._mode == "fill":
            return

        self._data = self._output._data

    def set(self, data: ArrayLike, check_taint: bool = False) -> bool:
        return self._output.set(data, check_taint)

    def _check_ndim(self, value: int, type: str):
        ndim = self._output.dd.dim
        if ndim == value:
            return

        raise InitializationError(f"Array ndim is {ndim}. {type} of {value} are not consistent")

    def set_edges(self, edges: Output | Node | Sequence[Output | Node]):
        if not edges:
            return

        if not isinstance(edges, Sequence):
            edges = (edges,)

        dd = self._output.dd
        if dd.axes_edges:
            raise InitializationError("Edges already set", node=self, output=self._output)

        self._check_ndim(len(edges), "Edges")

        dd.edges_inherited = False
        for edgesi in edges:
            if isinstance(edgesi, Output):
                dd.axes_edges += (edgesi,)
            elif isinstance(edgesi, Node):
                dd.axes_edges += (edgesi.outputs[0],)
            else:
                raise InitializationError(
                    "Array: edges must be `Output/Node` or `Sequence[Output/Node]`, "
                    f"got {edges=}, {type(edges)=}"
                )

    def set_mesh(self, meshes: Output | Node | Sequence[Output] | Sequence[Node]):
        if not meshes:
            return

        if not isinstance(meshes, Sequence):
            meshes = (meshes,)

        self._check_ndim(len(meshes), "Meshes")

        dd = self._output.dd
        if dd.axes_meshes:
            raise InitializationError("Meshes already set", node=self, output=self._output)

        dd.meshes_inherited = False
        for meshesi in meshes:
            if isinstance(meshesi, Output):
                dd.axes_meshes += (meshesi,)
            elif isinstance(meshesi, Node):
                dd.axes_meshes += (meshesi.outputs[0],)
            else:
                raise InitializationError(
                    (
                        "Array: meshes must be `Output/Node` or `Sequence[Output/Node]`, "
                        f"got {meshes=}, {type(meshes)=}"
                    ),
                    node=self,
                )
