from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import zeros
from scipy.sparse import csr_array

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray

    from .types import ShapeLike


def _readonly(array: NDArray) -> NDArray:
    view = array.view()
    view.flags.writeable = False
    return view


class CSRArray:
    """The arrays of a CSR matrix with room for `capacity` entries, filled in place.

    `indptr` has one entry per row plus a sentinel: row `i` is `data[indptr[i]:indptr[i + 1]]`.

    `matrix` is a read-only `scipy.sparse.csr_array` of the first `nnz` entries. The producer
    writes the canonical form (the columns of a row ascend and do not repeat) and calls `update()`.
    """

    __slots__ = ("data", "indices", "indptr", "matrix", "_data_ro", "_indices_ro")
    layout = "sparse-csr"
    ndim = 2

    data: NDArray
    indices: NDArray
    indptr: NDArray
    matrix: csr_array

    def __init__(self, data: NDArray, indices: NDArray, indptr: NDArray, shape: ShapeLike):
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self._data_ro = _readonly(data)
        self._indices_ro = _readonly(indices)
        self.matrix = csr_array(tuple(shape), dtype=data.dtype)
        self.matrix.indptr = _readonly(indptr)
        self.update()

    @classmethod
    def allocate(cls, shape: ShapeLike, dtype: DTypeLike, capacity: int) -> CSRArray:
        return cls(zeros(capacity, dtype), zeros(capacity, "i4"), zeros(shape[0] + 1, "i4"), shape)

    @property
    def shape(self) -> ShapeLike:
        return self.matrix.shape

    @property
    def dtype(self) -> DTypeLike:
        return self.data.dtype

    @property
    def nnz(self) -> int:
        return int(self.indptr[-1])

    @property
    def size(self) -> int:
        # stored entries, as in scipy
        return self.nnz

    @property
    def capacity(self) -> int:
        return self.data.size

    @property
    def nbytes(self) -> int:
        return self.data.nbytes + self.indices.nbytes + self.indptr.nbytes

    def update(self) -> None:
        """Point `matrix` to the filled entries"""
        nnz = self.indptr[-1]
        self.matrix.data = self._data_ro[:nnz]
        self.matrix.indices = self._indices_ro[:nnz]

    def reserve(self, capacity: int) -> None:
        """Make room for `capacity` entries, keeping the filled ones"""
        if capacity <= self.data.size:
            return
        nnz = self.nnz
        data, indices = zeros(capacity, self.data.dtype), zeros(capacity, self.indices.dtype)
        data[:nnz], indices[:nnz] = self.data[:nnz], self.indices[:nnz]
        self.data, self.indices = data, indices
        self._data_ro, self._indices_ro = _readonly(data), _readonly(indices)
        self.update()

    def toarray(self) -> NDArray:
        return self.matrix.toarray()
