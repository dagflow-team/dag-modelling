from __future__ import annotations

from typing import TYPE_CHECKING

from numba import njit
from numpy import count_nonzero

from ...core.global_parameters import NUMBA_CACHE_ENABLE

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.sparse import csr_array

    from ...core.csr_array import CSRArray


@njit(cache=NUMBA_CACHE_ENABLE)
def csr_set_sequential(
    data: NDArray,
    indices: NDArray,
    indptr: NDArray,
    nnz: int,
    row: int,
    irow: int,
    icol: int,
    value: float,
) -> tuple[int, int]:
    """Set `(irow, icol)` of a CSR matrix filled row by row to `value`, zeros are not stored.

    `nnz` is the number of entries so far and `row` is the last row started in `indptr`,
    both are returned updated.
    """
    if nnz > indptr[row] and row == irow and indices[nnz - 1] == icol:
        # the same cell again: replace it, as a dense matrix would
        nnz -= 1
    if value == 0.0:
        return nnz, row
    while row < irow:
        row += 1
        indptr[row] = nnz
    data[nnz] = value
    indices[nnz] = icol
    return nnz + 1, row


def with_headroom(count: int) -> int:
    """Room for `count` entries and a quarter more, rounded up: the next fills rarely grow it"""
    return count + (count + 3) // 4


def csr_copy(matrix: csr_array | CSRArray, output: CSRArray) -> None:
    """Copy a CSR matrix to `output`, the arrays of `output` grow if needed"""
    nnz = matrix.nnz
    if nnz > output.capacity:
        output.reserve(with_headroom(nnz))
    output.data[:nnz] = matrix.data[:nnz]
    output.indices[:nnz] = matrix.indices[:nnz]
    output.indptr[:] = matrix.indptr
    output.update()


def dense_to_csr(matrix: NDArray, output: CSRArray) -> None:
    """Write the non-zero cells of a dense matrix to `output`"""
    nnz = count_nonzero(matrix)
    if nnz > output.capacity:
        output.reserve(with_headroom(nnz))
    _dense_to_csr(matrix, output.data, output.indices, output.indptr)
    output.update()


@njit(cache=NUMBA_CACHE_ENABLE)
def _dense_to_csr(matrix: NDArray, data: NDArray, indices: NDArray, indptr: NDArray) -> None:
    nnz = 0
    indptr[0] = 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if matrix[row, col] != 0.0:
                data[nnz] = matrix[row, col]
                indices[nnz] = col
                nnz += 1
        indptr[row + 1] = nnz


@njit(cache=NUMBA_CACHE_ENABLE)
def csr_matvec(
    data: NDArray, indices: NDArray, indptr: NDArray, vector: NDArray, out: NDArray
) -> None:
    """Compute `M @ vector`"""
    # a while loop over a running index is faster than range() on short rows
    j = indptr[0]
    for row in range(indptr.shape[0] - 1):
        acc = 0.0
        end = indptr[row + 1]
        while j < end:
            acc += data[j] * vector[indices[j]]
            j += 1
        out[row] = acc


@njit(cache=NUMBA_CACHE_ENABLE)
def csr_vecmat(
    vector: NDArray, data: NDArray, indices: NDArray, indptr: NDArray, out: NDArray
) -> None:
    """Compute `vector @ M`"""
    out[:] = 0.0
    j = indptr[0]
    for row in range(indptr.shape[0] - 1):
        factor = vector[row]
        end = indptr[row + 1]
        while j < end:
            out[indices[j]] += factor * data[j]
            j += 1
