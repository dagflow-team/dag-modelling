from __future__ import annotations

from typing import TYPE_CHECKING

from numba import njit

from ...core.global_parameters import NUMBA_CACHE_ENABLE

if TYPE_CHECKING:
    from numpy.typing import NDArray


@njit(cache=NUMBA_CACHE_ENABLE)
def coo_add(
    rows: NDArray, cols: NDArray, values: NDArray, n: int, row: int, col: int, value: float
) -> int:
    """Add an entry at `n` of the COO arrays if it fits, return the number of entries"""
    if n < rows.shape[0]:
        rows[n] = row
        cols[n] = col
        values[n] = value
    return n + 1


@njit(cache=NUMBA_CACHE_ENABLE)
def coo_to_dense(rows: NDArray, cols: NDArray, values: NDArray, matrix: NDArray) -> None:
    """Write COO entries to a dense matrix, a repeated cell is summed"""
    matrix[:, :] = 0.0
    for k in range(rows.shape[0]):
        matrix[rows[k], cols[k]] += values[k]


@njit(cache=NUMBA_CACHE_ENABLE)
def coo_to_csr(
    rows: NDArray,
    cols: NDArray,
    values: NDArray,
    data: NDArray,
    indices: NDArray,
    indptr: NDArray,
    starts: NDArray,
) -> None:
    """Write COO entries with non-decreasing columns to a CSR matrix.

    A repeated cell is summed in the dtype of `data`, as `matrix[row, col] += value` would do.
    `data` and `indices` need room for all the entries, `starts` is a work array as `indptr`.
    """
    nrows = indptr.shape[0] - 1
    starts[:] = 0
    for k in range(rows.shape[0]):
        starts[rows[k] + 1] += 1
    for row in range(nrows):
        starts[row + 1] += starts[row]
    indptr[:] = starts
    for k in range(rows.shape[0]):
        row = rows[k]
        pos = indptr[row]
        if pos > starts[row] and indices[pos - 1] == cols[k]:
            data[pos - 1] += values[k]
        else:
            data[pos] = values[k]
            indices[pos] = cols[k]
            indptr[row] = pos + 1
    # close the gaps left by the repeated cells
    nnz = 0
    for row in range(nrows):
        start, end = starts[row], indptr[row]
        indptr[row] = nnz
        for j in range(start, end):
            data[nnz] = data[j]
            indices[nnz] = indices[j]
            nnz += 1
    indptr[nrows] = nnz
