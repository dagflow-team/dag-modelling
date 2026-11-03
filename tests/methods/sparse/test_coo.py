from numpy import array, array_equal, full, zeros
from pytest import mark
from scipy.sparse import csr_array

from dag_modelling.methods.sparse.coo import coo_add, coo_to_csr, coo_to_dense


def test_coo_add():
    rows, cols, values = zeros(2, dtype="i4"), zeros(2, dtype="i4"), zeros(2)
    count = 0
    for row, col, value in ((0, 1, 1.0), (1, 0, 2.0), (1, 1, 3.0)):
        count = coo_add(rows, cols, values, count, row, col, value)
    # the last entry does not fit, but is counted
    assert count == 3
    assert array_equal(rows, [0, 1]) and array_equal(cols, [1, 0])
    assert array_equal(values, [1.0, 2.0])


@mark.parametrize("dtype", ("d", "f"))
def test_coo(dtype: str):
    # column by column, the rows go down and the cell (2, 0) comes back
    rows = array([2, 0, 2, 1, 1, 2, 0], dtype="i4")
    cols = array([0, 0, 0, 1, 2, 2, 2], dtype="i4")
    values = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    desired = zeros((3, 3), dtype)
    for row, col, value in zip(rows, cols, values):
        desired[row, col] += value

    dense = full((3, 3), 9.0, dtype)
    coo_to_dense(rows, cols, values, dense)
    assert dense.tobytes() == desired.tobytes()

    data, indices = zeros(rows.size, dtype), zeros(rows.size, dtype="i4")
    indptr, starts = zeros(4, dtype="i4"), zeros(4, dtype="i4")
    coo_to_csr(rows, cols, values, data, indices, indptr, starts)
    nnz = indptr[-1]
    matrix = csr_array((data[:nnz], indices[:nnz], indptr), shape=(3, 3))
    assert nnz == 6 and matrix.has_canonical_format
    assert matrix.toarray().tobytes() == desired.tobytes()
