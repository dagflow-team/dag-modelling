from numpy import array, array_equal, zeros
from scipy.sparse import csr_array

from dag_modelling.core.csr_array import CSRArray
from dag_modelling.methods.sparse.csr import csr_copy, csr_set_sequential, with_headroom


def test_csr_set_sequential():
    data, indices, indptr = zeros(3), zeros(3, dtype="i4"), zeros(4, dtype="i4")
    nnz, row = 0, 0
    # a repeated cell is replaced, as a dense matrix would do; zeros are not stored
    cells = ((0, 1, 1.0), (0, 1, 2.0), (1, 0, 0.0), (2, 0, 3.0), (2, 0, 0.0))
    for irow, icol, value in cells:
        nnz, row = csr_set_sequential(data, indices, indptr, nnz, row, irow, icol, value)
    row += 1
    indptr[row:] = nnz

    matrix = csr_array((data[:nnz], indices[:nnz], indptr), shape=(3, 2))
    assert nnz == 1 and array_equal(matrix.indices, [1])
    assert array_equal(matrix.toarray(), [[0.0, 2.0], [0.0, 0.0], [0.0, 0.0]])


def test_with_headroom():
    # a quarter more, rounded up: 1 to 3 entries get a spare one too
    assert [with_headroom(count) for count in (0, 1, 3, 4, 5, 8)] == [0, 2, 4, 5, 7, 10]


def test_csr_copy():
    source = csr_array(array([[1.0, 0.0, 2.0], [0.0, 0.0, 3.0]]))
    output = CSRArray.allocate(source.shape, "d", 1)
    csr_copy(source, output)
    assert output.nnz == source.nnz and output.capacity == with_headroom(source.nnz)
    assert output.matrix.data.size == source.nnz
    assert output.toarray().tobytes() == source.toarray().tobytes()

    # a CSRArray with spare room is a source too
    other = CSRArray.allocate(source.shape, "d", 5)
    csr_copy(output, other)
    assert other.toarray().tobytes() == source.toarray().tobytes()
