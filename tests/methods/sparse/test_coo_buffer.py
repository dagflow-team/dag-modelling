from numpy import array_equal, eye, full, zeros
from pytest import raises

from dag_modelling.methods.sparse.coo import coo_add
from dag_modelling.methods.sparse.coo_buffer import COOBuffer
from dag_modelling.methods.sparse.csr import with_headroom


def diagonal_kernel(rows, cols, values) -> int:
    count = 0
    for k in range(5):
        count = coo_add(rows, cols, values, count, k, k, 1.0)
    return count


def test_COOBuffer_fill():
    # the entries do not fit: the arrays grow and the kernel runs again
    buffer = COOBuffer(2, 5)
    dense = full((5, 5), 9.0)
    buffer.fill(dense, diagonal_kernel)
    assert buffer.capacity == with_headroom(5)
    assert array_equal(dense, eye(5))


def test_COOBuffer_write():
    # the entries that did not fit are lost, writing them must fail
    with raises(RuntimeError, match="room for 6 only"):
        COOBuffer(6, 3).write(7, zeros((3, 3)))
