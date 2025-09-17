from pytest import raises
from numpy import arange

from dag_modelling.core.graph import Graph
from dag_modelling.lib.common import Dummy


def test_output_allocation_1():
    data = arange(12, dtype="d").reshape(3, 4)
    with Graph(close_on_exit=True):
        n1 = Dummy("node1")
        n2 = Dummy("node2")

        out1 = n1._add_output("o1", data=data, allocatable=False)
        in1 = n2._add_input("i1")

        out1 >> in1

    assert (data == out1.data).all()


def test_output_allocation_2():
    data = arange(12, dtype="d").reshape(3, 4)
    with Graph(close_on_exit=True):
        n1 = Dummy("node1")
        n2 = Dummy("node2")

        out1 = n1._add_output("o1", dtype=data.dtype, shape=data.shape)
        in1 = n2._add_input("i1", data=data)
        out1 >> in1

    with raises(ValueError):
        out1.data[:] = data[:]

    out1._data[:] = data[:]

    assert (data == out1.data).all()
    assert (data == in1.data).all()
    assert (data == in1._own_data).all()
    assert data.dtype == out1.data.dtype
    assert data.dtype == in1.data.dtype
    assert data.dtype == in1._own_data.dtype
