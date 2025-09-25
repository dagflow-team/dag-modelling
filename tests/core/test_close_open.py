from numpy import arange, array

from dag_modelling.core.graph import Graph
from dag_modelling.lib.common import Array
from dag_modelling.lib.arithmetic import Product, Sum
from dag_modelling.lib.summation import WeightedSum


def test_00(debug_graph):
    with Graph(debug=debug_graph, close_on_exit=True):
        arr = Array("arr", arange(3, dtype="d"), mode="fill")  # [0, 1, 2]
        ws = WeightedSum("weightedsum")
        (arr, arr) >> ws
        Array("weight", (2, 3), mode="fill") >> ws("weight")
    assert ws.closed
    assert (ws.outputs["result"].data == [0, 5, 10]).all()

    assert arr.open(open_children=False)
    assert not arr.closed
    assert ws.inputs["weight"].closed
    assert arr.close(close_children=False)
    assert arr.closed
    assert ws.inputs["weight"].closed

    assert arr.open(open_children=True)
    assert not arr.closed
    assert not ws.inputs["weight"].closed
    assert arr.close(close_children=True)
    assert arr.closed
    assert ws.inputs["weight"].closed


def test_01(debug_graph):
    with Graph(debug=debug_graph, close_on_exit=True):
        arr1 = Array("arr1", arange(3, dtype="d"), mode="fill")  # [0, 1, 2]
        arr2 = Array("arr2", array((3, 2, 1), dtype="d"), mode="fill")
        sum = Sum("sum")
        (arr1, arr2) >> sum
    assert sum.closed
    assert (sum.outputs["result"].data == [3, 3, 3]).all()
    assert sum.open()
    assert all((not sum.closed, arr1.closed, arr2.closed))
    assert arr1.open()
    assert all((not sum.closed, not arr1.closed, arr2.closed))
    assert arr2.open()
    assert all((not sum.closed, not arr1.closed, not arr2.closed))


def test_02(debug_graph):
    with Graph(debug=debug_graph, close_on_exit=True):
        arr1 = Array("arr1", arange(3, dtype="d"), mode="fill")  # [0, 1, 2]
        arr2 = Array("arr2", array((3, 2, 1), dtype="d"), mode="fill")
        arr3 = Array("unity", array((1, 1, 1), dtype="d"), mode="fill")
        sum1 = Sum("sum1")
        sum2 = Sum("sum2")
        prod = Product("product")
        (arr1, arr2, arr3) >> sum1  # [4, 4, 4]
        (arr3, sum1) >> prod  # [4, 4, 4]
        (arr1, prod) >> sum2  # [4, 5, 6]
    assert sum2.closed
    assert (sum2.outputs["result"].data == [4, 5, 6]).all()
    assert arr1.open(open_children=True)
    assert arr2.closed
    assert arr3.closed
    assert not arr1.closed
    assert not prod.closed
    assert not sum1.closed


def test_03(debug_graph):
    with Graph(debug=debug_graph, close_on_exit=False):
        arr1 = Array("arr1", arange(3, dtype="d"), mode="fill")  # [0, 1, 2]
        arr2 = Array("arr2", array((3, 2, 1), dtype="d"), mode="fill")
        arr3 = Array("unity", array((1, 1, 1), dtype="d"), mode="fill")
        sum1 = Sum("sum1")
        sum2 = Sum("sum2")
        prod = Product("product")
        (arr1, arr2, arr3) >> sum1  # [4, 4, 4]
        (arr3, sum1) >> prod  # [4, 4, 4]
        (arr1, prod) >> sum2  # [4, 5, 6]

    with Graph(debug=debug_graph, close_on_exit=True):
        arr4 = Array("arr1", arange(3, dtype="d"), mode="fill")  # [0, 1, 2]
        sum3 = Sum("sum3")
        (sum2, arr4) >> sum3  # [4, 7, 8]
    assert arr1.closed
    assert arr2.closed
    assert arr3.closed
    assert arr4.closed
    assert sum2.closed
    assert sum3.closed
    assert (sum3.outputs["result"].data == [4, 6, 8]).all()
    assert arr1.open(open_children=True)
    assert arr2.closed
    assert arr3.closed
    assert arr4.closed
    assert not arr1.closed
    assert not prod.closed
    assert not sum1.closed
    assert not sum2.closed
