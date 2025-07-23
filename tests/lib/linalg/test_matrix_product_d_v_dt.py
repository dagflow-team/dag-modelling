from numpy import allclose, array, diag
from pytest import mark

from dag_modelling.core.graph import Graph
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.lib.common import Array
from dag_modelling.lib.linalg import MatrixProductDVDt


@mark.parametrize("dtype", ("d", "f"))
def test_MatrixProductDVDt_2d(dtype):
    left = array([[1, 2, 3], [3, 4, 5]], dtype=dtype)
    square = array(
        [
            [9, 2, 1],
            [0, 4, 2],
            [1.5, 3, 1],
        ],
        dtype=dtype,
    )

    with Graph(close_on_exit=True) as graph:
        l_array = Array("Left", left, mode="fill")
        s_array = Array("Square", square, mode="fill")

        prod = MatrixProductDVDt("MatrixProductDVDt2d")
        l_array >> prod.inputs["left"]
        s_array >> prod.inputs["square"]

    desired = left @ square @ left.T
    actual = prod.get_data("result")

    assert allclose(desired, actual, atol=0, rtol=0)

    savegraph(graph, f"output/test_MatrixProductDVDt_2d_{dtype}.png")


@mark.parametrize("dtype", ("d", "f"))
def test_MatrixProductDVDt_1d(dtype):
    left = array([[1, 2, 3], [3, 4, 5]], dtype=dtype)
    diagonal = array([9, 4, 5], dtype=dtype)

    with Graph(close_on_exit=True) as graph:
        l_array = Array("Left", left, mode="fill")
        s_array = Array("Diagonal", diagonal, mode="fill")

        prod = MatrixProductDVDt("MatrixProductDVDt1d")
        l_array >> prod.inputs["left"]
        s_array >> prod.inputs["square"]

    desired = left @ diag(diagonal) @ left.T
    actual = prod.get_data("result")

    assert allclose(desired, actual, atol=0, rtol=0)

    savegraph(graph, f"output/test_MatrixProductDVDt_1d_{dtype}.png")
