from numpy import allclose, array
from pytest import mark

from dag_modelling.core.graph import Graph
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.lib.common import Array
from dag_modelling.lib.linalg import MatrixProductDDt


@mark.parametrize("dtype", ("d", "f"))
def test_MatrixProductDVDt_2d(dtype, output_path: str):
    left = array([[1, 2, 3], [3, 4, 5]], dtype=dtype)

    with Graph(close_on_exit=True) as graph:
        l_array = Array("Left", left, mode="fill")

        prod = MatrixProductDDt("MatrixProductDDt2d")
        l_array >> prod

    desired = left @ left.T
    actual = prod.get_data("result")

    assert allclose(desired, actual, atol=0, rtol=0)

    savegraph(graph, f"{output_path}/test_MatrixProductDDt_2d_{dtype}.png")
