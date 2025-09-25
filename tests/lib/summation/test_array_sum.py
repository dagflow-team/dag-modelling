from numpy import allclose, arange, finfo
from pytest import mark

from dag_modelling.core.graph import Graph
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.lib.common import Array
from dag_modelling.lib.summation import ArraySum


@mark.parametrize("a", (arange(12, dtype="d") * i for i in (1, 2, 3)))
def test_ArraySum_01(test_name, debug_graph, a, output_path: str):
    a2 = a * (-1.1)
    array_res = a.sum()
    array2_res = a2.sum()

    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        arra = Array("a", a, mode="fill")
        arra2 = Array("a2", a2, mode="fill")
        arraysum = ArraySum("arraysum")
        (arra, arra2) >> arraysum

    atol = finfo("d").resolution * 2
    assert arraysum.tainted
    assert allclose(arraysum.get_data(0)[0], array_res, rtol=0, atol=atol)
    assert allclose(arraysum.get_data(1)[0], array2_res, rtol=0, atol=atol)
    arraysum.taint()
    arraysum.touch()
    assert allclose(arraysum.get_data(0)[0], array_res, rtol=0, atol=atol)
    assert allclose(arraysum.get_data(1)[0], array2_res, rtol=0, atol=atol)
    assert arraysum.tainted is False

    savegraph(graph, f"{output_path}/{test_name}.png", show="all")
