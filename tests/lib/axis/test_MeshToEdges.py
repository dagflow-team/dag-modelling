from numpy import allclose, linspace
from pytest import mark

from dag_modelling.core.graph import Graph
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.lib.axis import MeshToEdges
from dag_modelling.lib.common import Array


@mark.parametrize("dtype", ("d", "f"))
def test_MeshToEdges_Center_01(testname, debug_graph, dtype):
    dtype = "d"
    array1 = linspace(0, 100, 26, dtype=dtype)

    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        arr1 = Array("linspace", array1, mode="fill")
        me = MeshToEdges("edges")
        arr1 >> me

    res1 = me.outputs[0].data
    edges_mid = (array1[1:] + array1[:-1]) * 0.5
    centers_back = (res1[1:] + res1[:-1]) * 0.5
    widths = res1[1:] - res1[:-1]

    assert res1.dtype == dtype
    assert allclose(res1[1:-1], edges_mid, rtol=0, atol=0)
    assert allclose(centers_back, array1, rtol=0, atol=0)
    assert allclose(widths[0], widths, rtol=0, atol=0)

    savegraph(graph, f"output/{testname}.png")
