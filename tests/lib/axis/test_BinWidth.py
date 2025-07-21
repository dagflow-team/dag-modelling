from matplotlib import pyplot as plt
from numpy import allclose, finfo, geomspace, linspace, ndarray, ones_like, zeros_like
from pytest import mark

from dagflow.core.graph import Graph
from dagflow.plot.graphviz import savegraph
from dagflow.lib.axis import BinWidth
from dagflow.lib.common import Array


@mark.parametrize("dtype", ("d", "f"))
def test_BinWidth_01(testname, debug_graph, dtype):
    array1 = linspace(0, 100, 26, dtype=dtype)
    array2 = geomspace(1e-2, 1e2, 10, dtype=dtype)

    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        arr1 = Array("linspace", array1, mode="fill")
        arr2 = Array("geomspace", array2, mode="fill")
        bc = BinWidth("centers")
        arr1 >> bc
        arr2 >> bc("geomspace")

    def bincenter(x: ndarray):
        return (x[1:] - x[:-1])

    res1 = bc.outputs[0].data
    res2 = bc.outputs[1].data

    assert res1.dtype == dtype
    assert (res1 == 4 ).all()
    assert (res1 == bincenter(array1)).all()

    assert res2.dtype == dtype
    assert (res2 == bincenter(array2)).all()

    assert bc.outputs[0].dd.axes_edges[0] == arr1.outputs[0]
    assert bc.outputs[1].dd.axes_edges[0] == arr2.outputs[0]

    plt.scatter(array1, zeros_like(array1), label="linspace edges")
    plt.scatter(res1, zeros_like(res1), marker="+", label="linspace centers")

    plt.scatter(array2, ones_like(array2), label="geomspace edges")
    plt.scatter(res2, ones_like(res2), marker="+", label="geomspace centers")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig(f"output/{testname}-plot.png")
    plt.close()

    savegraph(graph, f"output/{testname}.png")
