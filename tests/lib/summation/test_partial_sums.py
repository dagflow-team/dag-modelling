from numpy import allclose, arange, finfo, linspace
from pytest import mark, raises

from dag_modelling.core.exception import TypeFunctionError
from dag_modelling.core.graph import Graph
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.lib.common import Array
from dag_modelling.lib.summation import PartialSums


@mark.parametrize("a", (arange(12, dtype="d") * i for i in (1, 2, 3)))
def test_PartialSums_01(test_name, debug_graph, a, output_path: str):
    arrays_range = [0, 12], [0, 3], [4, 10], [11, 12]
    arrays_res = tuple(a[ranges[0] : ranges[1]].sum() for ranges in arrays_range)

    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        ranges = tuple(Array(f"range_{i}", arr, mode="fill") for i, arr in enumerate(arrays_range))
        arra = Array("a", a, mode="fill")
        ps = PartialSums("partialsums")
        arra >> ps("array")
        ranges >> ps

    atol = finfo("d").resolution * 2
    assert ps.tainted is True
    assert all(
        allclose(output.data[0], res, rtol=0, atol=atol)
        for output, res in zip(ps.outputs, arrays_res)
    )
    assert ps.tainted is False

    savegraph(graph, f"{output_path}/{test_name}.png", show="all")


@mark.parametrize("a", (arange(12, dtype="d") * i for i in (1, 2, 3)))
def test_PartialSums_edges(debug_graph, a):
    arrays_range = [0, 12], [0, 3], [4, 10], [11, 12]
    with Graph(close_on_exit=False, debug=debug_graph) as graph:
        edges = Array("edges", linspace(0, 13, 13), mode="fill")
        arra = Array("a", a, edges=edges["array"], mode="fill")
        ranges = tuple(Array(f"range_{i}", arr, mode="fill") for i, arr in enumerate(arrays_range))
        ps = PartialSums("partialsums")
        arra >> ps("array")
        ranges >> ps
    with raises(TypeFunctionError):
        graph.close()
