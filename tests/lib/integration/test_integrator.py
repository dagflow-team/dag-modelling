from matplotlib.pyplot import close, gca
from numpy import allclose, linspace, pi

from dag_modelling.core.graph import Graph
from dag_modelling.core.meta_node import MetaNode
from dag_modelling.lib.common import Array
from dag_modelling.lib.integration import Integrator
from dag_modelling.lib.trigonometry import Cos, Sin
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.plot.plot import plot_auto


def test_Integrator_trap(debug_graph, output_path: str):
    metaint = MetaNode()

    xlabel = "Edges for the integrator"
    with Graph(debug=debug_graph, close_on_exit=True) as graph:
        npoints = 10
        edges = Array("edges", linspace(0, pi, npoints + 1), label={"axis": xlabel}, mode="store")
        orders_x = Array("orders_x", [1000] * npoints, edges=edges["array"], mode="store")
        A = Array("A", edges._data[:-1], mode="fill")
        B = Array("B", edges._data[1:], mode="fill")

        metaint = Integrator(
            "trap",
            labels={"integrator": {"plot_title": "Integrator", "axis": "integral"}},
        )

        cosf = Cos("cos")
        sinf = Sin("sin")
        orders_x >> metaint.inputs["orders_x"]

        metaint.outputs["x"] >> (cosf(), sinf())

        (cosf.outputs[0], sinf.outputs[0]) >> metaint

        sincheck = Sin("sin")
        coscheck = Cos("cos")
        A >> (sincheck(), coscheck())
        B >> (sincheck(), coscheck())

        metaint.print()
    res1 = sincheck.outputs[1].data - sincheck.outputs[0].data
    res2 = -coscheck.outputs[1].data + coscheck.outputs[0].data
    assert allclose(metaint.outputs[0].data, res1, rtol=0, atol=1e-2)
    assert allclose(metaint.outputs[1].data, res2, rtol=0, atol=1e-2)
    assert metaint.outputs[0].dd.axes_edges == [edges["array"]]
    assert metaint.outputs[1].dd.axes_edges == [edges["array"]]

    plot_auto(metaint)
    ax = gca()
    assert ax.get_xlabel() == xlabel
    assert ax.get_ylabel() == "integral"
    assert ax.get_title() == "Integrator"
    close()

    savegraph(graph, f"{output_path}/test_Integrator_trap_meta.pdf", show="all")

