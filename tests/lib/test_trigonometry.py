from matplotlib.pyplot import close
from numpy import allclose, arccos, arcsin, arctan, cos, linspace, pi, sin, tan
from pytest import mark

from dag_modelling.core.graph import Graph
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.lib.common import Array
from dag_modelling.lib.trigonometry import ArcCos, ArcSin, ArcTan, Cos, Sin, Tan
from dag_modelling.plot.plot import plot_auto

fcnnames = ("cos", "sin", "tan", "arccos", "arcsin", "arctan")
fcns = (cos, sin, tan, arccos, arcsin, arctan)
fcndict = dict(zip(fcnnames, fcns))

nodes = (Cos, Sin, Tan, ArcCos, ArcSin, ArcTan)
nodedict = dict(zip(fcnnames, nodes))


@mark.parametrize("dtype", ("d", "f"))
@mark.parametrize("fcnname", fcnnames)
def test_Trigonometry_01(test_name, debug_graph, fcnname, dtype):
    fcn_np = fcndict[fcnname]
    fcn_node = nodedict[fcnname]

    if fcnname in ("cos", "sin", "tan"):
        arrays_in = tuple(linspace(-2 * pi, 2 * pi, 101, dtype=dtype) * i for i in (1, 2, 3))
    elif fcnname == "arctan":
        arrays_in = tuple(linspace(-10, 10, 101, dtype=dtype) * i for i in (1, 2, 3))
    else:
        arrays_in = tuple(linspace(-1, 1, 101, dtype=dtype) / i for i in (1, 2, 3))

    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        arrays = tuple(
            Array(f"arr_{i}", array_in, label={"text": f"X axis {i}"}, mode="fill")
            for i, array_in in enumerate(arrays_in)
        )
        node = fcn_node(fcnname)
        arrays >> node

    outputs = node.outputs
    ress = fcn_np(arrays_in)

    assert node.tainted == True
    assert all(output.dd.dtype == dtype for output in outputs)
    assert allclose(tuple(outputs.iter_data()), ress, rtol=0, atol=0)
    assert node.tainted == False

    plot_auto(node.outputs[0], label="input 0")
    plot_auto(node.outputs[1], label="input 1")
    plot_auto(node.outputs[2], label="input 2")
    close()

    savegraph(graph, f"output/{test_name}.png")
