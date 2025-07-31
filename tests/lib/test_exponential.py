from matplotlib.pyplot import close
from numpy import allclose, exp, expm1, finfo, linspace, log, log1p, log10
from pytest import mark

from dag_modelling.core.graph import Graph
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.lib.common import Array
from dag_modelling.lib.exponential import Exp, Expm1, Log, Log1p, Log10
from dag_modelling.plot.plot import plot_auto

fcnnames = ("exp", "expm1", "log", "log1p", "log10")
fcns = (exp, expm1, log, log1p, log10)
fcndict = dict(zip(fcnnames, fcns))

nodes = (Exp, Expm1, Log, Log1p, Log10)
nodedict = dict(zip(fcnnames, nodes))


@mark.parametrize("dtype", ("d", "f"))
@mark.parametrize("fcnname", fcnnames)
def test_Exponential_01(test_name, debug_graph, fcnname, dtype, output_path: str):
    fcn_np = fcndict[fcnname]
    fcn_node = nodedict[fcnname]

    n = 101
    if "m1" in fcnname or "1p" in fcnname:
        arrays_in = tuple(linspace(-0.3, 0.3, n, dtype=dtype) * i for i in (1, 2, 3))
    elif fcnname.startswith("log"):
        arrays_in = tuple(linspace(0, 10, n + 1, dtype=dtype)[1:] * i for i in (1, 2, 3))
    elif fcnname == "exp":
        arrays_in = tuple(linspace(-10, 10, n, dtype=dtype) * i for i in (1, 2, 3))
    else:
        raise RuntimeError()

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
    atol = finfo("d").resolution * 2
    assert allclose(tuple(outputs.iter_data()), ress, rtol=0, atol=atol)
    assert node.tainted == False

    plot_auto(node.outputs[0], label="input 0")
    plot_auto(node.outputs[1], label="input 1")
    plot_auto(node.outputs[2], label="input 2")
    close()

    savegraph(graph, f"{output_path}/{test_name}.png")
