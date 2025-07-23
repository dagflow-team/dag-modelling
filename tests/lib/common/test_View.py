from numpy import arange

from dag_modelling.core.graph import Graph
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.lib.common import Array, View

debug = False


def test_View_00():
    """Create four nodes: sum up three of them, multiply the result by the fourth
    Use graph context to create the graph.
    Use one-line code for connecting the nodes
    """
    array = arange(5.0)
    with Graph(close_on_exit=True) as graph:
        initial = Array("array", array, mode="fill")
        view = View("view")
        view2 = View("view2")

        initial >> view
        view >> view2

    assert initial.tainted == True
    assert view.tainted == True
    assert view2.tainted == True

    result = view.get_data()
    result2 = view2.get_data()
    assert (result == array).all()
    assert (result2 == array).all()
    assert view.tainted == False
    assert view2.tainted == False
    assert initial.tainted == False

    d1 = initial.outputs[0]._data
    d2 = view.outputs[0]._data
    d3 = view2.outputs[0]._data
    assert (d1 == d2).all()
    assert (d1 == d3).all()
    d1[:] = -1
    assert (d2 == -1).all()
    assert (d3 == -1).all()

    initial.taint()
    assert initial.tainted == True
    assert view.tainted == True
    assert view2.tainted == True

    view2.touch()
    savegraph(graph, "output/test_View_00.png")
