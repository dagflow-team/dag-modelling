from numpy import arange, concatenate, ones, zeros
from pytest import mark, raises

from dag_modelling.core.exception import ConnectionError
from dag_modelling.core.graph import Graph
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.lib.common import Array, View, ViewConcat
from dag_modelling.lib.statistics import NormalizeCorrelatedVarsTwoWays

debug = False


@mark.parametrize("closemode", ["graph", "recursive"])
def test_ViewConcat_00(closemode):
    """Create four nodes: sum up three of them, multiply the result by the fourth
    Use graph context to create the graph.
    Use one-line code for connecting the nodes
    """
    closegraph = closemode == "graph"

    array1 = arange(5.0)
    array2 = ones(shape=10, dtype="d")
    array3 = zeros(shape=12, dtype="d") - 1
    array = concatenate((array1, array2, array3))
    arrays = (array1, array2, array3)
    n1, n2, _ = (a.size for a in arrays)
    with Graph(debug=debug, close_on_exit=closegraph) as graph:
        inputs = [Array("array", array, mode="fill") for array in arrays]
        concat = ViewConcat("concat")
        view = View("view")

        inputs >> concat
        concat >> view

    if not closegraph:
        view.close()

    graph.print()

    assert all(initial.tainted == True for initial in inputs)
    assert concat.tainted == True
    assert view.tainted == True

    result = concat.get_data()
    result_view = view.get_data()
    assert (result == array).all()
    assert (result_view == array).all()
    assert concat.tainted == False
    assert view.tainted == False
    assert all(i.tainted == False for i in inputs)

    data1, data2, data3 = (i.outputs[0]._data for i in inputs)
    datac = concat.get_data(0)
    datav = view.get_data(0)
    assert all(data1 == datac[: data1.size])
    assert all(data2 == datac[n1 : n1 + data2.size])
    assert all(data3 == datac[n1 + n2 : n1 + n2 + data3.size])

    data1[2] = -1
    data2[:] = -1
    data3[::2] = -2
    assert all(data1 == datac[: data1.size])
    assert all(data2 == datac[n1 : n1 + data2.size])
    assert all(data3 == datac[n1 + n2 : n1 + n2 + data3.size])
    assert all(data1 == datav[: data1.size])
    assert all(data2 == datav[n1 : n1 + data2.size])
    assert all(data3 == datav[n1 + n2 : n1 + n2 + data3.size])

    inputs[1].taint()
    assert concat.tainted == True
    assert view.tainted == True

    view.touch()
    savegraph(graph, "output/test_ViewConcat_00.png")


def test_ViewConcat_01():
    with Graph() as graph:
        concat = ViewConcat("concat")
        concat2 = ViewConcat("concat 2")
        view = View("view")
        normnode = NormalizeCorrelatedVarsTwoWays("normvars")

        with raises(ConnectionError):
            view >> concat

        with raises(ConnectionError):
            normnode.outputs[0] >> concat

        with raises(ConnectionError):
            concat >> normnode.inputs[0]

        with raises(ConnectionError):
            concat >> concat2

    savegraph(graph, "output/test_ViewConcat_01.png")
