from numpy import concatenate, linspace

from dag_modelling.core.graph import Graph
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.lib.common import Array, Concatenation


def test_Concatenation_00(debug_graph):
    array1 = [1.0, 2.0]
    array2 = [3.0, 4.0, 5.0]
    array3 = [6.0]
    arrays = (array1, array2, array3)
    array = concatenate(arrays)
    with Graph(debug=debug_graph, close_on_exit=True) as graph:
        inputs = [Array("array", array, mode="fill") for array in arrays]
        concat = Concatenation("concat")
        inputs >> concat

    graph.print()

    assert all(initial.tainted == True for initial in inputs)
    assert concat.tainted == True

    result = concat.get_data()
    assert (result == array).all()
    assert concat.tainted == False
    assert all(i.tainted == False for i in inputs)

    data1, data2, data3 = (i.get_data(0) for i in inputs)
    n1, n2, n3 = (len(a) for a in arrays)
    datac = concat.get_data(0)
    assert all(data1 == datac[:n1])
    assert all(data2 == datac[n1 : n1 + n2])
    assert all(data3 == datac[n1 + n2 : n1 + n2 + n3])

    assert concat.sizes == (2, 3, 1)

    inputs[1].taint()
    assert concat.tainted == True

    savegraph(graph, "output/test_Concatatenation_00.png")


def test_Concatenation_01(debug_graph):
    array1 = linspace(0, 5, 5)
    array2 = linspace(5, 10, 10)
    array3 = linspace(10, 20, 100)
    arrays = (array1, array2, array3)
    array = concatenate(arrays)
    with Graph(debug=debug_graph, close_on_exit=True) as graph:
        inputs = [Array("array", array, mode="fill") for array in arrays]
        concat = Concatenation("concat")
        inputs >> concat

    graph.print()

    assert all(initial.tainted == True for initial in inputs)
    assert concat.tainted == True

    result = concat.get_data()
    assert (result == array).all()
    assert concat.tainted == False
    assert all(i.tainted == False for i in inputs)

    data1, data2, data3 = (i.get_data(0) for i in inputs)
    n1, n2, n3 = (a.size for a in arrays)
    datac = concat.get_data(0)
    assert all(data1 == datac[:n1])
    assert all(data2 == datac[n1 : n1 + n2])
    assert all(data3 == datac[n1 + n2 : n1 + n2 + n3])

    inputs[1].taint()
    assert concat.tainted == True

    savegraph(graph, "output/test_Concatatenation_01.png")
