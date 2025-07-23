from pytest import raises

from dag_modelling.core.exception import TypeFunctionError
from dag_modelling.core.graph import Graph
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.lib.arithmetic import Sum
from dag_modelling.lib.common import Array
from dag_modelling.lib.common import Dummy
from dag_modelling.core.type_functions import check_node_has_inputs, check_number_of_inputs


def test_inputs_number_00(debug_graph):
    with Graph(close_on_exit=True, debug=debug_graph):
        node = Dummy("node")
    with raises(TypeFunctionError):
        check_node_has_inputs(node)
    check_number_of_inputs(node, 0)


def test_inputs_number_01(testname, debug_graph):
    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        arr1 = Array("arr1", [1, 2, 3], mode="fill")
        arr2 = Array("arr2", [3, 2, 1], mode="fill")
        ssum = Sum("sum")
        (arr1, arr2) >> ssum
    check_node_has_inputs(ssum, (0, 1))
    check_number_of_inputs(ssum, 2)
    with raises(TypeFunctionError):
        check_node_has_inputs("arr3")
    savegraph(graph, f"output/{testname}.png")
