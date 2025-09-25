from numpy import array
from pytest import mark

from dag_modelling.core.graph import Graph
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.core.input_strategy import AddNewInputAddAndKeepSingleOutput
from dag_modelling.lib.common import Array
from dag_modelling.lib.common import Dummy
from dag_modelling.core.type_functions import (
    AllPositionals,
    check_edges_consistency_with_array,
    check_dtype_of_edges,
    check_edges_dimension_of_inputs,
    check_edges_equivalence_of_inputs,
    copy_from_inputs_to_outputs,
    copy_edges_from_inputs_to_outputs,
)


@mark.parametrize(
    "data,edgesdata",
    (
        ([1], [1, 2]),
        ([1, 2], [1, 2, 3]),
        ([1, 2, 3], [1, 2, 3, 4]),
    ),
)
def test_edges_00(test_name, debug_graph, data, edgesdata, output_path: str):
    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        edges = Array("edges", edgesdata, mode="fill").outputs["array"]
        arr1 = Array("arr1", array(data), edges=edges, mode="fill")
        arr2 = Array("arr2", 2 * array(data), edges=edges, mode="fill")
        node = Dummy(
            "node",
            input_strategy=AddNewInputAddAndKeepSingleOutput(output_fmt="result"),
        )
        arr1 >> node
        arr2 >> node
        copy_from_inputs_to_outputs(node, 0, "result")
        check_edges_dimension_of_inputs(node, AllPositionals)
        check_edges_equivalence_of_inputs(node, AllPositionals)
        check_dtype_of_edges(node)
        check_edges_consistency_with_array(node, "result")
        copy_edges_from_inputs_to_outputs(node, 0, "result")
    savegraph(graph, f"{output_path}/{test_name}.png")


@mark.parametrize(
    "data,edgesdataX,edgesdataY",
    (
        ([[1], [1]], [1, 2, 3], [1, 2]),
        ([[1, 2], [1, 2]], [1, 2, 3], [1, 2, 3]),
        ([[1, 2, 3], [1, 2, 3]], [1, 2, 3], [1, 2, 3, 4]),
        ([[1, 2], [1, 2], [1, 2]], [1, 2, 3, 4], [1, 2, 3]),
        ([[1, 2, 3], [1, 2, 3], [1, 2, 3]], [1, 2, 3, 4], [1, 2, 3, 4]),
    ),
)
def test_edges_01(test_name, debug_graph, data, edgesdataX, edgesdataY, output_path: str):
    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        edgesX = Array("edgesX", edgesdataX, mode="fill").outputs["array"]
        edgesY = Array("edgesY", edgesdataY, mode="fill").outputs["array"]
        edges = [edgesX, edgesY]
        arr1 = Array("arr1", array(data), edges=edges, mode="fill")
        arr2 = Array("arr2", 2 * array(data), edges=edges, mode="fill")
        node = Dummy(
            "node",
            input_strategy=AddNewInputAddAndKeepSingleOutput(output_fmt="result"),
        )
        arr1 >> node
        arr2 >> node
        copy_from_inputs_to_outputs(node, 0, "result")
        check_edges_dimension_of_inputs(node, AllPositionals)
        check_edges_equivalence_of_inputs(node, AllPositionals)
        check_dtype_of_edges(node)
        check_edges_consistency_with_array(node, "result")
        copy_edges_from_inputs_to_outputs(node, 0, "result")
    savegraph(graph, f"{output_path}/{test_name}.png")
