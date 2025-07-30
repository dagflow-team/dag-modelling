from numpy import array, floating, integer, linspace, newaxis
from pytest import mark, raises

from dag_modelling.core.exception import TypeFunctionError
from dag_modelling.core.graph import Graph
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.core.input_strategy import AddNewInputAddAndKeepSingleOutput
from dag_modelling.lib.common import Array, Dummy
from dag_modelling.core.type_functions import (
    AllPositionals,
    check_dimension_of_inputs,
    check_dtype_of_inputs,
    check_inputs_are_matrices_or_diagonals,
    check_shape_of_inputs,
    check_inputs_are_square_matrices,
    check_subtype_of_inputs,
    check_inputs_equivalence,
    check_inputs_are_matrix_multipliable,
    check_inputs_have_same_dtype,
    check_inputs_have_same_shape,
    check_subtype_of_outputs,
    copy_from_inputs_to_outputs,
)


@mark.parametrize(
    "data,dim,shape,dtype",
    (
        ([1, 2, 3], 1, (3,), "i"),
        ([[1, 2], [3, 4]], 2, (2, 2), "d"),
        ([[[1], [2]], [[3], [4]], [[5], [6]]], 3, (3, 2, 1), "float64"),
    ),
)
def test_check_input_common(test_name, debug_graph, data, dim, shape, dtype):
    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        arr1 = Array("arr1", array(data, dtype=dtype), mode="fill")
        node = Dummy(
            "node",
            input_strategy=AddNewInputAddAndKeepSingleOutput(output_fmt="result"),
        )
        arr1 >> node
        copy_from_inputs_to_outputs(node, 0, "result")
    check_dimension_of_inputs(node, 0, dim)
    check_shape_of_inputs(node, 0, shape)
    check_dtype_of_inputs(node, 0, dtype=dtype)
    with raises(TypeFunctionError):
        check_dimension_of_inputs(node, 0, dim + 1)
    with raises(TypeFunctionError):
        check_shape_of_inputs(node, 0, (1,))
    with raises(TypeFunctionError):
        check_dtype_of_inputs(node, 0, dtype=object)
    savegraph(graph, f"output/{test_name}.png")


@mark.parametrize("data", ([0, 1, 2], [1], [[1, 2], [1, 2, 3]], [[[], [], []]]))
def test_check_inputs_are_square_matrices_00(test_name, debug_graph, data):
    with Graph(close_on_exit=False, debug=debug_graph) as graph:
        arr1 = Array("arr1", array(data, dtype=object), mode="fill")
        arr2 = Array("arr2", array(data, dtype=object), mode="fill")
        node = Dummy(
            "node",
            input_strategy=AddNewInputAddAndKeepSingleOutput(output_fmt="result"),
        )
        arr1 >> node
        arr2 >> node
    with raises(TypeFunctionError):
        check_inputs_are_square_matrices(node, 0)
    if arr1.outputs["array"].dd.dim != 1:
        with raises(TypeFunctionError):
            check_inputs_are_matrices_or_diagonals(node, 0, check_square=True)
    savegraph(graph, f"output/{test_name}.png")


@mark.parametrize(
    "data",
    (
        linspace(0, 4, 4).reshape(2, 2),
        linspace(0, 9, 9).reshape(3, 3),
        linspace(0, 16, 16).reshape(4, 4),
    ),
)
def test_check_inputs_are_square_matrices_01(test_name, debug_graph, data):
    with Graph(close_on_exit=False, debug=debug_graph) as graph:
        arr1 = Array("arr1", array(data), mode="fill")
        node = Dummy(
            "node",
            input_strategy=AddNewInputAddAndKeepSingleOutput(output_fmt="result"),
        )
        arr1 >> node
        check_inputs_are_square_matrices(node, 0)
        check_inputs_are_matrices_or_diagonals(node, 0, check_square=True)
    savegraph(graph, f"output/{test_name}.png")


@mark.parametrize(
    "dtype,wrongarr",
    (
        ("i", array([5, 6, 7], dtype="i")),
        ("d", array([5, 6], dtype="i")),
        ("float64", array([5, 6, 7], dtype="i")),
    ),
)
def test_check_inputs_equivalence(test_name, debug_graph, dtype, wrongarr):
    # TODO: check edges and nodes
    with Graph(close_on_exit=False, debug=debug_graph) as graph:
        arr1 = Array("arr1", array([1, 2], dtype=dtype), mode="fill")
        arr2 = Array("arr2", array([3, 4], dtype=dtype), mode="fill")
        arr3 = Array("arr2", array([5, 6], dtype=dtype), mode="fill")
        node = Dummy(
            "node",
            input_strategy=AddNewInputAddAndKeepSingleOutput(output_fmt="result"),
        )
        (arr1, arr2, arr3) >> node
        check_inputs_equivalence(node)
        check_shape_of_inputs(node, AllPositionals, (2,))
        check_dtype_of_inputs(node, AllPositionals, dtype=dtype)
        check_inputs_have_same_dtype(node)
        check_inputs_have_same_shape(node)
        Array("wrong_array", wrongarr, mode="fill") >> node
        with raises(TypeFunctionError):
            check_inputs_equivalence(node)
        with raises(TypeFunctionError):
            # NOTE: at least one raises Exception, see `wrongarr`
            check_inputs_have_same_dtype(node)
            check_inputs_have_same_shape(node)
    savegraph(graph, f"output/{test_name}.png")


@mark.parametrize(
    "dtype",
    ("float64", "float32", "float16", "float", "double"),
)
def test_check_subtype(test_name, debug_graph, dtype):
    with Graph(close_on_exit=False, debug=debug_graph) as graph:
        arr1 = Array("arr1", array([1, 2], dtype=dtype), mode="fill")
        node = Dummy(
            "node",
            input_strategy=AddNewInputAddAndKeepSingleOutput(output_fmt="result"),
        )
        arr1 >> node
        check_subtype_of_inputs(node, 0, dtype=floating)
        check_subtype_of_outputs(node, "result", dtype=floating)
        with raises(TypeFunctionError):
            check_subtype_of_inputs(node, 0, dtype=integer)
        with raises(TypeFunctionError):
            check_subtype_of_outputs(node, "result", dtype=integer)
    savegraph(graph, f"output/{test_name}.png")


@mark.parametrize(
    "data1, data2",
    (
        (linspace(0, 4, 4).reshape(2, 2), linspace(0, 4, 4)),
        (linspace(0, 9, 9), linspace(0, 3, 3)[newaxis]),
        (linspace(0, 3, 3)[newaxis].T, linspace(0, 3, 3)[newaxis].T),
    ),
)
def test_check_inputs_are_matrix_multipliable_00(test_name, debug_graph, data1, data2):
    with Graph(close_on_exit=False, debug=debug_graph) as graph:
        arr1 = Array("arr1", array(data1), mode="fill")
        arr2 = Array("arr2", array(data2), mode="fill")
        node = Dummy(
            "node",
            input_strategy=AddNewInputAddAndKeepSingleOutput(output_fmt="result"),
        )
        (arr1, arr2) >> node
        with raises(TypeFunctionError):
            check_inputs_are_matrix_multipliable(node, 0, 1)
    savegraph(graph, f"output/{test_name}.png")


@mark.parametrize(
    "data1, data2",
    (
        (linspace(0, 4, 4).reshape(2, 2), linspace(0, 4, 4).reshape(2, 2)),
        (linspace(0, 9, 9).reshape(3, 3), linspace(0, 3, 3)[newaxis].T),
        (linspace(0, 3, 3)[newaxis], linspace(0, 3, 3)[newaxis].T),
    ),
)
def test_check_inputs_are_matrix_multipliable_01(test_name, debug_graph, data1, data2):
    with Graph(close_on_exit=False, debug=debug_graph) as graph:
        arr1 = Array("arr1", array(data1), mode="fill")
        arr2 = Array("arr2", array(data2), mode="fill")
        node = Dummy(
            "node",
            input_strategy=AddNewInputAddAndKeepSingleOutput(output_fmt="result"),
        )
        (arr1, arr2) >> node
        check_inputs_are_matrix_multipliable(node, 0, 1)
    savegraph(graph, f"output/{test_name}.png")
