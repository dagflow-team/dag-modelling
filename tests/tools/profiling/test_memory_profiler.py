from __future__ import annotations

from functools import reduce
from itertools import chain
from operator import mul
from typing import TYPE_CHECKING

from numpy import ndarray
from pandas import DataFrame

from dag_modelling.core.input import Input
from dag_modelling.core.output import Output
from dag_modelling.tools.profiling import MemoryProfiler

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _calc_numpy_size(data: NDArray) -> int:
    """Size of Numpy's `NDArray` in bytes."""
    length = reduce(mul, data.shape)
    return length * data.dtype.itemsize


def get_input_size(inp: Input) -> int:
    if inp.owns_buffer:
        return _calc_numpy_size(inp.own_data)
    return 0


def get_output_size(out: Output) -> int:
    if out.owns_buffer or (out.has_data and out._allocating_input is None):
        return _calc_numpy_size(out._data)
    return 0


def edge_size(edge: Output | Input | NDArray) -> int:
    """Return size of `edge` data in bytes."""
    if isinstance(edge, Input):
        return get_input_size(edge)
    elif isinstance(edge, Output):
        return get_output_size(edge)
    elif isinstance(edge, ndarray):
        return _calc_numpy_size(edge)

    raise RuntimeError("Invalid type")


def test_basic_edges_g0(graph_0):
    _, nodes = graph_0
    _, _, a2, _, _, _, _, s1, _, _, _, _ = nodes

    # A2 (1 output)
    out: Output = a2.outputs["array"]  # 'array' - default name for Array output

    actual_sizes = MemoryProfiler.estimate_node(a2)
    o_expected = edge_size(out)
    assert o_expected != 0
    assert o_expected == actual_sizes[1][out][0]

    # S1 (3 inputs, 2 outputs)
    actual_sizes = MemoryProfiler.estimate_node(s1)

    for inp in s1.inputs.iter_all():
        i_expected = edge_size(inp)
        assert i_expected == actual_sizes[0][inp][0]

    for out in s1.outputs.iter_all():
        o_expected = edge_size(out)
        assert o_expected == actual_sizes[1][out][0]


def test_array_store_mods_g0(graph_0):
    """Test profiling behavior with different array store modes."""
    _, nodes = graph_0
    _, a1, _, a3, _, p1, s0, _, _, _, _, _ = nodes

    # P1 (2 inputs, 2 outputs)
    #  parent output from 'A1' has Array node type with 'store_weak' mode
    sw_out: Output = a1.outputs["array"]
    conn_input: Input = p1.inputs[0]

    assert conn_input.parent_output == sw_out
    assert sw_out._allocating_input is None

    assert edge_size(conn_input) == 0
    assert edge_size(sw_out) != edge_size(conn_input)

    actual_sizes = MemoryProfiler.estimate_node(a1)
    sizes_outputs = actual_sizes[1]
    assert edge_size(sw_out) == sizes_outputs[sw_out][0]

    actual_sizes = MemoryProfiler.estimate_node(p1)
    sizes_inputs = actual_sizes[0]
    assert edge_size(conn_input) == sizes_inputs[conn_input][0]

    # S0 (2 inputs, 1 output)
    #  parent output from 'A3' has Array node type with 'fill' mode
    f_out: Output = a3.outputs["array"]
    conn_input: Input = s0.inputs[1]

    assert conn_input.parent_output == f_out


def test_estimate_all_edges(graph_0, graph_1):
    for graph in (graph_0, graph_1):
        _, nodes = graph

        mp = MemoryProfiler(nodes)
        mp.estimate_target_nodes()

        assert hasattr(mp, "_estimations_table")

        # check if "size" column exists and it is not empty
        assert len(mp._estimations_table["n_bytes"]) > 0

        expected = 0
        for node in nodes:
            for edge in chain(
                node.inputs.iter_all(), node.outputs.iter_all(), node._internal_buffers.values()
            ):
                expected += edge_size(edge)
        actual = mp._estimations_table["n_bytes"].sum()

        assert expected == actual, "expected and actual sizes of all edges does not match"


def test_total_size_property(graph_0):
    _, nodes = graph_0

    mp = MemoryProfiler(nodes)
    mp.estimate_target_nodes()

    assert sum(mp._estimations_table["n_bytes"]) == mp.total_size


def test_make_report(graph_0):
    _, nodes = graph_0

    mp = MemoryProfiler(nodes).estimate_target_nodes()
    report = mp.make_report()

    assert isinstance(report, DataFrame)


def test_print_report_g0(graph_0):
    _, nodes = graph_0

    mp = MemoryProfiler(nodes).estimate_target_nodes()
    mp.print_report()

    mp = MemoryProfiler(nodes).estimate_target_nodes()
    mp.print_report()


def test_chain_methods_g1(graph_1):
    _, nodes = graph_1

    report = MemoryProfiler(nodes).estimate_target_nodes().print_report()
    assert isinstance(report, DataFrame)

    report = MemoryProfiler(nodes).estimate_target_nodes().make_report()
    assert isinstance(report, DataFrame)
