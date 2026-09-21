from collections import Counter

from pandas import Series
from pytest import raises

from dag_modelling.core.node import Node
from dag_modelling.lib.arithmetic import Product, Sum
from dag_modelling.lib.common import Array
from dag_modelling.tools.profiling import NodeProfiler

n_runs = 1000


def check_inputs_taint(node: Node):
    return any(inp.tainted for inp in node.inputs)


def test_init(graph_0):
    _, nodes = graph_0
    a0, a1, a2, _, _, p1, _, _, s2, s3, _, _ = nodes
    target_nodes = [a0, a1, s3, s2]
    profiling = NodeProfiler(target_nodes)
    assert Counter(profiling._target_nodes) == Counter(target_nodes)

    sources, sinks = [a1, a2], [s2]
    target_nodes = [a1, a2, p1, s2]
    profiling = NodeProfiler(sources=sources, sinks=sinks)


def test_filter_nodes(graph_0):
    _, nodes = graph_0

    node_types = (Product, Sum)
    profiling = NodeProfiler(nodes, keep_types=node_types)
    assert all(isinstance(n, node_types) for n in profiling._target_nodes)

    node_types_str = ("Product", "Sum")
    profiling = NodeProfiler(nodes, keep_types=node_types_str)
    assert all(n.__class__.__name__ in node_types_str for n in profiling._target_nodes)

    node_types_mixed = ("Array", Sum)
    profiling = NodeProfiler(nodes, keep_types=node_types_mixed)
    assert all(isinstance(n, (Array, Sum)) for n in profiling._target_nodes)


def test_estimate_node_g0(graph_0):
    _, nodes = graph_0
    print(f"(graph 0) NodeProfiler.estimate_node (n_runs={n_runs}):")
    for node in nodes:
        print(f"\t{node.name} estimated with:", NodeProfiler.estimate_node(node, n_runs))
        assert check_inputs_taint(node) == False


def test_estimate_node_g1(graph_1):
    _, nodes = graph_1
    print(f"(graph 1) NodeProfiler.estimate_node (n_runs={n_runs}):")
    for node in nodes:
        print(f"\t{node.name} estimated with:", NodeProfiler.estimate_node(node, n_runs))
        assert check_inputs_taint(node) == False


def test_estimate_target_nodes_g0(graph_0):
    g, _ = graph_0
    target_nodes = g._nodes
    profiling = NodeProfiler(target_nodes, n_runs=n_runs)
    profiling.estimate_target_nodes()
    assert hasattr(profiling, "_estimations_table")


def test_t_percentage(graph_0):
    _, _n = graph_0
    profiler = NodeProfiler(_n).estimate_target_nodes()
    some_group = Series({"time": [0, 0, 0, 0, 0]})
    profiler._estimations_table["time"] = 0
    with raises(ZeroDivisionError) as excinfo:
        profiler._t_percentage(some_group)
    assert "is zero" in str(excinfo.value)


def test_make_report_g0(graph_0):
    g, _ = graph_0
    target_nodes = g._nodes
    profiling = NodeProfiler(target_nodes, n_runs=n_runs)

    profiling.estimate_target_nodes().make_report()

    profiling.make_report(group_by=None)

    with raises(AttributeError) as excinfo:
        NodeProfiler(target_nodes).make_report()


def test_make_report_g1(graph_1):
    g, _ = graph_1
    target_nodes = g._nodes
    profiling = NodeProfiler(target_nodes, n_runs=n_runs)
    profiling.estimate_target_nodes()

    profiling.make_report()


def test_print_report_g1_1(graph_1):
    g, _ = graph_1
    target_nodes = g._nodes
    profiling = NodeProfiler(target_nodes, n_runs=n_runs)
    profiling.estimate_target_nodes()

    profiling.print_report()
    profiling.print_report(group_by=None, rows=2)
    profiling.print_report(group_by=None, rows=20)


def test_print_report_g1_2(graph_1):
    g, _ = graph_1
    target_nodes = g._nodes

    for i in range(2, 5):
        n_runs = 10**i
        profiling = NodeProfiler(target_nodes, n_runs=n_runs)
        profiling.estimate_target_nodes()
        profiling.print_report()


def test_print_report_g1_3(graph_1):
    g, _ = graph_1
    target_nodes = g._nodes
    profiling = NodeProfiler(target_nodes, n_runs=n_runs)
    profiling.estimate_target_nodes()

    profiling.print_report()
    profiling.make_report()
