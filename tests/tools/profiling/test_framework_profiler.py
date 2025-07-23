from types import MethodType
from collections import Counter

from dag_modelling.tools.profiling import FrameworkProfiler


def test_init_g0(graph_0):
    _, nodes = graph_0
    _, a1, a2, _, p0, p1, s0, s1, s2, s3, _, _ = nodes

    target_nodes = [a1, a2, s0, p1, p0, s1, s2, s3]
    profiling = FrameworkProfiler(target_nodes, n_runs=123)
    assert Counter(profiling._target_nodes) == Counter(target_nodes)
    assert profiling._n_runs == 123

    sources, sinks = [a1, a2], [s3]
    profiling = FrameworkProfiler(sources=sources, sinks=sinks)
    profiling.estimate_framework_time()
    profiling.print_report()
    assert Counter(profiling._target_nodes) == Counter(target_nodes)


def test_reveal_source_sink_g0(graph_0):
    _, nodes = graph_0
    a0, a1, a2, a3, _, _, _, s1, s2, s3, l_matrix, mdvdt = nodes

    sources, sinks = [a0, a1, a2, a3, l_matrix], [mdvdt, s3]
    profiling = FrameworkProfiler(nodes)
    assert Counter(profiling._sources) == Counter(sources)
    assert Counter(profiling._sinks) == Counter(sinks)

    target_nodes = [s1, s2, mdvdt, s3]
    sources, sinks = [s1, s2], [mdvdt, s3]
    profiling = FrameworkProfiler(target_nodes)
    assert Counter(profiling._sources) == Counter(sources)
    assert Counter(profiling._sinks) == Counter(sinks)


def test_taint_nodes_g0(graph_0):
    _, nodes = graph_0

    profiling = FrameworkProfiler(nodes)
    profiling._taint_nodes()

    assert all(n.tainted for n in nodes)


def test_make_fcns_empty_g0(graph_0):
    _, nodes = graph_0
    _, a1, a2, _, _, p1, _, _, s2, s3, _, _ = nodes

    profiling = FrameworkProfiler(nodes)
    profiling._set_functions_empty()
    assert all(n.function == MethodType(FrameworkProfiler.function_stub, n) for n in nodes)

    profiling._taint_nodes()
    assert a2.tainted == a1.tainted == p1.tainted == s2.tainted == True
    s2.touch()
    assert a2.tainted == a1.tainted == False
    assert p1.tainted == False
    assert s2.tainted == False

    assert s3.tainted == True


def test_underscore_estimate_framework_time_g0(graph_0):
    _, nodes = graph_0

    original_fcns = [n.function for n in nodes]
    profiling = FrameworkProfiler(nodes, n_runs=1000)

    results = profiling._estimate_framework_time()
    assert len(results) == profiling._n_runs

    final_fcns = [n.function for n in profiling._target_nodes]
    assert final_fcns == original_fcns


def test_estimate_framework_time_g0(graph_0):
    _, nodes = graph_0

    FrameworkProfiler(nodes).estimate_framework_time()

    profiling = FrameworkProfiler(nodes)
    profiling.estimate_framework_time()
    profiling.estimate_framework_time()
    profiling.estimate_framework_time()


def test_print_report_g0(graph_0):
    _, nodes = graph_0

    profiling = FrameworkProfiler(nodes, n_runs=1000)
    profiling.estimate_framework_time().print_report()
    profiling.print_report()

    profiling.print_report(group_by=None)
    profiling.print_report(aggregations=["min", "max", "count"])


def test_print_report_g1(graph_1):
    _, nodes = graph_1

    profiling = FrameworkProfiler(nodes, n_runs=1000)
    profiling.estimate_framework_time()
    profiling.print_report(aggregations=["single", "sum", "count"])


def test_single_by_node_g0(graph_0):
    _, nodes = graph_0

    profiling = FrameworkProfiler(nodes, n_runs=1500)
    profiling.estimate_framework_time()
    profiling.print_report(aggregations=["count", "t_single", "single_by_node"])
    profiling.print_report(aggregations=["count", "single", "t_mean_by_node"])
