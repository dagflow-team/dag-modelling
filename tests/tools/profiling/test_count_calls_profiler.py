from __future__ import annotations

from pandas import DataFrame

from dag_modelling.tools.profiling import CountCallsProfiler


def eval_n_times(node, n):
    """Evaluate node n times.

    Helper function
    """
    for _ in range(n):
        node.eval()


def get_estimated_calls(profiler, node):
    """Return the estimated number of calls for the given node.

    This is a helper function.
    Note: This method of retrieving the number of calls should never be used in
    real code, as some nodes could have the same name.
    """
    indexing = profiler._estimations_table["name"] == node.name
    estimated_calls = profiler._estimations_table.loc[indexing, "calls"]
    return estimated_calls.iloc[0]


def test_one_call(graph_0):
    _, nodes = graph_0

    cc_profiler = CountCallsProfiler(nodes)
    cc_profiler.estimate_calls()

    for node in nodes:
        est_calls = get_estimated_calls(cc_profiler, node)
        assert est_calls == 1
        assert est_calls == node.n_calls


def test_multiple_calls(graph_0):
    _, nodes = graph_0
    a0, _, _, _, p0, _, _, _, _, s3, _, mdvdt = nodes

    target_calls = [11, 3, 17, 14]
    target_nodes = [mdvdt, a0, s3, p0]
    for tc, node in zip(target_calls, target_nodes):
        eval_n_times(node, tc - 1)  # "-1" since there is one estimation already

    cc_profiler = CountCallsProfiler(nodes).estimate_calls()

    for tc, node in zip(target_calls, target_nodes):
        calls = get_estimated_calls(cc_profiler, node)
        assert calls == tc

    # check if calls of other nodes is estimated correctly
    for node in filter(lambda x: x not in target_nodes, nodes):
        calls = get_estimated_calls(cc_profiler, node)
        assert calls == 1


def test_reports(graph_0):
    _, nodes = graph_0
    _, _, _, a3, _, p1, _, _, _, _, l_matrix, _ = nodes

    eval_n_times(a3, 42)
    eval_n_times(p1, 19)
    eval_n_times(l_matrix, 9)

    cc_profiler = CountCallsProfiler(nodes).estimate_calls()
    report = cc_profiler.make_report()
    print("\n", report)

    assert isinstance(report, DataFrame)
    assert report.empty == False, "report's DataFrame is empty"

    cc_profiler.print_report(group_by=None)
    cc_profiler.print_report(sort_by="sum")
