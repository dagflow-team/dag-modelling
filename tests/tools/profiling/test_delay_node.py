from sys import argv

from pytest import mark

from dag_modelling.core.graph import Graph
from dag_modelling.tools.profiling import DelayNode, FrameworkProfiler, NodeProfiler

EPS = 0.05


@mark.skipif("--include-long-time-tests" not in argv, reason="long-time tests switched off")
def test_one_delay_node():
    with Graph(close_on_exit=True) as graph:
        sl = DelayNode("SL0", sleep_time=0.25)
    sl["result"].data

    profiling = NodeProfiler(graph._nodes, n_runs=4)
    res = profiling.estimate_node(sl, n_runs=4)
    print("SL0 (must be ≈ 1):", res)
    profiling.estimate_target_nodes()
    report = profiling.print_report()


def _gen_graph(sleep_time: float):
    with Graph(close_on_exit=True) as graph:
        sl0 = DelayNode("SL0", sleep_time=sleep_time)
        sl1 = DelayNode("SL1", sleep_time=sleep_time)
        sl2 = DelayNode("SL2", sleep_time=sleep_time)
        (sl0, sl1) >> sl2
    sl2["result"].data
    return graph, [sl0, sl1, sl2]


@mark.skipif("--include-long-time-tests" not in argv, reason="long-time tests switched off")
def test_three_delay_nodes():
    for sleep_t in (0.1, 0.25, 0.5):
        g, nodes = _gen_graph(sleep_time=sleep_t)
        print("\nsleep_time =", sleep_t)

        profiling = NodeProfiler(nodes, n_runs=5)
        profiling.estimate_target_nodes()

        report = profiling.make_report(group_by=None)
        assert all(abs(report["time"] - sleep_t) < EPS)
        profiling.print_report(group_by=None)

        fprofiling = FrameworkProfiler(nodes)
        fprofiling.estimate_framework_time()

        report = fprofiling.make_report(group_by=None)
        assert all(report["time"] < EPS)
        fprofiling.print_report()
