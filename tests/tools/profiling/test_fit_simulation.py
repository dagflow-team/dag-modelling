from pandas import DataFrame
from pytest import raises

from dag_modelling.tools.profiling import FitSimulationProfiler


def test_estimate_separately(graph_0):
    _, nodes = graph_0

    # see graph structure in test/output/test_profiling_graph_0.png
    a0, a1, a2, _, _, _, _, _, _, s3, _, mdvdt = nodes
    params = [a0, a1, a2]
    endpoints = [s3, mdvdt]

    fit_profiling = FitSimulationProfiler(
        mode="parameter-wise", parameters=params, endpoints=endpoints, n_runs=1000
    )

    fit_profiling.estimate_fit()
    report = fit_profiling.make_report()

    assert isinstance(report, DataFrame)
    assert not report.empty

    report = fit_profiling.print_report()

    assert isinstance(report, DataFrame)
    assert not report.empty


def test_estimate_simultaneous(graph_0):
    _, nodes = graph_0

    # see graph structure in test/output/test_profiling_graph_0.png
    a0, a1, _, _, _, _, _, _, _, s3, l_matrix, mdvdt = nodes
    params = [a0, a1, l_matrix]
    endpoints = [s3, mdvdt]

    fit_profiling = FitSimulationProfiler(
        mode="simultaneous", parameters=params, endpoints=endpoints, n_runs=1000
    )

    fit_profiling.estimate_fit()
    report = fit_profiling.make_report()

    assert isinstance(report, DataFrame)
    assert not report.empty

    report = fit_profiling.print_report()

    assert isinstance(report, DataFrame)
    assert not report.empty


def test_init(graph_0):
    _, nodes = graph_0

    # see graph structure in test/output/test_profiling_graph_0.png
    a0, a1, _, _, _, _, _, _, _, s3, _, mdvdt = nodes

    with raises(ValueError, match="one parameter and at least one endpoint"):
        FitSimulationProfiler()

    with raises(ValueError, match="Unknown mode"):
        FitSimulationProfiler(
            mode="some non-existring mode", parameters=[a0, a1], endpoints=[s3, mdvdt]
        )
