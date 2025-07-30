#!/usr/bin/env python

from numpy import allclose, arange, log

from dag_modelling.core.graph import Graph
from dag_modelling.lib.common import Array
from dag_modelling.lib.statistics import LogPoissonRatio
from dag_modelling.plot.graphviz import savegraph


def test_LogPoissonRatio_01(debug_graph, test_name, output_path: str):
    n = 10
    start = 10
    offset = 1.0
    dataArr = arange(start, start + n, dtype="d")
    theoryArr = dataArr + offset

    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        data = Array("data", dataArr, mark="Data", mode="fill")
        theory = Array("theory", theoryArr, mark="Theory", mode="fill")
        logPoisson = LogPoissonRatio("lr")
        (data, theory) >> logPoisson

    res = logPoisson.outputs["result"].data[0]
    truth = 2.0 * ((theoryArr - dataArr) + dataArr * log(dataArr / theoryArr)).sum()
    assert allclose(res, truth, atol=0, rtol=0)

    savegraph(graph, f"{output_path}/{test_name}.png")
