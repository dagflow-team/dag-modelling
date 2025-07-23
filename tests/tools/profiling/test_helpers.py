import numpy as np

from dag_modelling.plot.graphviz import GraphDot


def test_exec_graph_0(graph_0):
    _, nodes = graph_0
    s3, mdvdt = nodes[-3], nodes[-1]
    s3_data = s3.outputs["result"]._data
    mdvdt_data = mdvdt.outputs["result"]._data
    # check for all zeros
    assert np.any(s3_data), "graph_0, `s3` was not evaluated"
    assert np.any(mdvdt_data), "graph_0, `mdvdt` was not evaluated"


def test_exec_graph_1(graph_1):
    _, nodes = graph_1
    p2 = nodes[-1]
    p2_data = p2.outputs["result"]._data
    assert np.any(p2_data), "graph_1, `p2` was not evaluated"


def test_invoke_and_save(graph_0, graph_1):
    graphs = [graph_0, graph_1]
    for i, g in enumerate(graphs):
        graph_dot = GraphDot(g[0])
        graph_dot.savegraph(f"output/test_profiling_graph_{i}.png")
