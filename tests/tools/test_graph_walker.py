import pytest
from numpy import arange

from dag_modelling.core.graph import Graph
from dag_modelling.lib.common import Array
from dag_modelling.lib.arithmetic import Product, Sum
from dag_modelling.tools.graph_walker import GraphWalker

from pytest import mark


@pytest.fixture
def nodes(debug_graph):
    array = arange(4)
    names = "n1", "n2", "n3", "n4"
    with Graph(debug=debug_graph) as graph:
        initials = [Array(name, array, mode="fill") for name in names]
        s = Sum("add")
        m = Product("mul")

        initials[:-1] >> s
        initials[-1] >> m
        s >> m

    graph.close()
    nodes = {name: initial for name, initial in zip(names, initials)}
    nodes.update({"sum": s, "product": m})
    return nodes

@mark.parametrize(
    "min_depth,max_depth,n_nodes,enable_process_full_graph",
    [(0, 0, 1, True), (None, None, 5, False), (None, None, 6, True), (-1, 1, 6, True)]
)
def test_graph_walker(min_depth, max_depth, n_nodes, enable_process_full_graph, nodes):
    graph_walker = GraphWalker(
        min_depth=min_depth,
        max_depth=max_depth,
        enable_process_full_graph=enable_process_full_graph
    )
    graph_walker.process_from_node(nodes["sum"])

    for node, depth in graph_walker.nodes.items():
        n_nodes -= 1
        if min_depth is not None and depth < min_depth:
            raise RuntimeError("Too deep node")
        if max_depth is not None and depth > max_depth:
            raise RuntimeError("Too high node")

    assert n_nodes == 0, "Unnexpected number of visited nodes"
