import pytest
from numpy import arange
from pytest import mark

from dag_modelling.core.graph import Graph
from dag_modelling.lib.arithmetic import Product, Sum
from dag_modelling.lib.common import Array
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.tools.graph_walker import GraphWalker, get_subgraph_nodes


@pytest.fixture
def nodes(debug_graph):
    array = arange(4)
    names = "n1", "n2", "n3", "n4"
    with Graph(debug=debug_graph) as graph:
        initials = [Array(name, array, mode="fill") for name in names]
        s0 = Sum("sum0")
        m = Product("product")
        s1 = Sum("sum1")

        initials[:-1] >> s0
        initials[-1] >> m
        s0 >> m

        (initials[-1], s0) >> s1

    graph.close()
    nodes = {name: initial for name, initial in zip(names, initials)}
    nodes.update({"sum0": s0, "product": m, "sum1": s1})
    return nodes


@mark.parametrize(
    "first_node,min_depth,max_depth,n_nodes,enable_process_full_graph",
    [
        ("sum0", 0, 0, 1, True),
        ("sum0", None, None, 6, False),
        ("sum0", None, None, 7, True),
        ("sum0", -1, 1, 6, False),
        ("sum0", -1, 1, 7, True),
        ("sum1", 0, 0, 1, True),
        ("sum1", None, None, 6, False),
        ("sum1", None, None, 7, True),
        ("sum1", -1, 1, 3, False),
        ("sum1", -1, 1, 4, True),
    ],
)
def test_graph_walker(first_node, min_depth, max_depth, n_nodes, enable_process_full_graph, nodes):
    graph_walker = GraphWalker(
        min_depth=min_depth,
        max_depth=max_depth,
        enable_process_full_graph=enable_process_full_graph,
    )
    graph_walker.process_from_node(nodes[first_node])

    for node, depth in graph_walker.nodes.items():
        n_nodes -= 1
        if min_depth is not None and depth < min_depth:
            raise RuntimeError("Too deep node")
        if max_depth is not None and depth > max_depth:
            raise RuntimeError("Too high node")

    assert n_nodes == 0, "Unnexpected number of visited nodes"


@mark.parametrize(
    "source_names,sink_names,subgraph_expect",
    [
        (["n1"], ["product"], {"n1", "sum0", "product"}),
        (["n1", "n2"], ["product"], {"n1", "n2", "sum0", "product"}),
        (["n1", "n3"], ["product"], {"n1", "n3", "sum0", "product"}),
        (["n1", "n3"], ["product", "sum1"], {"n1", "n3", "sum0", "product", "sum1"}),
    ],
)
def test_subgraphs(source_names, sink_names, subgraph_expect, nodes, output_path: str):
    gname = f"{output_path}/test_subgraphs.pdf"
    savegraph(nodes["sum0"], gname, transform_kwargs={"keep_direction": False})
    print("Write:", gname)

    sources = list(map(nodes.__getitem__, source_names))
    sinks = list(map(nodes.__getitem__, sink_names))
    subgraph = get_subgraph_nodes(sources, sinks)

    subgraph_names = set(node.name for node in subgraph)
    assert subgraph_names == subgraph_expect

