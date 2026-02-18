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
        s = Sum("sum")
        m = Product("product")
        s2 = Sum("sum2")

        initials[:-1] >> s
        initials[-1] >> m
        s >> m

        (initials[-1], s) >> s2

    graph.close()
    nodes = {name: initial for name, initial in zip(names, initials)}
    nodes.update({"sum": s, "product": m, "sum2": s2})
    return nodes


@mark.parametrize(
    "first_node,min_depth,max_depth,n_nodes,enable_process_full_graph",
    [
        ("sum", 0, 0, 1, True),
        ("sum", None, None, 6, False),
        ("sum", None, None, 7, True),
        ("sum", -1, 1, 6, False),
        ("sum", -1, 1, 7, True),
        ("sum2", 0, 0, 1, True),
        ("sum2", None, None, 6, False),
        ("sum2", None, None, 7, True),
        ("sum2", -1, 1, 3, False),
        ("sum2", -1, 1, 4, True),
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
        (["n1"], ["product"], {"n1", "sum", "product"}),
        (["n1", "n2"], ["product"], {"n1", "n2", "sum", "product"}),
        (["n1", "n3"], ["product"], {"n1", "n3", "sum", "product"}),
        (["n1", "n3"], ["product", "sum2"], {"n1", "n3", "sum", "product", "sum2"}),
    ],
)
def test_subgraphs(source_names, sink_names, subgraph_expect, nodes, output_path: str):
    gname = f"{output_path}/test_subgraphs.pdf"
    savegraph(nodes["sum"], gname, transform_kwargs={"keep_direction": False})
    print("Write:", gname)

    sources = list(map(nodes.__getitem__, source_names))
    sinks = list(map(nodes.__getitem__, sink_names))
    subgraph = get_subgraph_nodes(sources, sinks)

    subgraph_names = set(node.name for node in subgraph)
    assert subgraph_names == subgraph_expect

