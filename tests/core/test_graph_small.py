from numpy import arange

from dag_modelling.core.graph import Graph
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.lib.common import Array
from dag_modelling.lib.arithmetic import Product, Sum


def test_graph_small(test_name, debug_graph, output_path: str):
    """Create four arrays: sum up three of them, multiply the result by the fourth
    Use graph context to create the graph.
    """
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

    s.print()
    m.print()

    result = m.outputs["result"].data
    print("Evaluation result:", result)

    savegraph(graph, f"{output_path}/{test_name}.pdf")
