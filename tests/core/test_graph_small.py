from numpy import arange

from dag_modelling.core.graph import Graph
from dag_modelling.core.node import Node
from dag_modelling.lib.arithmetic import Product, Sum
from dag_modelling.lib.common import Array
from dag_modelling.plot.graphviz import savegraph


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

    n1, n2, n3, n4 = initials
    assert m.tainted_above_threshold() is True
    assert m.tainted_above_threshold(1) is False

    n4.taint_below_threshold(1)
    assert m.tainted_above_threshold() is True
    assert m.tainted_above_threshold(1) is True
    assert m.tainted_above_threshold(2) is False

    assert n4.tainted_above_threshold() is True
    assert n4.tainted_above_threshold(1) is True
    assert n4.tainted_above_threshold(2) is False

    assert n3.tainted_above_threshold() is True
    assert n3.tainted_above_threshold(1) is False

    n3.taint_below_threshold(1)
    assert m.tainted_above_threshold() is True
    assert m.tainted_above_threshold(1) is True
    assert m.tainted_above_threshold(2) is False

    assert s.tainted_above_threshold() is True
    assert s.tainted_above_threshold(1) is True
    assert s.tainted_above_threshold(2) is False

    assert n4.tainted_above_threshold() is True
    assert n4.tainted_above_threshold(1) is True
    assert n4.tainted_above_threshold(2) is False

    assert n3.tainted_above_threshold() is True
    assert n3.tainted_above_threshold(1) is True
    assert n3.tainted_above_threshold(2) is False

    n4.taint_below_threshold(2)
    assert m.tainted_above_threshold(2) is True
    assert m.tainted_above_threshold(3) is False

    assert n4.tainted_above_threshold(2) is True
    assert n4.tainted_above_threshold(3) is False

    assert n3.tainted_above_threshold(1) is True
    assert n3.tainted_above_threshold(2) is False

    assert n1.tainted_above_threshold() is True
    assert n1.tainted_above_threshold(1) is False

    n3.taint_below_threshold(2)
    assert m.tainted_above_threshold(2) is True
    assert m.tainted_above_threshold(3) is False

    assert s.tainted_above_threshold(2) is True
    assert s.tainted_above_threshold(3) is False

    assert n4.tainted_above_threshold(2) is True
    assert n4.tainted_above_threshold(3) is False

    assert n3.tainted_above_threshold(2) is True
    assert n3.tainted_above_threshold(3) is False

    n4.taint_below_threshold(3)
    assert m.tainted_above_threshold(3) is True
    assert m.tainted_above_threshold(4) is False

    assert n4.tainted_above_threshold(3) is True
    assert n4.tainted_above_threshold(4) is False

    assert n3.tainted_above_threshold(2) is True
    assert n3.tainted_above_threshold(3) is False

    assert n1.tainted_above_threshold() is True
    assert n1.tainted_above_threshold(1) is False

    n1.taint_below_threshold(2)
    assert n1.tainted_above_threshold(2) is True
    assert n1.tainted_above_threshold(3) is False

    assert n2.tainted_above_threshold(0) is True
    assert n2.tainted_above_threshold(1) is False

    assert s.tainted_above_threshold(2) is True
    assert s.tainted_above_threshold(3) is False

    assert m.tainted_above_threshold(3) is True
    assert m.tainted_above_threshold(4) is False

    n1.taint_below_threshold(4)
    assert n1.tainted_above_threshold(4) is True
    assert n1.tainted_above_threshold(5) is False

    assert s.tainted_above_threshold(4) is True
    assert s.tainted_above_threshold(5) is False

    assert m.tainted_above_threshold(4) is True
    assert m.tainted_above_threshold(5) is False

    #
    # Patch functions to propagate taint threshold
    #
    def fcn(node: Node, taint_threshold: int):
        for input in node.inputs.iter_all():
            input.touch_above_threshold(taint_threshold)

    for node in initials + [s, m]:
        node.function = fcn.__get__(node)

    m.touch_above_threshold(5)
    assert m.tainted_above_threshold(4) is True
    assert m.tainted_above_threshold(5) is False

    assert n1.tainted_above_threshold(4) is True
    assert n1.tainted_above_threshold(5) is False

    assert s.tainted_above_threshold(4) is True
    assert s.tainted_above_threshold(5) is False

    m.touch_above_threshold(4)
    assert m.tainted_above_threshold(3) is True
    assert m.tainted_above_threshold(4) is False

    assert n1.tainted_above_threshold(3) is True
    assert n1.tainted_above_threshold(4) is False

    assert n2.tainted_above_threshold(0) is True
    assert n2.tainted_above_threshold(1) is False

    assert s.tainted_above_threshold(3) is True
    assert s.tainted_above_threshold(4) is False

    assert n4.tainted_above_threshold(3) is True
    assert n4.tainted_above_threshold(4) is False

    #
    # Validate sequential application
    #
    threshold = m.tainted_value

    assert m.tainted_above_threshold(threshold) is False
    m.taint_below_threshold(threshold)
    assert m.tainted_above_threshold(threshold) is True
    assert m.tainted_value == threshold+1

    threshold+=1
    assert m.tainted_above_threshold(threshold) is False
    assert m.tainted_value == threshold

    m.touch_above_threshold(threshold)
    assert m.tainted_above_threshold(threshold) is False
    assert m.tainted_value == threshold

    m.taint_below_threshold(threshold)
    assert m.tainted_above_threshold(threshold) is True
    assert m.tainted_value == threshold+1

    m.touch_above_threshold(threshold)
    assert m.tainted_above_threshold(threshold) is False
    assert m.tainted_value == threshold

    savegraph(graph, f"{output_path}/{test_name}.pdf")
