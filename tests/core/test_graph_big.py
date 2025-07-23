from sys import argv

from pytest import mark

from dag_modelling.core.graph import Graph
from dag_modelling.lib.common import Dummy
from dag_modelling.plot.graphviz import GraphDot

counter = 0


@mark.skipif("--include-long-time-tests" not in argv, reason="long-time tests switched off")
def test_graph_big_01():
    """Create a graph of nodes and test evaluation features"""
    g = Graph()
    label = None

    def plot(suffix=""):
        global counter
        d = GraphDot(g)
        newlabel = label and label + suffix or suffix
        if newlabel is not None:
            d.set_label(newlabel)
        d.savegraph(f"output/test_graph_big_{counter:03d}.png")
        counter += 1

    def plotter(function, node):
        plot(f"[start evaluating {node.name}]")
        function()
        plot(f"[done evaluating {node.name}]")

    def _function(self):
        self.fd.frozen = True

    class DummyFrozen(Dummy):
        def _function(self):
            self.fd.frozen = True

    with g:
        A1 = Dummy("A1")
        A2 = DummyFrozen("A2", label="{name}|frozen")
        A3 = Dummy("A3", immediate=True, label="{name}|immediate")
        B = Dummy("B")
        C1 = Dummy("C1")
        C2 = Dummy("C2")
        D = Dummy("D")
        E = Dummy("E")
        F = Dummy("F")
        H = Dummy("H")
        P = Dummy("P", immediate=True, label="{name}|immediate")

    A1._add_output("o1", allocatable=False)
    A2._add_output("o1", allocatable=False)
    P._add_output("o1", allocatable=False)
    A3._add_pair("i1", "o1", output_kws={"allocatable": False})
    B._add_inputs(("i1", "i2", "i3", "i4"))
    B._add_outputs(("o1", "o2"), allocatable=False)
    C1._add_output("o1", allocatable=False)
    C2._add_output("o1", allocatable=False)
    D._add_pair("i1", "o1", output_kws={"allocatable": False})
    D._add_pair("i2", "o2", output_kws={"allocatable": False})
    H._add_pair("i1", "o1", output_kws={"allocatable": False})
    _, other = F._add_pair("i1", "o1", output_kws={"allocatable": False})
    _, final = E._add_pair("i1", "o1", output_kws={"allocatable": False})

    P >> A3
    (A1, A2, A3, D.outputs[0]) >> B
    B.outputs[0] >> E
    B.outputs[1] >> H
    (C1, C2) >> D
    D[:, 1] >> F

    g.print()
    g.close()

    label = "Initial graph state."
    plot()

    label = "Read E..."
    plot()
    plot()
    plot()
    final.data
    label = "Done reading E."
    plot()

    label = "Taint D."
    plot()
    plot()
    plot()
    D.taint()
    plot()
    label = "Read F..."
    other.data
    label = "Done reading F."
    plot()

    label = "Read E..."
    plot()
    plot()
    plot()
    final.data
    label = "Done reading E."
    plot()

    label = "Taint A2."
    plot()
    plot()
    plot()
    A2.taint()
    plot()
    label = "Read E..."
    plot()
    final.data
    label = "Done reading E."
    plot()

    label = "Unfreeze A2 (tainted)."
    plot()
    plot()
    plot()
    A2.unfreeze()
    plot()
    label = "Read E..."
    plot()
    final.data
    label = "Done reading E."
    plot()

    label = "Unfreeze A2 (not tainted)."
    plot()
    plot()
    plot()
    A2.unfreeze()
    plot()
    label = "Read E..."
    plot()
    final.data
    label = "Done reading E."
    plot()

    label = "Taint P"
    plot()
    plot()
    plot()
    P.taint()
    plot()
    label = "Read E..."
    plot()
    final.data
    label = "Done reading E."
    plot()

    label = "Invalidate P"
    plot()
    plot()
    plot()
    P.invalid = True
    plot()

    label = "Validate P"
    plot()
    plot()
    plot()
    P.invalid = False
    plot()
    label = "Read E..."
    plot()
    final.data
    label = "Done reading E."
    plot()

    label = "Open/Close D"
    plot()
    plot()
    plot()
    D.open()
    D.close(close_children=True)
    plot()
    plot()
    final.data
    label = "Done opening/closing D."
    plot()
