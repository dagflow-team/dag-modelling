from dag_modelling.core.graph import Graph
from dag_modelling.plot.graphviz import GraphDot
from dag_modelling.lib.common import Dummy


def test_01(output_path: str):
    """Simple test of the graph plotter"""
    with Graph() as g:
        n1 = Dummy("node1")
        n2 = Dummy("node2")
        n3 = Dummy("node3")

    out1 = n1._add_output("o1", allocatable=False)
    out2 = n1._add_output("o2", allocatable=False)

    _, out3 = n2._add_pair("i1", "o1", output_kws={"allocatable": False})
    n2._add_input("i2")
    n3._add_pair("i1", "o1", output_kws={"allocatable": False})

    print(f"{out1=}, {out2=}")
    (out1, out2) >> n2
    out3 >> n3
    g.close()

    d = GraphDot(g)
    d.savegraph(f"{output_path}/test1_00.png")


def test_02(output_path: str):
    """Simple test of the graph plotter"""
    with Graph() as g:
        n1 = Dummy("node1")
        n2 = Dummy("node2")
        n3 = Dummy("node3")

    out1 = n1._add_output("o1", allocatable=False)
    out2 = n1._add_output("o2", allocatable=False)

    _, out3 = n2._add_pair("i1", "o1", output_kws={"allocatable": False})
    n2._add_input("i2")

    _, final = n3._add_pair("i1", "o1", output_kws={"allocatable": False})

    (out1, out2) >> n2
    out3 >> n3
    g.close()

    d = GraphDot(g)
    d.savegraph(f"{output_path}/test2_00.png")

    final.data
    d = GraphDot(g)
    d.savegraph(f"{output_path}/test2_01.png")


def test_02a(output_path: str):
    """Simple test of the graph plotter"""
    with Graph() as g:
        n1 = Dummy("node1")
        n2 = Dummy("node2")
        n3 = Dummy("node3")
        n4 = Dummy("node4")

    out1 = n1._add_output("o1", allocatable=False)

    in2, _ = n2._add_pair("i1", "o1", output_kws={"allocatable": False})
    in3, _ = n3._add_pair("i1", "o1", output_kws={"allocatable": False})
    in4, out4 = n4._add_pair("i1", "o1", output_kws={"allocatable": False})

    out1 >> (in2, in3, in4)
    g.close()

    d = GraphDot(g)
    d.savegraph(f"{output_path}/test2a_00.png")

    print(out4.data)
    d = GraphDot(g)
    d.savegraph(f"{output_path}/test2a_01.png")
