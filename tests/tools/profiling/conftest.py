import numpy as np
from pytest import fixture

from dag_modelling.core.graph import Graph
from dag_modelling.core.node import Node
from dag_modelling.lib.arithmetic import Product, Sum
from dag_modelling.lib.common import Array
from dag_modelling.lib.linalg import MatrixProductDVDt


# dummy graph structures on which the profiler is being tested
@fixture
def graph_0() -> tuple[Graph, list[Node]]:
    with Graph(close_on_exit=True) as graph:
        a0 = Array("A0", [8, 7, 13])
        a1 = Array("A1", [1, 2, 4], mode="store_weak")
        a2 = Array("A2", [12, 22, 121])
        a3 = Array("A3", [4, 3, 3], mode="fill")

        p0 = Product("P0")
        p1 = Product("P1")
        s0 = Sum("S0")

        (a0, a1) >> p0
        (a1, a2) >> p1
        (a2, a3) >> s0

        l_matrix = Array("L-Matrix", [[5, 2, 1], [1, 4, 12], [31, 7, 2]])
        s1 = Sum("S1 (diagonal of S-Matrix )")
        s2 = Sum("S2")

        (p0, p1, s0) >> s1
        (p1) >> s2

        mdvdt = MatrixProductDVDt("Matrix Product DVDt")
        s3 = Sum("S3")

        l_matrix >> mdvdt.inputs["left"]
        s1 >> mdvdt.inputs["square"]
        (s1, s2) >> s3
    s3["result"].data
    mdvdt["result"].data

    nodes = [a0, a1, a2, a3, p0, p1, s0, s1, s2, s3, l_matrix, mdvdt]
    return graph, nodes


@fixture
def graph_1() -> tuple[Graph, list[Node]]:
    with Graph(close_on_exit=True) as graph:
        array_nodes = [Array(f"A{i}", np.arange(i, i + 3, dtype="f")) for i in range(5)]
        s1 = Sum("S1")
        array_nodes[:3] >> s1

        s2 = Sum("S2")
        (array_nodes[2:4]) >> s2  # ("A2", "A3") >> s2

        p1 = Product("p1")
        (array_nodes[4], s1) >> p1

        p2 = Product("p2")
        (s2, p1) >> p2
    p2["result"].data

    nodes = [*array_nodes, s1, s2, p1, p2]
    return graph, nodes
