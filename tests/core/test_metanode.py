from numpy import linspace, pi, sin, cos
from pytest import raises

from dag_modelling.core.exception import CriticalError
from dag_modelling.core.graph import Graph
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.lib.common import Array
from dag_modelling.lib.trigonometry import Cos, Sin
from dag_modelling.core.meta_node import MetaNode


def test_meta_node_strategy_leading_node(testname):
    data1 = linspace(0, 1, 11, dtype="d") * pi
    data2 = linspace(1, 2, 11, dtype="d") * pi
    data3 = linspace(2, 3, 11, dtype="d") * pi
    data4 = linspace(3, 4, 11, dtype="d") * pi
    with Graph(close_on_exit=True) as graph:
        meta_node = MetaNode()
        arr1 = Array("x1", data1, mode="fill")
        arr2 = Array("x2", data2, mode="fill")
        arr3 = Array("x3", data3, mode="fill")
        arr4 = Array("x4", data4, mode="fill")
        node = Cos("cos")
        arr1 >> node
        # adds an input and an output to MetaNode
        meta_node._add_node(node, inputs_pos=True, outputs_pos=True)
        # auto creation of and import of input and output
        arr2 >> meta_node()
        arr3 >> meta_node()
        arr4 >> meta_node(name="arr4", nodename="cos")
    assert len(meta_node._nodes) == 1
    assert len(meta_node.inputs) == 4
    assert len(meta_node.outputs) == 4
    res = meta_node._leading_node.outputs
    assert len(res) == 4
    assert (res[0].data == cos(data1)).all()
    assert (res[1].data == cos(data2)).all()
    assert (res[2].data == cos(data3)).all()
    assert (res[3].data == cos(data4)).all()
    savegraph(graph, f"output/{testname}.png")


def test_meta_node_strategy_new_node(testname):
    data1 = linspace(0, 1, 11, dtype="d") * pi
    data2 = linspace(1, 2, 11, dtype="d") * pi
    data3 = linspace(2, 3, 11, dtype="d") * pi
    with Graph(close_on_exit=True) as graph:
        meta_node = MetaNode(strategy="NewNode", new_node_cls=Cos)
        arr1 = Array("x1", data1, mode="fill")
        arr2 = Array("x2", data2, mode="fill")
        arr3 = Array("x3", data3, mode="fill")
        node = Cos("cos1")
        arr1 >> node
        meta_node._add_node(
            node, inputs_pos=True, outputs_pos=True
        )  # adds an input and an output to MetaNode
        arr2 >> meta_node(
            node_args={"name": "cos2"}
        )  # NOTE: creates a new Cos node and connect with arr2
        arr3 >> meta_node(
            new_node_cls=Sin, node_args={"name": "sin"}
        )  # NOTE: creates a Sin node and connect with arr3
    assert len(meta_node._nodes) == 3
    assert len(meta_node.inputs) == 3
    assert len(meta_node.outputs) == 3
    res = meta_node._nodes
    assert len(res) == 3
    assert (res[0].outputs[0].data == cos(data1)).all()
    assert (res[1].outputs[0].data == cos(data2)).all()
    assert (res[2].outputs[0].data == sin(data3)).all()
    savegraph(graph, f"output/{testname}.png")


def test_meta_node_exceptions():
    with raises(CriticalError):
        MetaNode(strategy="")
    with raises(CriticalError):
        MetaNode(strategy="LeadingNode")()
    with raises(CriticalError):
        MetaNode()(nodename="cos")
    with raises(RuntimeError):
        mn = MetaNode()
        node = Cos("cos1")
        mn._add_node(node)
        mn._add_node(node)
    with raises(RuntimeError):
        mn = MetaNode()
        node = Cos("cos1")
        node()
        mn._add_node(node, inputs_pos=True)
        mn._import_pos_inputs(node)
