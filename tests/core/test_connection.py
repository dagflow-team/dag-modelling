"""
The file contains tests of graph parts connections via `>>` and `<<` operators.
Names of tests refer to the corresponding operator:
`..._to_...` means ``>>`  and `..._from_...` means `<<`
"""

from collections.abc import Sequence
from copy import deepcopy

from pytest import mark, raises

from dag_modelling.core.exception import ClosedGraphError, ConnectionError, UnclosedGraphError
from dag_modelling.core.graph import Graph
from dag_modelling.core.input import Input, Inputs
from dag_modelling.core.node import Node
from dag_modelling.core.output import Output
from dag_modelling.core.storage import NodeStorage
from dag_modelling.lib.abstract import BlockToOneNode, OneToOneNode
from dag_modelling.lib.common import Dummy
from dag_modelling.parameters import Parameter
from dag_modelling.plot.graphviz import savegraph
from nested_mapping import NestedMapping


def check_connection(obj):
    """Useful method for a check whether the object is connected or not"""
    if isinstance(obj, Input):
        return obj.connected()
    if isinstance(obj, Node):
        return all(res.connected() for res in obj.inputs)
    if isinstance(obj, (Sequence, Inputs)):
        return all(check_connection(res) for res in obj)
    return False


##################################################################################
######### Operator >>
##################################################################################
@mark.parametrize("LHS", (Output("output", None), Parameter(Output("output", None), parent=None)))
@mark.parametrize(
    "RHS",
    (
        Input("input", None),
        OneToOneNode("node"),
        Inputs([Input(f"input_{i}", None) for i in range(3)]),
        [Input(f"input_{i}", None) for i in range(3)],
        [OneToOneNode(f"node_{i}") for i in range(3)],
    ),
)
def test_Output_or_Parameter_to_Input_or_Node_or_Sequence(LHS, RHS):
    """
    Test of a connection in the following cases:
      * `Output | Parameter >> Input | Node | Inputs`;
      * `Output | Parameter >> Sequence[Input | Node]`;
    """
    # NOTE: LHS and RHS are initialized only once for all the test cases,
    #       so we need to create their copies to avoid reconnections!
    lhs, rhs = deepcopy(LHS), deepcopy(RHS)
    lhs >> rhs

    assert lhs.connected()
    if isinstance(rhs, Input):
        assert check_connection(rhs)
        if isinstance(lhs, Output):
            assert rhs.parent_output == lhs
        elif isinstance(lhs, Parameter):
            assert rhs.parent_output == lhs.output
    elif isinstance(rhs, Node):
        assert check_connection(rhs)
        assert len(rhs.outputs) == 1
    elif isinstance(rhs, (Sequence, Inputs)):
        for obj in rhs:
            assert check_connection(obj)
            if isinstance(rhs, Input):
                assert rhs.parent_output == lhs
            elif isinstance(rhs, Node):
                assert len(obj.outputs) == 1


@mark.parametrize("LHS", (Output("output", None), Parameter(Output("output", None), parent=None)))
@mark.parametrize(
    "RHS",
    (
        {f"i{i}": Input(f"input_{i}", None) for i in range(3)},
        {f"i{i}": Inputs([Input(f"input_{j}{i}", None) for j in range(3)]) for i in range(3)},
        {f"i{i}": [Input(f"input_{j}{i}", None) for j in range(3)] for i in range(3)},
        {f"n{i}": OneToOneNode(f"node_{i}") for i in range(3)},
        {f"n{i}": [OneToOneNode(f"node_{j}{i}") for j in range(3)] for i in range(3)},
    ),
)
@mark.parametrize("rhscls", (dict, NestedMapping, NodeStorage))
def test_Output_or_Parameter_to_Mapping(LHS, RHS, rhscls):
    """
    Test of a connection in the following cases:
      * `Output | Parameter >> Mapping[Input | Sequence[Input] | Inputs]`;
      * `Output | Parameter >> Mapping[Node | Sequence[Node]]`.

    Here `Mapping` is `dict | NestedMapping | NodeStorage`.
    """
    # NOTE: LHS and RHS are initialized only once for all the test cases,
    #       so we need to create their copies to avoid reconnections!
    lhs, rhs = deepcopy(LHS), deepcopy(RHS)
    constructor = lambda obj: obj if isinstance(rhscls, dict) else rhscls(dic=obj)
    rhs = constructor(rhs)
    lhs >> rhs

    assert lhs.connected()
    for obj in NestedMapping(dic=rhs).walkvalues():
        assert check_connection(obj)


@mark.parametrize("lcls", (Output, Parameter))
@mark.parametrize(
    "RHS",
    (
        NodeStorage(
            dic={"storage": {"objects": {f"i_{i}": Input(f"input_{i}", None) for i in range(3)}}}
        ),
        NodeStorage(
            dic={
                "storage": {
                    "objects": {
                        f"i_{i}": Inputs([Input(f"input_{j}{i}", None) for j in range(3)])
                        for i in range(3)
                    }
                }
            }
        ),
        NodeStorage(
            dic={
                "storage": {
                    "objects": {
                        f"i_{i}": [Input(f"input_{j}{i}", None) for j in range(3)] for i in range(3)
                    }
                }
            }
        ),
        NodeStorage(
            dic={"storage": {"objects": {f"i_{i}": OneToOneNode(f"node_{i}") for i in range(3)}}}
        ),
        NodeStorage(
            dic={
                "storage": {
                    "objects": {
                        f"i_{i}": [OneToOneNode(f"node_{j}{i}") for j in range(3)] for i in range(3)
                    }
                }
            }
        ),
    ),
)
def test_NodeStorage_to_NodeStorage(lcls, RHS):
    """
    Test of a connection in the following cases:
      * `NodeStorage[Output | Parameter] >> NodeStorage[Input | Sequence[Input] | Inputs]`;
      * `NodeStorage[Output | Parameter] >> NodeStorage[Node | Sequence[Node]]`;
    """
    # NOTE: LHS and RHS are initialized only once for all the test cases,
    #       so we need to create their copies to avoid reconnections!
    constructor = (
        lambda name: Output(name, None)
        if isinstance(lcls, Output)
        else Parameter(Output(name, None), parent=None)
    )
    lhs = NodeStorage(
        dic={"storage": {"objects": {f"i_{i}": constructor(f"output_{i}") for i in range(3)}}}
    )
    rhs = deepcopy(RHS)

    lhs >> rhs

    for obj in lhs.walkvalues():
        assert obj.connected()
    for obj in rhs.walkvalues():
        assert check_connection(obj)


@mark.parametrize("lcls", (Output, Parameter))
@mark.parametrize("lclsseq", (tuple, list))
@mark.parametrize("rcls", (OneToOneNode, BlockToOneNode))
def test_Sequence_Output_or_Parameter_to_Node(lcls, lclsseq, rcls):
    """
    Test of a connection in the following cases:
      * `Sequence[Output|Parameter] >> Node`;
      * `Sequence[Output|Parameter] >> BlockToOneNode`;
    """
    constructor = (
        lambda name: Output(name, None)
        if isinstance(lcls, Output)
        else Parameter(Output(name, None), parent=None)
    )
    lhs = lclsseq(constructor(f"output_{i}") for i in range(3))
    rhs = rcls("node")
    lhs >> rhs

    assert all(obj.connected() for obj in lhs)
    assert check_connection(rhs)
    assert len(rhs.outputs) == (1 if isinstance(rhs, BlockToOneNode) else 3)


def test_NodeWith1Output_to_Node():
    """Such connections is the same as `Node.outputs[0] >> ...`"""
    n1 = Node("node1")
    n2 = Node("node2")

    n1._add_output("o1")
    n2._add_input("i1")

    n1 >> n2
    assert n2.inputs[0].connected()
    assert n1.outputs.len_all() == 1
    assert n2.inputs.len_all() == 1


@mark.parametrize("nodes_count", (1, 2, 3, 4))
def test_BlockOfNodesWith1Outputs_to_BlockToOneNode(nodes_count):
    """Such connections is the same as `(Node.outputs[0], ..., Node.outputs[0]) >> BlockToOneNode`"""
    blocknode = BlockToOneNode("BlockToOneNode")

    nodes = [Node(f"node_{i}") for i in range(nodes_count)]
    for node in nodes:
        node._add_output("o1")
    nodes >> blocknode

    assert blocknode.outputs.len_all() == 1
    assert blocknode.inputs.len_all() == nodes_count
    assert all(inp.connected() for inp in blocknode.inputs)


def test_NodeWith2Outputs_to_Node():
    """Such connections are ambiguous, so not allowed"""
    n1 = Node("node1")
    n2 = Node("node2")

    n1._add_output("o1")
    n1._add_output("o2")

    n2._add_input("i1")
    n2._add_input("i2")
    n2._add_output("o1")

    with raises(ConnectionError):
        n1 >> n2


##################################################################################
######### Operator <<
##################################################################################
@mark.parametrize("lcls", (OneToOneNode, BlockToOneNode))
@mark.parametrize("rcls", (Output, Parameter))
def test_Node_from_NodeStorage(lcls, rcls):
    """
    Test of a connection in the following cases:
      * `Node << NodeStorage[Output|Parameter]`;
      * `Node << NodeStorage[Output|Parameter]`;
    """
    constructor = (
        lambda name: Output(name, None)
        if isinstance(rcls, Output)
        else Parameter(Output(name, None), parent=None)
    )
    lhs = lcls(name="node")
    lhs._add_input("arr_1")
    lhs._add_input("arr_2")
    rhs = NodeStorage(dic={f"arr_{i}": constructor(name=f"arr_{i}") for i in range(3)})
    lhs << rhs

    for obj in rhs.walkvalues():
        if isinstance(obj, Parameter):
            name = obj._connectible_output.name
            if name == "arr_0":
                assert not obj.connected()
                with raises(KeyError):
                    lhs.inputs[name]
                continue
            assert obj.connected() and obj._connectible_output.child_inputs[0] == lhs.inputs[name]
        elif isinstance(obj, Output):
            name = obj.name
            if name == "arr_0":
                assert not obj.connected()
                with raises(KeyError):
                    lhs.inputs[name]
                continue
            assert obj.connected() and obj.child_inputs[0] == lhs.inputs[name]
    assert check_connection(lhs)


@mark.parametrize("lclsseq", (tuple, list))
@mark.parametrize("rcls", (Output, Parameter))
@mark.parametrize("add_inputs", (True, False))
def test_Sequence_Node_from_NodeStorage(lclsseq, rcls, add_inputs):
    """
    Test of a connection in the following cases:
      * `Sequence[Node] << NodeStorage[Output|Parameter]`;
      * `Sequence[Node] << NodeStorage[Output|Parameter]`;
    """
    constructor = (
        lambda name: Output(name, None)
        if isinstance(rcls, Output)
        else Parameter(Output(name, None), parent=None)
    )
    lhs = lclsseq(OneToOneNode(name=f"node_{i}") for i in range(3))
    if add_inputs:
        for node in lhs:
            for i in range(3):
                node._add_input(f"arr_{i}")
    rhs = NodeStorage(dic={f"arr_{i}": constructor(name=f"arr_{i}") for i in range(3)})
    lhs << rhs

    if add_inputs:
        assert all(obj.connected() for obj in rhs.walkvalues())
        assert check_connection(lhs)
    else:
        assert all(not obj.connected() for obj in rhs.walkvalues())
        assert all(node.inputs.len_all() == 0 for node in lhs)
    assert all(node.outputs.len_all() == 0 for node in lhs)


##################################################################################
######### Other old tests
##################################################################################
def test_05():
    n1 = Dummy("node1")
    n2 = Dummy("node2")

    out1 = n1._add_output("o1", allocatable=False)
    out2 = n1._add_output("o2", allocatable=False)

    _, final = n2._add_pair("i1", "o1", output_kws={"allocatable": False})
    n2._add_input("i2")

    (out1, out2) >> n2

    n2.close()
    assert n2.closed
    assert n1.closed
    with raises(ClosedGraphError):
        n2.add_input("i3")
    with raises(ClosedGraphError):
        n1.add_output("o3")
    final.data


def test_06():
    n1 = Dummy("node1")
    n2 = Dummy("node2")

    out1 = n1._add_output("o1", allocatable=False)
    out2 = n1._add_output("o2", allocatable=False)

    _, final = n2._add_pair("i1", "o1", output_kws={"allocatable": False})
    n2._add_input("i2")

    (out1, out2) >> n2

    n1.close(close_parents=False)
    assert n1.closed
    assert not n2.closed
    n2.close(close_parents=False)
    assert n2.closed
    with raises(ClosedGraphError):
        n2.add_input("i3")
    with raises(ClosedGraphError):
        n1.add_output("o3")
    final.data


def test_07():
    with Graph() as g:
        n1 = Dummy("node1")
        n2 = Dummy("node2")

        out1 = n1._add_output("o1", allocatable=False)
        out2 = n1._add_output("o2", allocatable=False)

        _, final = n2._add_pair("i1", "o1", output_kws={"allocatable": False})
        n2._add_input("i2")

        (out1, out2) >> n2

    with raises(UnclosedGraphError):
        final.data
    g.close()
    with raises(ClosedGraphError):
        n2.add_input("i3")
    with raises(ClosedGraphError):
        n1.add_output("o3")
    final.data


def test_08():
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

    with raises(UnclosedGraphError):
        final.data
    g.close()
    with raises(ClosedGraphError):
        n2.add_input("i3")
    with raises(ClosedGraphError):
        n1.add_output("o3")
    with raises(ClosedGraphError):
        n3.add_pair("i3", "o3")
    final.data

    print()
    final.data

    print("Taint n2")
    n2.taint()
    final.data

    print("Taint n3")
    n3.taint()
    final.data


def test_09(test_name, output_path: str):
    """Test <<"""
    with Graph(close_on_exit=True) as g:
        n1 = Dummy("node1")
        n2 = Dummy("node2")

        out11 = n1._add_output("o1", allocatable=False)
        out12 = n1._add_output("o2", allocatable=False)
        out13 = n1._add_output("o3", allocatable=False)

        n2._add_input("i1")
        in22 = n2._add_input("i2")
        n2._add_input("i3")
        out2 = n2._add_output("o2", allocatable=False)

        out12 >> in22

        with raises(ConnectionError):
            n2 << {"i1": object()}

        with raises(ConnectionError):
            n2 << {"i2": object()}

        n2 << {"i1": out11, "i2": out12, "i3": out13}

    out2.data

    savegraph(g, f"{output_path}/{test_name}.pdf")
