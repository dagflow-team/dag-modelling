"""Test missing input handlers"""

from pytest import mark, raises

from dag_modelling.core.graph import Graph
from dag_modelling.core.input_strategy import (
    AddNewInput,
    AddNewInputAddAndKeepSingleOutput,
    AddNewInputAddNewOutput,
    AddNewInputAddNewOutputForBlock,
    AddNewInputAddNewOutputForNInputs,
    InputStrategyBase,
)
from dag_modelling.lib.common import Dummy
from dag_modelling.plot.graphviz import savegraph


@mark.parametrize("strategy", (None, InputStrategyBase))
def test_InputStrategyBase(strategy, test_name, output_path: str):
    """Test default strategy: fail on connect"""
    with Graph() as graph:
        nodes = [Dummy(f"n{i}") for i in range(4)]
        for node in nodes:
            node.add_output("o1", allocatable=False)

            s = Dummy("add", input_strategy=strategy)
    graph.close()

    with raises(RuntimeError):
        nodes[0].outputs >> s
    with raises(RuntimeError):
        ([out for out in node.outputs] for node in nodes[:-1]) >> s
    with raises(RuntimeError):
        nodes[0].outputs[0] >> s
    with raises(RuntimeError):
        (node.outputs[0] for node in nodes[:-1]) >> s
    savegraph(graph, f"{output_path}/{test_name}.pdf", label="Fail on connect")


def test_AddNewInput(test_name, output_path: str):
    """Test AddNewInput strategy: add new input on each new connect"""
    with Graph() as graph:
        nodes = [Dummy(f"n{i}") for i in range(4)]
        for node in nodes:
            node.add_output("o1", allocatable=False)

        s = Dummy(
            "add",
            input_strategy=AddNewInput(output_kws={"allocatable": False}),
        )

    (node.outputs[0] for node in nodes[:-1]) >> s
    (out for out in nodes[-1].outputs) >> s

    print()
    print(test_name)
    s.print()
    graph.close()

    savegraph(graph, f"{output_path}/{test_name}.pdf", label="Add only inputs")


def test_AddNewInputAddNewOutput(test_name, output_path: str):
    """
    Test AddNewInputAddNewOutput strategy: add new input on each new connect
    and connect them as inputs to another input
    """
    with Graph() as graph:
        nodes = [Dummy(f"n{i}") for i in range(4)]
        for node in nodes:
            node.add_output("o1", allocatable=False)

        s = Dummy(
            "add",
            input_strategy=AddNewInputAddNewOutput(output_kws={"allocatable": False}),
        )

    (node.outputs[0] for node in nodes[:-1]) >> s
    (out for out in nodes[-1].outputs) >> s

    print()
    print(test_name)
    s.print()

    for input, output in zip(s.inputs, s.outputs):
        assert input.child_output is output
    graph.close()

    savegraph(
        graph,
        f"{output_path}/{test_name}.pdf",
        label="Add inputs and an output for each input",
    )


@mark.parametrize("child_output", (False, True))
def test_AddNewInputAddAndKeepSingleOutput(test_name, child_output, output_path: str):
    """
    Test AddNewInputAddAndKeepSingleOutput strategy: add new input on each new connect and
    add an output if needed
    """
    with Graph() as graph:
        nodes = [Dummy(f"n{i}") for i in range(4)]
        for node in nodes:
            node.add_output("o1", allocatable=False)

        s = Dummy(
            "add",
            input_strategy=AddNewInputAddAndKeepSingleOutput(
                add_child_output=child_output, output_kws={"allocatable": False}
            ),
        )

    (node.outputs[0] for node in nodes[:-1]) >> s
    (out for out in nodes[-1].outputs) >> s

    print()
    print(test_name)
    s.print()

    assert len(s.outputs) == 1
    if child_output:
        output = s.outputs[0]
        for input in s.inputs:
            assert input.child_output is output
    graph.close()

    savegraph(
        graph,
        f"{output_path}/{test_name}.pdf",
        label="Add only inputs and only one output",
    )


@mark.parametrize("n_blocks", (1, 2, 3, 4))
@mark.parametrize("child_output", (False, True))
def test_AddNewInputAddNewOutputForBlock(test_name, child_output, n_blocks, output_path: str):
    """
    Test AddNewInputAddNewOutputForBlock strategy: add new input on each new connect and
    add an output for each >> group
    """
    with Graph() as graph:
        nodes = [Dummy(f"n{i}") for i in range(4)]
        for node in nodes:
            node.add_output("o1", allocatable=False)

        s = Dummy(
            "add",
            input_strategy=AddNewInputAddNewOutputForBlock(
                add_child_output=child_output, output_kws={"allocatable": False}
            ),
        )

    match n_blocks:
        case 1:
            (node.outputs[0] for node in nodes) >> s
        case 2:
            (node.outputs[0] for node in nodes[:2]) >> s
            (node.outputs[0] for node in nodes[2:]) >> s
        case 3:
            for node in nodes[:2]:
                node.outputs[0] >> s
            (node.outputs[0] for node in nodes[2:]) >> s
        case 4:
            for node in nodes:
                node.outputs[0] >> s
        case _:
            pass

    print()
    print(test_name)
    s.print()

    assert len(s.outputs) == n_blocks
    if child_output:
        match n_blocks:
            case 1:
                for inp in s.inputs:
                    assert inp.child_output is s.outputs[0]
            case 2:
                for inp in s.inputs[:2]:
                    assert inp.child_output is s.outputs[0]
                for inp in s.inputs[2:]:
                    assert inp.child_output is s.outputs[1]
            case 3:
                assert s.inputs[0].child_output is s.outputs[0]
                assert s.inputs[1].child_output is s.outputs[1]
                for inp in s.inputs[2:]:
                    assert inp.child_output is s.outputs[2]
            case 4:
                for inp, out in zip(s.inputs, s.outputs):
                    assert inp.child_output is out
            case _:
                pass
    graph.close()

    savegraph(
        graph,
        f"{output_path}/{test_name}.pdf",
        label="Add inputs and an output for each block",
    )


@mark.parametrize("n_inp", (1, 2, 4))
@mark.parametrize("child_output", (False, True))
@mark.parametrize("starts_from_0", (False, True))
def test_AddNewInputAddNewOutputForNInputs(test_name, child_output, n_inp, starts_from_0, output_path: str):
    """
    Test AddNewInputAddNewOutputForNInputs strategy: add new input on each new connect and
    add an output for each n inputs in >> operator
    """
    with Graph() as graph:
        nodes = [Dummy(f"n{i}") for i in range(4)]
        for node in nodes:
            node.add_output("o1", allocatable=False)

        s = Dummy(
            "add",
            input_strategy=AddNewInputAddNewOutputForNInputs(
                n=n_inp,
                starts_from_0=starts_from_0,
                add_child_output=child_output,
                output_kws={"allocatable": False},
            ),
        )

    (node.outputs[0] for node in nodes) >> s
    assert len(s.outputs) == int(4 / n_inp)

    print()
    print(test_name)
    s.print()

    if child_output:
        match n_inp:
            case 1:
                for inp, out in zip(s.inputs, s.outputs):
                    assert inp.child_output is out
            case 2:
                # TODO: Now it starts from `0`-input -> there is a shift in 1 element.
                #       Instead of 2 and 4 we get 0 and 2. Is it correct?
                if starts_from_0:
                    assert s.inputs[0].child_output is s.outputs[0]
                    assert s.inputs[2].child_output is s.outputs[1]
                else:
                    assert s.inputs[1].child_output is s.outputs[0]
                    assert s.inputs[3].child_output is s.outputs[1]
            case 4:
                # TODO: Now it starts from `0`-input.
                #       Instead of the adding output within the last input. Is it correct?
                if starts_from_0:
                    assert s.inputs[0].child_output is s.outputs[-1]
                else:
                    assert s.inputs[3].child_output is s.outputs[-1]
    graph.close()

    savegraph(
        graph,
        f"{output_path}/{test_name}.pdf",
        label="Add inputs and an output for each block",
    )
