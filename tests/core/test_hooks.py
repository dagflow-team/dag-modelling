from numpy import arange, copyto
from pytest import raises

from dag_modelling.core.exception import CriticalError, ReconnectionError, UnclosedGraphError
from dag_modelling.core.graph import Graph
from dag_modelling.lib.common import Array
from dag_modelling.lib.abstract import ManyToOneNode
from dag_modelling.lib.summation import WeightedSum, WeightedSumArgs


class ThreeInputsSum(ManyToOneNode):
    def _function(self):
        out = self.outputs["result"]._data
        copyto(out, self.inputs[0].data.copy())
        for _input in self.inputs[1:3]:
            out += _input.data
        return out

    def _type_function(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        super()._type_function()
        if (y := len(self.inputs)) != 3:
            raise CriticalError(
                f"The node must have only 3 inputs, but given {y}: {self.inputs}!"
            )
        self.logger.debug(
            f"Node '{self.name}': dtype={self.outputs['result'].dd.dtype}, "
            f"shape={self.outputs['result'].dd.shape}"
        )


def test_00(debug_graph):
    with Graph(debug=debug_graph, close_on_exit=True):
        arr = Array("arr", arange(3, dtype="i"), mode="fill")  # [0, 1, 2]
        node = ThreeInputsSum("threesum")
        for _ in range(3):
            # Error while evaluating before len(inputs) == 3
            with raises(UnclosedGraphError):
                node.eval()
            arr >> node
    assert (node.outputs["result"].data == [0, 3, 6]).all()


def test_01(debug_graph):
    weight_arr = (2, 3)
    with Graph(debug=debug_graph, close_on_exit=True):
        arr = Array("arr", arange(3, dtype="i"), mode="fill")  # [0, 1, 2]
        weight = Array("weight", weight_arr, mode="fill")
        weight1 = Array("weight", weight_arr[:1], mode="fill")

        ws1 = WeightedSum("weightedsum")
        ws2 = WeightedSum("weightedsum-sameweight")
        ws3 = WeightedSumArgs("weightedsum_args", weight=weight_arr)
        ws4 = WeightedSumArgs("weightedsum_args-sameweight", weight=weight_arr[:1])
        (arr, arr) >> ws1
        (arr, arr) >> ws2
        (arr, arr) >> ws3
        (arr, arr) >> ws4

        # Error while eval before setting the weight input
        with raises(UnclosedGraphError):
            ws1.eval()

        # multiply the first input by 2 and the second one by 3
        weight >> ws1("weight")
        weight1 >> ws2("weight")

    with raises(ReconnectionError):
        weight >> ws1("weight")

    assert (ws1.outputs["result"].data == [0, 5, 10]).all()
    assert (ws2.outputs["result"].data == [0, 4, 8]).all()
    assert (ws3.outputs["result"].data == [0, 5, 10]).all()
    assert (ws4.outputs["result"].data == [0, 4, 8]).all()
