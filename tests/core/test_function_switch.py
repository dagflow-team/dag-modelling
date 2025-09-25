from numpy import arange, array, copyto, result_type

from dag_modelling.core.graph import Graph
from dag_modelling.core.input_strategy import AddNewInputAddAndKeepSingleOutput
from dag_modelling.lib.common import Array
from dag_modelling.core.node import Node


class SumIntOrProductFloatOrDoNothing(Node):
    def __init__(self, name, **kwargs):
        kwargs.setdefault("input_strategy", AddNewInputAddAndKeepSingleOutput(output_fmt="result"))
        super().__init__(name, **kwargs)
        self._functions_dict.update({"int": self._fcn_int, "float": self._fcn_float})

    def _function(self):
        pass

    def _fcn_int(self):
        out = self.outputs[0]._data
        copyto(out, self.inputs[0].data.copy())
        if len(self.inputs) > 1:
            for _input in self.inputs[1:]:
                out += _input.data
        return out

    def _fcn_float(self):
        out = self.outputs[0]._data
        copyto(out, self.inputs[0].data.copy())
        if len(self.inputs) > 1:
            for _input in self.inputs[1:]:
                out *= _input.data
        return out

    def _type_function(self) -> bool:
        if self.inputs[0].dd.dtype == "i":
            self.function = self._functions_dict.get("int")
        elif self.inputs[0].dd.dtype == "d":
            self.function = self._functions_dict.get("float")
        self.outputs["result"].dd.shape = self.inputs[0].dd.shape
        self.outputs["result"].dd.dtype = result_type(*tuple(inp.dd.dtype for inp in self.inputs))
        self.logger.debug(
            f"Node '{self.name}': dtype={self.outputs['result'].dd.dtype}, "
            f"shape={self.outputs['result'].dd.shape}, function={self.function.__name__}"
        )
        return True


def test_00(debug_graph):
    with Graph(debug=debug_graph, close_on_exit=True):
        arr = Array("arr", array(("1", "2", "3")), mode="fill")
        node = SumIntOrProductFloatOrDoNothing("node")
        (arr, arr) >> node
    assert (node.outputs["result"].data == ["", "", ""]).all()


def test_01(debug_graph):
    with Graph(debug=debug_graph, close_on_exit=True):
        arr = Array("arr", arange(3, dtype="i"), mode="fill")  # [0, 1, 2]
        node = SumIntOrProductFloatOrDoNothing("node")
        (arr, arr) >> node
    assert (node.outputs["result"].data == [0, 2, 4]).all()


def test_02(debug_graph):
    with Graph(debug=debug_graph, close_on_exit=True):
        arr = Array("arr", arange(3, dtype="d"), mode="fill")  # [0, 1, 2]
        node = SumIntOrProductFloatOrDoNothing("node")
        (arr, arr) >> node
    assert (node.outputs["result"].data == [0, 1, 4]).all()
