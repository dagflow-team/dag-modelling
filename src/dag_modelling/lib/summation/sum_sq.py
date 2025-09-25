from numpy import add, empty, square
from numpy.typing import NDArray

from ...core.type_functions import (
    AllPositionals,
    copy_shape_from_inputs_to_outputs,
    evaluate_dtype_of_outputs,
)
from ..abstract import ManyToOneNode


class SumSq(ManyToOneNode):
    """Sum of the squares of all the inputs"""

    __slots__ = ("_buffer",)
    _buffer: NDArray

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "Σ()²")

    def _function(self):
        for callback in self._input_nodes_callbacks:
            callback()

        output_data = self._output_data
        square(self._input_data0, out=output_data)
        for input_data in self._input_data_other:
            square(input_data, out=self._buffer)
            add(self._buffer, output_data, out=output_data)

    def _type_function(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        super()._type_function()
        copy_shape_from_inputs_to_outputs(self, 0, "result")
        evaluate_dtype_of_outputs(self, AllPositionals, "result")

    def _post_allocate(self) -> None:
        super()._post_allocate()
        inpdd = self.inputs[0].dd
        self._buffer = empty(shape=inpdd.shape, dtype=inpdd.dtype)
