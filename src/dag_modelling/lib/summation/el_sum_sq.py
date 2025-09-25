from numba import njit
from numpy.typing import NDArray

from ...core.input_strategy import AddNewInputAddAndKeepSingleOutput
from ..abstract import ManyToOneNode


@njit(cache=True)
def _sumsq(data: NDArray, out: NDArray):
    sm = 0.0
    for v in data:
        sm += v * v
    out[0] += sm


class ElSumSq(ManyToOneNode):
    """Sum of the squared of all the inputs."""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("input_strategy", AddNewInputAddAndKeepSingleOutput(output_fmt="result"))
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "Σa²")

    def _function(self):
        for callback in self._input_nodes_callbacks:
            callback()

        output_data = self._output_data
        output_data[0] = 0.0

        for input_data in self._input_data:
            _sumsq(input_data, output_data)

    def _type_function(self) -> None:
        """A output takes this function to determine the dtype and shape."""
        super()._type_function()
        self.outputs[0].dd.shape = (1,)
