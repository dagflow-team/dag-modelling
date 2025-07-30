from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import zeros

from ...core.input_strategy import InputStrategyViewConcat
from ...core.node import Node
from ...core.type_functions import check_dimension_of_inputs, check_dtype_of_inputs

if TYPE_CHECKING:
    from ...core.output import Output


class ViewConcat(Node):
    """Creates a node with a single data output which is a concatenated memory
    of the inputs."""

    __slots__ = ("_output", "_offsets")
    _output: Output
    _offsets: list[int]

    def __init__(self, name, outname="concat", **kwargs) -> None:
        super().__init__(
            name, **kwargs, input_strategy=InputStrategyViewConcat(node=self)
        )
        self._output = self._add_output(
            outname, allocatable=False, forbid_reallocation=True
        )
        self._offsets = []

    def _function(self):
        self.inputs.touch()

    def _type_function(self) -> None:
        """A output takes this function to determine the dtype and shape."""
        size = 0
        self._offsets = []
        cdtype = self.inputs[0].dd.dtype
        check_dtype_of_inputs(self, slice(None), dtype=cdtype)
        check_dimension_of_inputs(self, slice(None), 1)
        for _input in self.inputs:
            self._offsets.append(size)
            size += _input.dd.shape[0]

        output = self.outputs[0]
        output.dd.dtype = cdtype
        output.dd.shape = (size,)
        data = zeros(shape=size, dtype=cdtype)
        output._set_data(data, owns_buffer=True)

        for offset, _input in zip(self._offsets, self.inputs):
            size = _input.dd.shape[0]
            idata = data[offset : offset + size]
            _input.set_own_data(idata, owns_buffer=False)
