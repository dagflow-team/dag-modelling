from __future__ import annotations

from typing import TYPE_CHECKING

from numba import njit
from numpy import integer

from ...core.exception import TypeFunctionError
from ...core.type_functions import (
    AllPositionals,
    check_node_has_inputs,
    check_dimension_of_inputs,
    check_shape_of_inputs,
    check_subtype_of_inputs,
    copy_dtype_from_inputs_to_outputs,
)
from ..abstract import OneToOneNode

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ...core.input import Input


@njit(cache=True)
def _psum(data: NDArray, range: NDArray, out: NDArray):
    out[0] = data[range[0] : range[1]].sum()


class PartialSums(OneToOneNode):
    """
    inputs:
        `array`: array to sum
        `0`, `...`: range to partial sum

    outputs:
        `0`, `...`: partial sum

    The node performs partial sums of the input array `a`
    within ranges which are the positional inputs.

    .. note:: now works only with 1d arrays
    """

    __slots__ = ("_array",)
    _array: Input

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._array = self._add_input("array", positional=False)

    def _type_function(self) -> None:
        check_node_has_inputs(self, "array")
        check_node_has_inputs(self, AllPositionals)
        check_dimension_of_inputs(self, (AllPositionals, "array"), 1)
        check_subtype_of_inputs(self, AllPositionals, dtype=integer)
        check_shape_of_inputs(self, AllPositionals, (2,))
        copy_dtype_from_inputs_to_outputs(self, "array", AllPositionals)
        for out in self.outputs:
            out.dd.shape = (1,)
        # TODO: axes_edges and axes_meshes?
        # now edges are restricted
        add = self._array.dd
        if add.axes_edges:
            raise TypeFunctionError(
                "The PartialSums doesn't support edges functional, " f"but given {add.axes_edges}",
                node=self,
                input=self._array,
            )

    def _function(self):
        data = self._array.data
        for indata, outdata in zip(self.inputs.iter_data(), self.outputs.iter_data_unsafe()):
            _psum(data, indata, outdata)
