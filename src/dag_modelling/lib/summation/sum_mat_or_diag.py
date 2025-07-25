from numba import njit
from numpy import add, copyto
from numpy.typing import NDArray

from ...core.input_strategy import AddNewInputAddAndKeepSingleOutput
from ...core.type_functions import (
    AllPositionals,
    check_inputs_consistency_with_square_matrices_or_diagonals,
    check_inputs_have_same_dtype,
    check_node_has_inputs,
    copy_shape_from_inputs_to_outputs,
    evaluate_dtype_of_outputs,
)
from ..abstract import ManyToOneNode


@njit(cache=True)
def _settodiag1(inarray: NDArray, outmatrix: NDArray):
    outmatrix[:] = 0
    for i in range(inarray.size):
        outmatrix[i, i] = inarray[i]


@njit(cache=True)
def _addtodiag(inarray: NDArray, outmatrix: NDArray):
    for i in range(inarray.size):
        outmatrix[i, i] += inarray[i]


class SumMatOrDiag(ManyToOneNode):
    """Sum of all the inputs together.

    Inputs are square matrices or diagonals of square matrices
    """

    __slots__ = ("_ndim",)
    _ndim: int

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("input_strategy", AddNewInputAddAndKeepSingleOutput(output_fmt="result"))
        super().__init__(*args, **kwargs)
        self._functions_dict.update({2: self._fcn2d, 1: self._fcn1d})

    def _fcn2d(self):
        for callback in self._input_nodes_callbacks:
            callback()

        output_data = self._output_data
        input_data0 = self._input_data0
        if len(input_data0.shape) == 1:
            _settodiag1(input_data0, output_data)
        else:
            output_data[:] = input_data0

        if len(self.inputs) > 1:
            for input_data in self._input_data_other:
                if len(input_data.shape) == 1:
                    _addtodiag(input_data, output_data)
                else:
                    add(input_data, output_data, out=output_data)

    def _fcn1d(self):
        for callback in self._input_nodes_callbacks:
            callback()

        output_data = self._output_data
        copyto(output_data, self._input_data0)
        for input_data in self._input_data_other:
            add(output_data, input_data, out=output_data)

    def _type_function(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        check_node_has_inputs(self)
        copy_shape_from_inputs_to_outputs(self, 0, "result")
        self._ndim = check_inputs_consistency_with_square_matrices_or_diagonals(self)
        check_inputs_have_same_dtype(self)
        evaluate_dtype_of_outputs(self, AllPositionals, "result")

        size = self.inputs[0].dd.shape[0]
        output = self.outputs[0]
        if self._ndim == 2:
            output.dd.shape = size, size
        elif self._ndim == 1:
            output.dd.shape = (size,)
        else:
            assert False

        self.function = self._functions_dict[self._ndim]
