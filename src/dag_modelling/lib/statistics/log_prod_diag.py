from numpy import diag, empty, log
from numpy.typing import NDArray

from ...core.input_strategy import AddNewInputAddNewOutput
from ...core.type_functions import (
    AllPositionals,
    check_inputs_are_matrices_or_diagonals,
    check_node_has_inputs,
    copy_dtype_from_inputs_to_outputs,
)
from ..abstract import OneToOneNode


class LogProdDiag(OneToOneNode):
    """Compute the LogProdDiag of a matrix log|V|=log|LL̃ᵀ|=2Σlog(Lᵢᵢ) based on
    Cholesky decomposition of matrix V 1d input is considered to be a squared
    root diagonal of square matrix.

    inputs:
        `matrix`: cholesky decomposition of matrix

    outputs:
        `0`: sum of logarithm of diagonal elements
    """

    __slots__ = ("_buffer",)
    _buffer: NDArray

    def __init__(self, *args, **kwargs):
        kwargs.setdefault(
            "input_strategy",
            AddNewInputAddNewOutput(input_fmt="matrix", output_fmt="log_V"),
        )
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", r"2log\|L\|")

        self._functions_dict.update(
            {"square": self._fcn_square, "diagonal": self._fcn_diagonal}
        )

    def _fcn_square(self):
        """Compute logarithm of determinant of matrix using Cholesky
        decomposition."""
        for callback in self._input_nodes_callbacks:
            callback()

        for indata, outdata in self._input_output_data:
            log(diag(indata), out=self._buffer)
            outdata[0] = 2 * self._buffer.sum()

    def _fcn_diagonal(self):
        """Compute "LogProdDiag" using of a diagonal of a square matrix."""
        for callback in self._input_nodes_callbacks:
            callback()

        for indata, outdata in self._input_output_data:
            log(indata, out=self._buffer)
            outdata[0] = 2 * self._buffer.sum()

    def _type_function(self) -> None:
        check_node_has_inputs(self, AllPositionals)
        ndim = check_inputs_are_matrices_or_diagonals(
            self, AllPositionals, check_square=True
        )
        copy_dtype_from_inputs_to_outputs(self, AllPositionals, AllPositionals)
        for out in self.outputs:
            out.dd.shape = (1,)

        if ndim == 2:
            self.function = self._functions_dict["square"]
        else:
            self.function = self._functions_dict["diagonal"]

    def _post_allocate(self) -> None:
        super()._post_allocate()
        inpdd = self.inputs[0].dd
        self._buffer = empty(shape=(inpdd.shape[0],), dtype=inpdd.dtype)
