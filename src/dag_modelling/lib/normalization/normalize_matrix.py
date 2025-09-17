from __future__ import annotations

from typing import TYPE_CHECKING

from numba import njit

from ...core.input_strategy import AddNewInputAddNewOutput
from ...core.type_functions import AllPositionals, check_dimension_of_inputs, check_inputs_equivalence
from ..abstract import OneToOneNode

if TYPE_CHECKING:
    from typing import Literal

    from numpy.typing import NDArray


class NormalizeMatrix(OneToOneNode):
    __slots__ = ("_mode",)

    _mode: str

    def __init__(self, *args, mode: Literal["columns", "rows"] = "columns", **kwargs) -> None:
        super().__init__(
            *args,
            input_strategy=AddNewInputAddNewOutput(input_fmt="matrix", output_fmt="result"),
            **kwargs,
        )
        self._mode = mode

        if mode == "columns":
            self._labels.setdefault("mark", "norm cols")
        elif mode == "rows":
            self._labels.setdefault("mark", "norm rows")
        else:
            raise RuntimeError(f"Invalid NormalizeMatrix mode {mode}")

        self._functions_dict.update(
            {
                "columns": self._fcn_norm_columns,
                "rows": self._fcn_norm_rows,
            }
        )

    def _fcn_norm_rows(self) -> None:
        for indata, outdata in zip(self.inputs.iter_data(), self.outputs.iter_data_unsafe()):
            _norm_rows(indata, outdata)

    def _fcn_norm_columns(self) -> None:
        for indata, outdata in zip(self.inputs.iter_data(), self.outputs.iter_data_unsafe()):
            _norm_columns(indata, outdata)

    def _type_function(self) -> None:
        super()._type_function()
        check_dimension_of_inputs(self, AllPositionals, ndim=2)
        check_inputs_equivalence(self, AllPositionals, check_dtype=True, check_shape=True)

        self.function = self._functions_dict[self._mode]


@njit(cache=True)
def _norm_rows(matrix: NDArray, out: NDArray):
    ncols = matrix.shape[1]
    for row in range(matrix.shape[0]):
        total_sum = 0.0
        for column in range(ncols):
            total_sum += matrix[row, column]
        if total_sum == 0.0:
            out[row, :] = 0.0
            continue
        for column in range(ncols):
            out[row, column] += matrix[row, column] / total_sum


@njit(cache=True)
def _norm_columns(matrix: NDArray, out: NDArray):
    nrows = matrix.shape[0]
    for column in range(matrix.shape[1]):
        total_sum = 0.0
        for row in range(nrows):
            total_sum += matrix[row, column]
        if total_sum == 0.0:
            out[:, column] = 0.0
            continue
        for row in range(nrows):
            out[row, column] += matrix[row, column] / total_sum
