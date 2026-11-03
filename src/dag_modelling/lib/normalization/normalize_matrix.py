from __future__ import annotations

from typing import TYPE_CHECKING, override

from numba import njit
from numpy import zeros

from ...core.data_descriptor import Layout
from ...core.global_parameters import NUMBA_CACHE_ENABLE
from ...core.input_strategy import AddNewInputAddNewOutput
from ...core.type_functions import (
    AllPositionals,
    check_dimension_of_inputs,
    check_inputs_equivalence,
)
from ...methods.sparse.csr import csr_copy
from ..abstract import OneToOneNode

if TYPE_CHECKING:
    from typing import Literal

    from numpy.typing import NDArray


class NormalizeMatrix(OneToOneNode):
    __slots__ = ("_mode", "_colsum")

    _mode: str
    _colsum: NDArray

    def __init__(self, *args, mode: Literal["columns", "rows"] = "columns", **kwargs) -> None:
        super().__init__(
            *args,
            input_strategy=AddNewInputAddNewOutput(input_fmt="matrix", output_fmt="result"),
            allowed_layouts={AllPositionals: (Layout.SPARSE_CSR.value,)},
            **kwargs,
        )
        self._mode = mode

        if mode == "columns":
            self._labels.setdefault("mark", "norm cols")
        elif mode == "rows":
            self._labels.setdefault("mark", "norm rows")
        else:
            raise RuntimeError(f"Invalid NormalizeMatrix mode {mode}")

        self._evaluator._functions_dict.update(
            {
                "columns": self._fcn_norm_columns,
                "rows": self._fcn_norm_rows,
                "columns_csr": self._fcn_norm_columns_csr,
                "rows_csr": self._fcn_norm_rows_csr,
            }
        )

    def _fcn_norm_rows(self) -> None:
        for indata, outdata in zip(self.inputs.iter_data(), self.outputs.iter_data_unsafe()):
            _norm_rows(indata, outdata)

    def _fcn_norm_columns(self) -> None:
        for indata, outdata in zip(self.inputs.iter_data(), self.outputs.iter_data_unsafe()):
            _norm_columns(indata, outdata)

    def _fcn_norm_rows_csr(self) -> None:
        for matrix, out in zip(self.inputs.iter_data(), self.outputs.iter_data_unsafe()):
            csr_copy(matrix, out)
            _norm_rows_csr(out.data, out.indptr)

    def _fcn_norm_columns_csr(self) -> None:
        for matrix, out in zip(self.inputs.iter_data(), self.outputs.iter_data_unsafe()):
            csr_copy(matrix, out)
            _norm_columns_csr(out.data, out.indices, out.nnz, self._colsum)

    @override
    def _type_function(self) -> None:
        super()._type_function()
        check_dimension_of_inputs(self, AllPositionals, ndim=2)
        check_inputs_equivalence(self, AllPositionals, check_dtype=True, check_shape=True)

        self._evaluator.function = self._evaluator._functions_dict[self._mode]
        if self._copy_layout_from_inputs() == Layout.DENSE:
            return
        if self._mode == "columns":
            self._colsum = zeros(self.inputs[0].dd.shape[1])
            self._internal_buffers["colsum"] = self._colsum
        self._evaluator.function = self._evaluator._functions_dict[f"{self._mode}_csr"]


@njit(cache=NUMBA_CACHE_ENABLE)
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
            out[row, column] = matrix[row, column] / total_sum


@njit(cache=NUMBA_CACHE_ENABLE)
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
            out[row, column] = matrix[row, column] / total_sum


@njit(cache=NUMBA_CACHE_ENABLE)
def _norm_rows_csr(data: NDArray, indptr: NDArray):
    for row in range(indptr.shape[0] - 1):
        start, end = indptr[row], indptr[row + 1]
        total_sum = 0.0
        for j in range(start, end):
            total_sum += data[j]
        if total_sum == 0.0:
            data[start:end] = 0.0
            continue
        for j in range(start, end):
            data[j] /= total_sum


@njit(cache=NUMBA_CACHE_ENABLE)
def _norm_columns_csr(data: NDArray, indices: NDArray, nnz: int, colsum: NDArray):
    colsum[:] = 0.0
    for j in range(nnz):
        colsum[indices[j]] += data[j]
    for j in range(nnz):
        total_sum = colsum[indices[j]]
        if total_sum == 0.0:
            data[j] = 0.0
        else:
            data[j] /= total_sum
