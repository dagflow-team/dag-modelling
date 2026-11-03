from __future__ import annotations

from typing import TYPE_CHECKING, override

from numba import njit
from numpy import zeros

from ...core.data_descriptor import Layout
from ...core.global_parameters import NUMBA_CACHE_ENABLE
from ...core.input_strategy import AddNewInputAddNewOutput
from ...core.type_functions import (
    AllPositionals,
    check_inputs_are_square_matrices,
    check_inputs_equivalence,
    check_shape_of_inputs,
    evaluate_dtype_of_outputs,
)
from ..abstract import OneToOneNode

if TYPE_CHECKING:
    from typing import Callable, Literal

    from numpy.typing import NDArray

    from ...core.input import Input


class RenormalizeDiag(OneToOneNode):
    __slots__ = ("_mode", "_ndiag", "_scale", "_colsum")

    _mode: str
    _ndiag: int
    _scale: Input
    _colsum: NDArray

    def __init__(
        self, *args, mode: Literal["diag", "offdiag"] = "diag", ndiag: int = 1, **kwargs
    ) -> None:
        super().__init__(
            *args,
            input_strategy=AddNewInputAddNewOutput(input_fmt="matrix", output_fmt="result"),
            allowed_layouts={AllPositionals: (Layout.SPARSE_CSR.value,)},
            **kwargs,
        )

        if mode in {"diag", "offdiag"}:
            self._labels.setdefault("mark", f"renormalize {mode}")
        else:
            raise RuntimeError(f"Invalid RenormalizeDiag mode {mode}")
        self._mode = mode

        if ndiag < 1 or not isinstance(ndiag, int):
            raise RuntimeError(
                f"Invalid RenormalizeDiag {ndiag=}, {type(ndiag)=} (must be int >1)!"
            )
        self._ndiag = ndiag

        self._scale = self._add_input("scale", positional=False)
        self._evaluator._functions_dict.update(
            {"diag": self._fcn_diag, "offdiag": self._fcn_offdiag, "csr": self._fcn_csr}
        )

    def _fcn_diag(self) -> None:
        scale = self._scale.data[0]
        for indata, outdata in zip(self.inputs.iter_data(), self.outputs.iter_data_unsafe()):
            _renorm_diag_numba(indata, outdata, scale, self._ndiag)

    def _fcn_offdiag(self) -> None:
        scale = self._scale.data[0]
        for indata, outdata in zip(self.inputs.iter_data(), self.outputs.iter_data_unsafe()):
            _renorm_offdiag_numba(indata, outdata, scale, self._ndiag)

    def _fcn_csr(self) -> None:
        scale = self._scale.data[0]
        offdiag = self._mode == "offdiag"
        for matrix, out in zip(self.inputs.iter_data(), self.outputs.iter_data_unsafe()):
            out.reserve(matrix.nnz)
            _renorm_csr(
                matrix.data,
                matrix.indices,
                matrix.indptr,
                out.data,
                out.indices,
                out.indptr,
                self._colsum,
                scale,
                self._ndiag,
                offdiag,
            )
            out.update()

    @override
    def _type_function(self) -> None:
        super()._type_function()
        check_shape_of_inputs(self, "scale", (1,))
        check_inputs_are_square_matrices(self, 0)
        check_inputs_equivalence(self, AllPositionals, check_dtype=True, check_shape=True)
        evaluate_dtype_of_outputs(self, (AllPositionals, "scale"), "result")
        self._evaluator.function = self._evaluator._functions_dict[self._mode]
        if self._copy_layout_from_inputs() == Layout.SPARSE_CSR:
            self._colsum = zeros(self.inputs[0].dd.shape[1])
            self._internal_buffers["colsum"] = self._colsum
            self._evaluator.function = self._evaluator._functions_dict["csr"]


@njit(cache=NUMBA_CACHE_ENABLE)
def _norming(matrix: NDArray) -> None:
    for icol in range(matrix.shape[-1]):
        colsum = 0.0
        for irow in range(matrix.shape[0]):
            colsum += matrix[irow, icol]
        if colsum == 0.0:
            continue
        for irow in range(matrix.shape[0]):
            matrix[irow, icol] /= colsum


def _renorm_diag_python(matrix: NDArray, out: NDArray, scale: float, ndiag: int) -> None:
    out[:, :] = matrix[:, :]
    n = out.shape[0]
    # main diagonal
    for i in range(n):
        out[i, i] *= scale
    # other diagonals
    for idiag in range(1, ndiag):
        i = 0
        while i + idiag < n:
            out[i + idiag, i] *= scale
            out[i, i + idiag] *= scale
            i += 1
    _norming(out)


def _renorm_offdiag_python(matrix: NDArray, out: NDArray, scale: float, ndiag: int) -> None:
    out[:, :] = matrix[:, :]
    out *= scale
    n = out.shape[0]
    # main diagonal
    for i in range(n):
        out[i, i] = matrix[i, i]
    # # other diagonals
    for idiag in range(1, ndiag):
        i = 0
        while i + idiag < n:
            out[i + idiag, i] = matrix[i + idiag, i]
            out[i, i + idiag] = matrix[i, i + idiag]
            i += 1
    _norming(out)


@njit(cache=NUMBA_CACHE_ENABLE)
def _renorm_csr(
    data_in: NDArray,
    indices_in: NDArray,
    indptr_in: NDArray,
    data: NDArray,
    indices: NDArray,
    indptr: NDArray,
    colsum: NDArray,
    scale: float,
    ndiag: int,
    offdiag: bool,
) -> None:
    """Scale the cells of `ndiag` diagonals, or the other ones, and normalize the columns"""
    indptr[:] = indptr_in
    colsum[:] = 0.0
    for row in range(indptr.shape[0] - 1):
        for j in range(indptr[row], indptr[row + 1]):
            col = indices_in[j]
            value = data_in[j]
            if (abs(row - col) < ndiag) != offdiag:
                value *= scale
            data[j] = value
            indices[j] = col
            # the stored value, not `value`: float32 data must sum as in the dense kernel
            colsum[col] += data[j]
    for j in range(indptr[-1]):
        total = colsum[indices[j]]
        if total != 0.0:
            data[j] /= total


_renorm_diag_numba: Callable[[NDArray, NDArray, float, float], None] = njit(
    cache=NUMBA_CACHE_ENABLE
)(_renorm_diag_python)
_renorm_offdiag_numba: Callable[[NDArray, NDArray, float, float], None] = njit(
    cache=NUMBA_CACHE_ENABLE
)(_renorm_offdiag_python)
