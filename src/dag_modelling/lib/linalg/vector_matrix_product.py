from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import matmul, multiply

from ...core.exception import TypeFunctionError
from ...core.input_strategy import AddNewInputAddNewOutput
from ...core.node import Node
from ...core.type_functions import (
    AllPositionals,
    check_dimension_of_inputs,
    check_inputs_are_matrices_or_diagonals,
    check_inputs_are_matrix_multipliable,
    evaluate_dtype_of_outputs,
)

if TYPE_CHECKING:
    from typing import Literal

    from ...core.input import Input


class VectorMatrixProduct(Node):
    """Compute matrix product `C=row(v)@M` or `C=M@column(v)`"""

    __slots__ = ("_mat", "_matrix_column")

    _mat: Input
    _matrix_column: bool

    def __init__(self, *args, mode: Literal["column", "row"], **kwargs) -> None:
        kwargs.setdefault(
            "input_strategy",
            AddNewInputAddNewOutput(input_fmt="vector", output_fmt="result"),
        )
        super().__init__(*args, **kwargs, allowed_kw_inputs=("matrix",))
        self._mat = self._add_input("matrix", positional=False)

        if mode == "column":
            self._matrix_column = True
            self._labels.setdefault("mark", "M@col(v)")
        elif mode == "row":
            self._matrix_column = False
            self._labels.setdefault("mark", "row(v)@M")
        else:
            raise RuntimeError(f"Invalid VectorMatrixProduct mode {mode}")

        self._functions_dict.update(
            {
                "row_diagonal": self._fcn_row_diagonal,
                "row_block": self._fcn_row_block,
                "diagonal_column": self._fcn_diagonal_column,
                "block_column": self._fcn_block_column,
            }
        )

    def _fcn_row_block(self):
        mat = self._mat.data
        for row, outdata in zip(
            self.inputs.iter_data(), self.outputs.iter_data_unsafe()
        ):
            matmul(row, mat, out=outdata)

    def _fcn_block_column(self):
        mat = self._mat.data
        for column, outdata in zip(
            self.inputs.iter_data(), self.outputs.iter_data_unsafe()
        ):
            matmul(mat, column, out=outdata)

    def _fcn_row_diagonal(self):
        mat = self._mat.data
        for diag, outdata in zip(
            self.inputs.iter_data(), self.outputs.iter_data_unsafe()
        ):
            multiply(mat, diag, out=outdata)

    def _fcn_diagonal_column(self):
        diag = self._mat.data
        for col, outdata in zip(
            self.inputs.iter_data(), self.outputs.iter_data_unsafe()
        ):
            multiply(diag, col, out=outdata)

    def _type_function(self) -> None:
        check_dimension_of_inputs(self, AllPositionals, ndim=1)
        ndim_mat = check_inputs_are_matrices_or_diagonals(self, "matrix")
        if ndim_mat not in (1, 2):
            raise TypeFunctionError(f"Matrix dimension >2: {ndim_mat}", node=self)

        if self._matrix_column:
            for i, out in enumerate(self.outputs):
                (resshape,) = check_inputs_are_matrix_multipliable(self, "matrix", i)
                out.dd.shape = (resshape[0],)
            self.function = (
                ndim_mat == 2 and self._fcn_block_column or self._fcn_diagonal_column
            )
        else:
            for i, out in enumerate(self.outputs):
                (resshape,) = check_inputs_are_matrix_multipliable(self, i, "matrix")
                out.dd.shape = (resshape[-1],)
            self.function = (
                ndim_mat == 2 and self._fcn_row_block or self._fcn_row_diagonal
            )

        # column: [MxN] x [Nx1] -> [Mx1]
        # row: [1xM] x [MxN] -> [1xN]
        mat_edges = self.inputs["matrix"].dd.axes_edges
        if mat_edges:
            edges = (
                (mat_edges[not self._matrix_column],)
                if ndim_mat == 2
                else (mat_edges[0],)
            )
            for out in self.outputs:
                out.dd.axes_edges = edges
        evaluate_dtype_of_outputs(self, AllPositionals, AllPositionals)
