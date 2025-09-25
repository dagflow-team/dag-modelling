from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import empty, matmul, multiply

from ...core.node import Node
from ...core.type_functions import (
    check_node_has_inputs,
    check_dimension_of_inputs,
    check_inputs_are_matrices_or_diagonals,
    check_inputs_are_matrix_multipliable,
    evaluate_dtype_of_outputs,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ...core.input import Input
    from ...core.output import Output


class MatrixProductDVDt(Node):
    """
    Compute matrix product `C=D@V@Dᵀ`,
    where `D` is a matrix and `V` is a square matrix.
    `V` maybe 1d array representing the diagonal of the diagonal matrix.
    """

    __slots__ = ("_left", "_square", "_out", "_buffer")

    _left: Input
    _square: Input
    _out: Output
    _buffer: NDArray

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs, allowed_kw_inputs=("left", "square"))
        self._left = self._add_input("left")
        self._square = self._add_input("square")
        self._out = self._add_output("result")
        self._functions_dict.update({"diagonal": self._fcn_diagonal, "square": self._fcn_square})
        self._labels.setdefault("mark", "DVDᵀ")

    def _fcn_diagonal(self):
        # square matrix stored as diagonal
        left = self._left.data
        multiply(left, self._square.data, out=self._buffer)
        matmul(self._buffer, left.T, out=self._out._data)

    def _fcn_square(self):
        left = self._left.data
        matmul(left, self._square.data, out=self._buffer)
        matmul(self._buffer, left.T, out=self._out._data)

    def _type_function(self) -> None:
        check_node_has_inputs(self, ("left", "square"))
        check_dimension_of_inputs(self, "left", ndim=2)
        ndim = check_inputs_are_matrices_or_diagonals(self, "square", check_square=True)
        check_inputs_are_matrix_multipliable(self, "left", "square")
        evaluate_dtype_of_outputs(self, slice(None), "result")
        self._out.dd.shape = (self._left.dd.shape[0], self._left.dd.shape[0])
        self.function = self._functions_dict["diagonal" if ndim == 1 else "square"]

    def _post_allocate(self) -> None:
        self._buffer = empty(shape=self._left.dd.shape, dtype=self._left.dd.dtype)
