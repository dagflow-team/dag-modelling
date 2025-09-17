from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import matmul, multiply

from ...core.exception import TypeFunctionError
from ...core.node import Node
from ...core.type_functions import (
    check_node_has_inputs,
    check_inputs_are_matrices_or_diagonals,
    check_inputs_are_matrix_multipliable,
    evaluate_dtype_of_outputs,
)

if TYPE_CHECKING:
    from ...core.input import Input
    from ...core.output import Output


class MatrixProductAB(Node):
    """
    Compute matrix product `C=A@B`,
    """

    __slots__ = ("_left", "_right", "_out")

    _left: Input
    _right: Input
    _out: Output

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs, allowed_kw_inputs=("left", "right"))
        self._left = self._add_input("left")
        self._right = self._add_input("right")
        self._out = self._add_output("result")
        self._labels.setdefault("mark", "A@B")

        self._functions_dict.update(
            {
                "diagonal_diagonal": self._fcn_diagonal_diagonal,
                "diagonal_block": self._fcn_diagonal_block,
                "block_diagonal": self._fcn_block_diagonal,
                "block_block": self._fcn_block_block,
            }
        )

    def _fcn_block_block(self):
        matmul(self._left.data, self._right.data, out=self._out._data)

    def _fcn_block_diagonal(self):
        multiply(self._left.data, self._right.data, out=self._out._data)

    def _fcn_diagonal_block(self):
        multiply(self._left.data[:, None], self._right.data, out=self._out._data)

    def _fcn_diagonal_diagonal(self):
        multiply(self._left.data, self._right.data, out=self._out._data)

    def _type_function(self) -> None:
        check_node_has_inputs(self, ("left", "right"))
        ndim_left = check_inputs_are_matrices_or_diagonals(self, "left")
        ndim_right = check_inputs_are_matrices_or_diagonals(self, "right")

        ndim = ndim_left, ndim_right
        ndim_out = 2
        if ndim == (2, 2):
            self.function = self._fcn_block_block
        elif ndim == (2, 1):
            self.function = self._fcn_block_diagonal
        elif ndim == (1, 2):
            self.function = self._fcn_diagonal_block
        elif ndim == (1, 1):
            self.function = self._fcn_diagonal_diagonal
            ndim_out = 1
        else:
            raise TypeFunctionError(f"One of the inputs' dimension >2: {ndim}", node=self)

        (resshape,) = check_inputs_are_matrix_multipliable(self, "left", "right")
        evaluate_dtype_of_outputs(self, slice(None), "result")

        self._out.dd.shape = resshape[:ndim_out]
