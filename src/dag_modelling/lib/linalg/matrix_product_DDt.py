from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import matmul

from ...core.node import Node
from ...core.type_functions import check_dimension_of_inputs, evaluate_dtype_of_outputs

if TYPE_CHECKING:
    from ...core.input import Input
    from ...core.output import Output


class MatrixProductDDt(Node):
    """
    Compute matrix product `C=D@Dᵀ`.
    """

    __slots__ = ("_matrix", "_out",)

    _matrix: Input
    _out: Output

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs, allowed_kw_inputs=("matrix",))
        self._matrix = self._add_input("matrix")
        self._out = self._add_output("result")
        self._labels.setdefault("mark", "DDᵀ")

    def _function(self):
        matrix = self._matrix.data
        matmul(matrix, matrix.T, out=self._out._data)

    def _type_function(self) -> None:
        check_dimension_of_inputs(self, "matrix", ndim=2)
        evaluate_dtype_of_outputs(self, slice(None), "result")
        self._out.dd.shape = (self._matrix.dd.shape[0], self._matrix.dd.shape[0])
