from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import multiply

from ...core.node import Node
from ...core.type_functions import (
    check_dimension_of_inputs,
    check_inputs_are_square_matrices,
    check_inputs_are_matrix_multipliable,
    copy_from_inputs_to_outputs,
)

if TYPE_CHECKING:
    from ...core.input import Input
    from ...core.output import Output


class CovmatrixFromCormatrix(Node):
    """Compute covariance matrix from correlation matrix:
    Vₖₘ=Cₖₘσₖσₘ
    """

    __slots__ = ("_mode", "_sigma", "_cormatrix", "_covmatrix")

    _mode: str
    _sigma: Input
    _cormatrix: Input
    _covmatrix: Output

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, allowed_kw_inputs=("sigma", "cormatrix"))
        self._labels.setdefault("mark", "C→V")

        self._cormatrix, self._covmatrix = self._add_pair(
            "cormatrix", "covmatrix", output_kws={"positional": True}
        )
        self._sigma = self._add_input("sigma", positional=False)

    def _function(self):
        self.inputs.touch()
        C = self._cormatrix.data
        sigma = self._sigma.data
        V = self._covmatrix._data

        multiply(C, sigma[None, :], out=V)
        multiply(V, sigma[:, None], out=V)

    def _type_function(self) -> None:
        check_inputs_are_square_matrices(self, "cormatrix")
        check_dimension_of_inputs(self, "sigma", 1)
        check_inputs_are_matrix_multipliable(self, "cormatrix", "sigma")
        copy_from_inputs_to_outputs(self, slice(None), slice(None))
