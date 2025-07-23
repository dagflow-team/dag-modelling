from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import add, divide, matmul, multiply, subtract
from scipy.linalg import solve_triangular

from ...core.exception import InitializationError
from ...core.input_strategy import AddNewInputAddNewOutput
from ...core.node import Node
from ...core.type_functions import (
    check_node_has_inputs,
    check_dimension_of_inputs,
    check_inputs_are_matrices_or_diagonals,
    check_inputs_equivalence,
    check_inputs_are_matrix_multipliable,
    copy_from_inputs_to_outputs,
)

if TYPE_CHECKING:
    from ...core.input import Input


class NormalizeCorrelatedVars(Node):
    """Normalize correlated variables or correlate normal variables with linear expression

    If x is a vector of values, μ are the central values and L is a cholesky decomposition
    of the covariance matrix V=LLᵀ then
    z = L⁻¹(x - μ)
    x = Lz + μ
    """

    __slots__ = ("_mode", "_matrix", "_central")
    _mode: str
    _matrix: Input
    _central: Input

    def __init__(self, *args, mode="forward", **kwargs):
        if mode == "forward":
            mark = "c→u"
        elif mode == "backward":
            mark = "u→c"
        else:
            raise InitializationError(
                f'Invalid NormalizeCorrelatedVars mode={mode}. Expect "forward" or "backward"',
                node=self,
            )

        self._mode = mode

        super().__init__(
            *args,
            input_strategy=AddNewInputAddNewOutput(),
            **kwargs,
            allowed_kw_inputs=("matrix", "central"),
        )
        self._labels.setdefault("mark", mark)

        self._matrix = self._add_input("matrix", positional=False)
        self._central = self._add_input("central", positional=False)

        self._functions_dict.update(
            {
                "forward_2d": self._fcn_forward_2d,
                "backward_2d": self._fcn_backward_2d,
                "forward_1d": self._fcn_forward_1d,
                "backward_1d": self._fcn_backward_1d,
            }
        )

    def _fcn_forward_2d(self):
        self.inputs.touch()
        L = self._matrix.data
        central = self._central.data
        for indata, outdata in zip(self.inputs.iter_data(), self.outputs.iter_data_unsafe()):
            subtract(indata, central, out=outdata)
            solve_triangular(L, outdata, lower=True, overwrite_b=True, check_finite=False)

    def _fcn_backward_2d(self):
        self.inputs.touch()
        L = self._matrix.data
        central = self._central.data
        for indata, outdata in zip(self.inputs.iter_data(), self.outputs.iter_data_unsafe()):
            matmul(L, indata, out=outdata)
            add(outdata, central, out=outdata)

    def _fcn_forward_1d(self):
        self.inputs.touch()
        Ldiag = self._matrix.data
        central = self._central.data
        for indata, outdata in zip(self.inputs.iter_data(), self.outputs.iter_data_unsafe()):
            subtract(indata, central, out=outdata)
            divide(outdata, Ldiag, out=outdata)

    def _fcn_backward_1d(self):
        self.inputs.touch()
        Ldiag = self._matrix.data
        central = self._central.data
        for indata, outdata in zip(self.inputs.iter_data(), self.outputs.iter_data_unsafe()):
            multiply(Ldiag, indata, out=outdata)
            add(outdata, central, out=outdata)

    def _type_function(self) -> None:
        check_node_has_inputs(self)
        ndim = check_inputs_are_matrices_or_diagonals(self, "matrix", check_square=True)
        check_dimension_of_inputs(self, "central", 1)
        check_inputs_equivalence(self, ("central", slice(None)))
        check_inputs_are_matrix_multipliable(self, "matrix", slice(None))
        copy_from_inputs_to_outputs(self, slice(None), slice(None))

        key = f"{self._mode}_{ndim}d"
        try:
            self.function = self._functions_dict[key]
        except KeyError as exc:
            raise InitializationError(
                f'Invalid mode "{key}". Expect: {self._functions_dict.keys()}'
            ) from exc
