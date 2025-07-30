from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from numba import njit
from numpy import allclose, finfo, isclose

from ...core.exception import InitializationError
from ...core.input_strategy import AddNewInput
from ...core.node import Node
from ...core.type_functions import (
    AllPositionals,
    assign_edges_from_inputs_to_outputs,
    check_dimension_of_inputs,
    check_inputs_equivalence,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from ...core.input import Input
    from ...core.output import Output


RebinModes = {"python", "numba"}
RebinModesType = Literal[RebinModes]


class RebinMatrix(Node):
    """For a given `edges_old` and `edges_new` computes the conversion
    matrix."""

    __slots__ = (
        "_edges_old",
        "_edges_old_clones",
        "_edges_new",
        "_result",
        "_atol",
        "_rtol",
        "_mode",
    )

    _edges_old: Input
    _edges_old_clones: tuple[Input, ...]
    _edges_new: Input
    _result: Output
    _atol: float
    _rtol: float
    _mode: str

    def __init__(
        self,
        *args,
        atol: float = float(finfo("d").resolution) * 10.0,
        rtol: float = 0.0,
        mode: RebinModesType = "numba",
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
            input_strategy=AddNewInput(input_fmt="edges_old"),
            allowed_kw_inputs=("edges_new",),
        )
        self.labels.setdefaults(
            {
                "text": "Matrix for rebinning",
            }
        )
        if mode not in RebinModes:
            raise InitializationError(
                f"mode must be in {RebinModes}, but given {mode}!", node=self
            )
        self._mode = mode
        self._atol = atol
        self._rtol = rtol
        self._edges_old = self._add_input("edges_old")  # input: 0
        self._edges_new = self._add_input("edges_new", positional=False)  # input: 1
        self._result = self._add_output("matrix")  # output: 0
        self._functions_dict.update(
            {
                "python": self._function_python,
                "numba": self._function_numba,
            }
        )

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def atol(self) -> float:
        return self._atol

    @property
    def rtol(self) -> float:
        return self._rtol

    def _function_python(self):
        edges_old = self._edges_old.data
        ret = _calc_rebin_matrix_python(
            edges_old, self._edges_new.data, self._result._data, self.atol, self.rtol
        )
        if ret[0] > 0:
            self.__raise_exception_at_wrong_edges(*ret)
        for i, input in enumerate(self._edges_old_clones):
            if not allclose(edges_old, input.data, atol=self._atol, rtol=self._rtol):
                raise RuntimeError(f"Clones of old edges are inconsistent (input {i})")

    def _function_numba(self):
        edges_old = self._edges_old.data
        ret = _calc_rebin_matrix_numba(
            self._edges_old.data,
            self._edges_new.data,
            self._result._data,
            self.atol,
            self.rtol,
        )
        if ret[0] > 0:
            self.__raise_exception_at_wrong_edges(*ret)
        for i, input in enumerate(self._edges_old_clones):
            if not allclose(edges_old, input.data, atol=self._atol, rtol=self._rtol):
                raise RuntimeError(f"Clones of old edges are inconsistent (input {i})")

    def __raise_exception_at_wrong_edges(
        self, retcode, iold, edge_old, inew, edge_new
    ) -> None:
        print("Old edges:", self._edges_old.dd.size, self._edges_old.data)
        print("New edges:", self._edges_new.dd.size, self._edges_new.data)
        edges_kind = (
            "first edge is before old first",
            "last edge is after the old last",
            "inconsistent new edge",
            "old edges (clones) are not consistent",
        )[retcode - 1]
        raise RuntimeError(
            f"Inconsistent edges ({edges_kind}): old {iold} {edge_old}, new {inew} {edge_new}, diff {edge_old-edge_new:.2g}"
        )

    def _type_function(self) -> None:
        """A output takes this function to determine the dtype and shape."""
        check_dimension_of_inputs(self, ("edges_old", "edges_new"), 1)
        check_inputs_equivalence(
            self, AllPositionals, check_dtype=True, check_shape=True
        )
        self._result.dd.shape = (
            self._edges_new.dd.size - 1,
            self._edges_old.dd.size - 1,
        )
        self._result.dd.dtype = "d"
        self.function = self._functions_dict[self.mode]
        assign_edges_from_inputs_to_outputs(
            (self._edges_new, self._edges_old), self._result
        )

        self._edges_old_clones = tuple(self.inputs[1:])


def _calc_rebin_matrix_python(
    edges_old: NDArray,
    edges_new: NDArray,
    rebin_matrix: NDArray,
    atol: float,
    rtol: float,
) -> tuple[int, int, float, int, float]:
    """
    For a column C of size N: Cnew = M C
    Cnew = [Mx1]
    M = [MxN]
    C = [Nx1]
    """

    if edges_new[0] < edges_old[0] and not isclose(
        edges_new[0], edges_old[0], atol=atol, rtol=rtol
    ):
        return 1, 0, edges_old[0], 0, edges_new[0]
    if edges_new[-1] > edges_old[-1] and not isclose(
        edges_new[-1], edges_old[-1], atol=atol, rtol=rtol
    ):
        return 2, -1, edges_old[-1], -1, edges_new[-1]

    inew = 0
    iold = 0
    edge_old = edges_old[0]
    edge_new_prev = edges_new[0]
    # nold = edges_old.size

    stepper_old = enumerate(edges_old)
    iold, edge_old = next(stepper_old)
    for inew, edge_new in enumerate(edges_new[1:], 1):
        while edge_old < edge_new and not isclose(
            edge_new, edge_old, atol=atol, rtol=rtol
        ):
            if edge_old >= edge_new_prev or isclose(
                edge_old, edge_new_prev, atol=atol, rtol=rtol
            ):
                rebin_matrix[inew - 1, iold] = 1.0

            iold, edge_old = next(stepper_old)

        if not isclose(edge_new, edge_old, atol=atol, rtol=rtol):
            return 3, iold, edge_old, inew, edge_new_prev

    return 0, iold, edge_old, inew, edge_new_prev


_calc_rebin_matrix_numba: Callable[
    [NDArray, NDArray, NDArray, float, float], tuple[int, int, float, int, float]
] = njit(cache=True)(_calc_rebin_matrix_python)
