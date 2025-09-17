from __future__ import annotations

from math import sqrt
from typing import TYPE_CHECKING

from numba import njit

from ...core.type_functions import (
    AllPositionals,
    check_dimension_of_inputs,
    check_inputs_have_same_shape,
    check_inputs_number_is_divisible_by_N,
    copy_from_inputs_to_outputs,
)
from ..abstract import BlockToOneNode

if TYPE_CHECKING:
    from typing import Literal

    from numpy import double
    from numpy.typing import NDArray


@njit(cache=True)
def _cnp_uncertainty(
    data: NDArray[double],
    theory: NDArray[double],
    result: NDArray[double],
) -> None:
    coeff = sqrt(3.0)
    for i in range(len(data)):
        result[i] = coeff / sqrt(1.0 / data[i] + 2.0 / theory[i])


@njit(cache=True)
def _cnp_variance(
    data: NDArray[double],
    theory: NDArray[double],
    result: NDArray[double],
) -> None:
    for i in range(len(data)):
        result[i] = 3.0 / (1.0 / data[i] + 2.0 / theory[i])


class CNPStat(BlockToOneNode):
    r"""Combined Neyman–Pearson statistic uncertainty: errors = sqrt(3) /
    sqrt(1/dataᵢ+2/theoryᵢ), so if we connect it to `Chi2` node we get χ² =
    (1/3) Σᵢ [(1/dataᵢ+2/theoryᵢ)·(theoryᵢ-dataᵢ)²]

    inputs:
        `i`: data 1d array
        `i+1`: theory 1d array

    outputs:
        `i`: the resulting array with errors.

    .. note:: The node must have only 2N inputs!
    """

    __slots__ = ()

    def __init__(
        self,
        name,
        *args,
        mode: Literal["uncertainty", "variance"] = "uncertainty",
        **kwargs,
    ):
        super().__init__(name, *args, **kwargs)
        assert mode in {"variance", "uncertainty"}
        self.labels.setdefaults(
            {
                "text": f"CNP stat. {mode}",
                "plot_title": f"CNP stat. {mode}",
                "latex": f"CNP stat. {mode}",
                "axis": f"CNP stat. {mode}",
                "mark": "σ(CNP)",
            }
        )

        self._functions_dict.update(
            {"uncertainty": self._cnp_uncertainty, "variance": self._cnp_variance}
        )
        self.function = self._functions_dict[mode]

    @staticmethod
    def _input_names() -> tuple[str, ...]:
        return "data", "theory"

    def _cnp_uncertainty(self) -> None:
        for callback in self._input_nodes_callbacks:
            callback()

        i = 0
        for i, output_data in enumerate(self._output_data):
            _cnp_uncertainty(
                self._input_data[2 * i], self._input_data[2 * i + 1], output_data
            )

    def _cnp_variance(self) -> None:
        for callback in self._input_nodes_callbacks:
            callback()

        i = 0
        for i, output_data in enumerate(self._output_data):
            _cnp_variance(
                self._input_data[2 * i], self._input_data[2 * i + 1], output_data
            )
            i += 2

    def _type_function(self) -> None:
        """A output takes this function to determine the dtype and shape."""
        check_inputs_number_is_divisible_by_N(self, 2)
        check_dimension_of_inputs(self, AllPositionals, 1)
        i = 0
        while i < self.inputs.len_pos():
            check_inputs_have_same_shape(self, (i, i + 1))
            copy_from_inputs_to_outputs(self, i, i // 2)
            i += 2
