from __future__ import annotations

from math import sqrt
from typing import TYPE_CHECKING

from numba import njit

from ...core.node import Node
from ...core.type_functions import (
    AllPositionals,
    assign_axes_from_inputs_to_outputs,
    check_dimension_of_inputs,
    check_shape_of_inputs,
    copy_from_inputs_to_outputs,
)

if TYPE_CHECKING:
    from numpy import double
    from numpy.typing import NDArray

    from ...core.input import Input
    from ...core.output import Output


@njit(cache=True)
def _rel_sigma(
    a: double,
    b: double,
    c: double,
    Energy: NDArray[double],
    Sigma: NDArray[double],
):
    a2 = a * a
    b2 = b * b
    c2 = c * c
    for i in range(len(Energy)):
        e = Energy[i]
        Sigma[i] = sqrt(a2 + b2 / e + c2 / (e * e))  # sqrt(a^2 + b^2/E + c^2/E^2)


class EnergyResolutionSigmaRelABC(Node):
    r"""Energy resolution $\sqrt(a^2 + b^2/E + c^2/E^2)$

    inputs:
        `a_nonuniform`: parameter a, due to energy deposition nonuniformity (size=1)
        `b_stat`: parameter b, due to stat fluctuations (size=1)
        `c_noise`: parameter c, due to dark noise (size=1)
        `0` or `Energy`: Input bin Energy (N elements)

    outputs:
        `0` or `RelSigma`: relative RelSigma for each bin (N elements)
    """

    __slots__ = ("_a_nonuniform", "_b_stat", "_c_noise", "_energy", "_rel_sigma")

    _a_nonuniform: Input
    _b_stat: Input
    _c_noise: Input
    _energy: Input
    _rel_sigma: Output

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.labels.setdefaults(
            {
                "text": r"Relative energy resolution σ/E",
                "latex": r"Relative energy resolution $\sigma/E$",
                "axis": r"$\sigma/E$",
            }
        )
        self._a_nonuniform, self._b_stat, self._c_noise = (
            self._add_inputs(  # pyright: ignore reportGeneralTypeIssues
                ("a_nonuniform", "b_stat", "c_noise"), positional=False
            )
        )
        self._energy = self._add_input("Energy")  # input: 0
        self._rel_sigma = self._add_output("RelSigma")  # output: 0

    def _function(self) -> None:
        _rel_sigma(
            self._a_nonuniform.data[0],
            self._b_stat.data[0],
            self._c_noise.data[0],
            self._energy.data,
            self._rel_sigma._data,
        )

    def _type_function(self) -> None:
        """A output takes this function to determine the dtype and shape."""
        check_shape_of_inputs(self, ("a_nonuniform", "b_stat", "c_noise"), (1,))
        check_dimension_of_inputs(self, AllPositionals, 1)
        copy_from_inputs_to_outputs(self, "Energy", "RelSigma")
        assign_axes_from_inputs_to_outputs(
            self, "Energy", "RelSigma", assign_meshes=True
        )
