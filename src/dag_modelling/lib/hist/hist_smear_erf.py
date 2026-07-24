from __future__ import annotations

from math import exp, sqrt, erf
from typing import TYPE_CHECKING

from numba import njit
from numpy import pi

from ...core.global_parameters import NUMBA_CACHE_ENABLE
from ...core.node import Node
from ...core.type_functions import (
    AllPositionals,
    check_dimension_of_inputs,
    check_size_of_inputs,
    find_max_size_of_inputs,
)

if TYPE_CHECKING:
    from numpy import double
    from numpy.typing import NDArray

    from ...core.input import Input
    from ...core.output import Output


@njit(cache=NUMBA_CACHE_ENABLE)
def _resolution(
    rel_sigma: NDArray[double],
    edges: NDArray[double],
    edges_out: NDArray[double],
    result: NDArray[double],
    min_events: float,
) -> None:
    n_bins_in = len(edges) - 1
    n_bins_out = len(edges_out) - 1

    _invtwosqrt = 1.0 / sqrt(2.0)
    _invtwopisqrt = _invtwosqrt / sqrt(pi)

    for jtrue in range(n_bins_in):
        delta_jtrue = edges[jtrue + 1] - edges[jtrue]
        sigma_jtrue = edges[jtrue] * rel_sigma[jtrue]
        if sigma_jtrue == 0.0:
            for krec in range(n_bins_out):
                result[krec, jtrue] = 0.0
            continue
        factor0 = _invtwosqrt / sigma_jtrue
        for krec in range(n_bins_out):
            delta_10 = edges_out[krec + 1] - edges[jtrue]
            delta_00 = edges_out[krec] - edges[jtrue]
            delta_11 = edges_out[krec + 1] - edges[jtrue + 1]
            delta_01 = edges_out[krec] - edges[jtrue + 1]
            reldelta_10 = delta_10 * factor0
            reldelta_00 = delta_00 * factor0
            reldelta_11 = delta_11 * factor0
            reldelta_01 = delta_01 * factor0
            result[krec, jtrue] = (
                0.5 * (
                    + delta_10 * erf(reldelta_10) - delta_00 * erf(reldelta_00) - delta_11 * erf(reldelta_11) + delta_01 * erf(reldelta_01)
                )
                + _invtwopisqrt * sigma_jtrue * (
                    + exp(-reldelta_10 * reldelta_10) - exp(-reldelta_00 * reldelta_00) - exp(-reldelta_11 * reldelta_11) + exp(-reldelta_01 * reldelta_01)
                )
            ) / delta_jtrue


class HistSmearErf(Node):
    """Energy resolution.

    inputs:
        `0` or `RelSigma`: Relative Sigma value for each bin (N elements)
        `Edges`: Input bin Edges (N elements)
        `EdgesOut`: Output bin Edges (M elements)

    outputs:
        `0` or `SmearMatrix`: SmearMatrixing weights (NxM)
    """

    __slots__ = ("_edges", "_edges_out", "_rel_sigma", "_smear_matrix", "_min_events")

    _edges: Input
    _edges_out: Input
    _rel_sigma: Input
    _smear_matrix: Output
    _min_events: float

    def __init__(self, name, min_events: float = 1e-10, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.labels.setdefaults(
            {
                "text": r"Energy resolution $E_{res}$, MeV",
                "plot_title": r"Energy resolution $E_{res}$, MeV",
                "latex": r"$E_{res}$, MeV",
                "axis": r"$E_{res}$, MeV",
            }
        )
        self._min_events = min_events
        self._rel_sigma = self._add_input("RelSigma")  # input: 0
        self._edges = self._add_input("Edges", positional=False)
        self._edges_out = self._add_input("EdgesOut", positional=False)
        self._smear_matrix = self._add_output("SmearMatrix")  # output: 0

    @property
    def min_events(self) -> float:
        return self._min_events

    def _function(self):
        _resolution(
            self._rel_sigma.data,
            self._edges.data,
            self._edges_out.data,
            self._smear_matrix._data,
            self._min_events,
        )
        # print(self._smear_matrix._data)
        # print(self._rel_sigma.data)

    def _type_function(self) -> None:
        """A output takes this function to determine the dtype and shape."""
        check_dimension_of_inputs(self, AllPositionals, 1)
        size = find_max_size_of_inputs(self, "RelSigma")
        check_size_of_inputs(self, "Edges", exact=size + 1)
        # check_inputs_have_same_shape(self, ["Edges", "EdgesOut"])

        rel_sigma_dd = self._rel_sigma.dd
        edges_out_dd = self._edges_out.dd
        self._smear_matrix.dd.shape = (edges_out_dd.shape[0], rel_sigma_dd.shape[0])
        self._smear_matrix.dd.dtype = rel_sigma_dd.dtype
        edges = self._edges._parent_output
        edges_out = self._edges_out._parent_output
        self._smear_matrix.dd.axes_edges = (edges_out, edges)
