from __future__ import annotations

from math import erf, exp, sqrt
from typing import TYPE_CHECKING

from numba import njit
from numpy import allclose, pi

from ...core.global_parameters import NUMBA_CACHE_ENABLE
from ...core.node import Node
from ...core.type_functions import (
    AllPositionals,
    check_dimension_of_inputs,
    check_inputs_have_same_shape,
    check_size_of_inputs,
    find_max_size_of_inputs,
)

if TYPE_CHECKING:
    from numpy import double
    from numpy.typing import NDArray

    from ...core.input import Input
    from ...core.output import Output


@njit(cache=NUMBA_CACHE_ENABLE)
def _resolution_matrix(
    rel_sigma: NDArray[double],
    edges: NDArray[double],
    edges_out: NDArray[double],
    result: NDArray[double],
) -> None:
    assert edges is edges_out or allclose(edges, edges_out, atol=0.0, rtol=0.0)
    n_bins_in = len(edges) - 1
    n_bins_out = len(edges_out) - 1

    _inv_sqrt_two = 1.0 / sqrt(2.0)
    _inv_sqrt_twopi = _inv_sqrt_two / sqrt(pi)

    for jtrue in range(n_bins_in):
        jtrue_next = jtrue + 1
        delta_jtrue = edges[jtrue_next] - edges[jtrue]
        sigma_jtrue = edges[jtrue] * rel_sigma[jtrue]
        sigma_jtrue_next = edges[jtrue_next] * rel_sigma[jtrue_next]
        if sigma_jtrue == 0.0:
            for krec in range(n_bins_out):
                result[krec, jtrue] = 0.0
            result[jtrue, jtrue] = 1.0
            continue
        factor0 = _inv_sqrt_two / sigma_jtrue
        factor1 = _inv_sqrt_two / sigma_jtrue_next
        for krec in range(n_bins_out):
            krec_next = krec + 1
            delta_10 = edges_out[krec_next] - edges[jtrue]
            delta_00 = edges_out[krec] - edges[jtrue]
            delta_11 = edges_out[krec_next] - edges[jtrue_next]
            delta_01 = edges_out[krec] - edges[jtrue_next]
            reldelta_10 = delta_10 * factor0
            reldelta_00 = delta_00 * factor0
            reldelta_11 = delta_11 * factor1
            reldelta_01 = delta_01 * factor1
            part_erf = (
                delta_10 * erf(reldelta_10)
                + delta_01 * erf(reldelta_01)
                - (delta_00 * erf(reldelta_00) + delta_11 * erf(reldelta_11))
            )
            part_exp = (
                exp(-reldelta_10 * reldelta_10)
                + exp(-reldelta_01 * reldelta_01)
                - (exp(-reldelta_00 * reldelta_00) + exp(-reldelta_11 * reldelta_11))
            )
            result[krec, jtrue] = (
                0.5 * part_erf + _inv_sqrt_twopi * sigma_jtrue * part_exp
            ) / delta_jtrue


class HistSmearNormalMatrixIntegral(Node):
    """Energy resolution.

    inputs:
        `0` or `RelSigma`: Relative Sigma value for each bin (N elements)
        `Edges`: Input bin Edges (N elements)
        `EdgesOut`: Output bin Edges (M elements)

    outputs:
        `0` or `SmearMatrix`: SmearMatrixing weights (NxM)
    """

    __slots__ = ("_edges", "_edges_out", "_rel_sigma", "_smear_matrix")

    _edges: Input
    _edges_out: Input
    _rel_sigma: Input
    _smear_matrix: Output

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.labels.setdefaults(
            {
                "text": r"Energy resolution matrix (via flat integration)",
                "axis": r"C",
            }
        )
        self._rel_sigma = self._add_input("RelSigma")  # input: 0
        self._edges = self._add_input("Edges", positional=False)
        self._edges_out = self._add_input("EdgesOut", positional=False)
        self._smear_matrix = self._add_output("SmearMatrix")  # output: 0

    def _function(self):
        _resolution_matrix(
            self._rel_sigma.data,
            self._edges.data,
            self._edges_out.data,
            self._smear_matrix._data,
        )

    def _type_function(self) -> None:
        """A output takes this function to determine the dtype and shape."""
        check_dimension_of_inputs(self, AllPositionals, 1)
        size = find_max_size_of_inputs(self, "RelSigma")
        check_size_of_inputs(self, "Edges", exact=size + 1)
        check_inputs_have_same_shape(self, ["Edges", "EdgesOut"])

        edges_in_dd = self._edges.dd
        edges_out_dd = self._edges_out.dd
        rel_sigma_dd = self._rel_sigma.dd
        self._smear_matrix.dd.shape = (edges_out_dd.shape[0] - 1, edges_in_dd.shape[0] - 1)
        self._smear_matrix.dd.dtype = rel_sigma_dd.dtype
        edges = self._edges._parent_output
        edges_out = self._edges_out._parent_output
        self._smear_matrix.dd.axes_edges = (edges_out, edges)
