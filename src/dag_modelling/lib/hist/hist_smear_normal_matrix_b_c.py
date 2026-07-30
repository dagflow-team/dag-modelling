from __future__ import annotations

from math import exp, sqrt
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
def __resolution_width(e_true: double, e_rec: double, rel_sigma: double) -> double:
    _invtwopisqrt = 1.0 / sqrt(2.0 * pi)
    sigma = e_true * rel_sigma
    reldiff = (e_true - e_rec) / sigma
    return exp(-0.5 * reldiff * reldiff) * _invtwopisqrt / sigma


@njit(cache=NUMBA_CACHE_ENABLE)
def _resolution_matrix(
    rel_sigma: NDArray[double],
    edges: NDArray[double],
    edges_out: NDArray[double],
    result: NDArray[double],
    value_threshold: float,
) -> None:
    assert edges is edges_out or allclose(edges, edges_out, atol=0.0, rtol=0.0)

    bincenter = lambda i: (edges[i] + edges[i + 1]) * 0.5
    nbins = len(rel_sigma)
    for itrue in range(nbins):
        is_right_edge = False
        etrue = bincenter(itrue)
        rel_sigma_i = rel_sigma[itrue]
        for jrec in range(nbins):
            erec = bincenter(jrec)
            d_erec = edges[jrec + 1] - edges[jrec]
            r_events = d_erec * __resolution_width(etrue, erec, rel_sigma_i)
            if r_events < value_threshold:
                if is_right_edge:
                    result[jrec:, itrue] = 0.0
                    break
                result[jrec, itrue] = 0.0
                continue
            is_right_edge = True
            result[jrec, itrue] = r_events


class HistSmearNormalMatrixBC(Node):
    """Energy resolution.

    inputs:
        `0` or `RelSigma`: Relative Sigma value for each bin (N elements)
        `Edges`: Input bin Edges (N elements)
        `EdgesOut`: Output bin Edges (N elements), should be consistent with Edges.

    outputs:
        `0` or `SmearMatrix`: SmearMatrixing weights (NxN)
    """

    __slots__ = ("_edges", "_edges_out", "_rel_sigma", "_smear_matrix", "_value_threshold")

    _edges: Input
    _edges_out: Input
    _rel_sigma: Input
    _smear_matrix: Output
    _value_threshold: float

    def __init__(self, name, value_threshold: float = 1e-10, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.labels.setdefaults(
            {
                "text": r"Energy resolution matrix (via bin centers)",
                "axis": r"C",
            }
        )
        self._value_threshold = value_threshold
        self._rel_sigma = self._add_input("RelSigma")  # input: 0
        self._edges = self._add_input("Edges", positional=False)
        self._edges_out = self._add_input("EdgesOut", positional=False)
        self._smear_matrix = self._add_output("SmearMatrix")  # output: 0

    @property
    def value_threshold(self) -> float:
        return self._value_threshold

    def _function(self):
        _resolution_matrix(
            self._rel_sigma.data,
            self._edges.data,
            self._edges_out.data,
            self._smear_matrix._data,
            self._value_threshold,
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
