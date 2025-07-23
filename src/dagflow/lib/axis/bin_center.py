from __future__ import annotations

from typing import TYPE_CHECKING

from numba import njit

from ...core.type_functions import AllPositionals, check_node_has_inputs, check_dimension_of_inputs
from ..abstract import OneToOneNode

if TYPE_CHECKING:
    from numpy import double
    from numpy.typing import NDArray


@njit(cache=True)
def _bincenter(edges: NDArray[double], centers: NDArray[double]) -> None:
    nbins = len(centers)
    for i in range(nbins):
        centers[i] = (edges[i] + edges[i + 1]) * 0.5


class BinCenter(OneToOneNode):
    """
    The node finds centers of bins by the edges

    inputs:
        `i`: array with edges of bins (N)

    outputs:
        `i`: array with centers of bins (N-1)
    """

    __slots__ = ()

    def _function(self):
        for _input, _output in zip(self.inputs.iter_data(), self.outputs.iter_data_unsafe()):
            _bincenter(_input, _output)

    def _type_function(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        check_node_has_inputs(self, check_named=True)
        check_dimension_of_inputs(self, (AllPositionals, *self.inputs.kw.keys()), ndim=1)
        for _input, _output in zip(self.inputs, self.outputs):
            inputdd = _input.dd
            _output.dd.dtype = inputdd.dtype
            _output.dd.shape = (inputdd.shape[0] - 1,)
            _output.dd.axes_edges = (_input.parent_output,)
