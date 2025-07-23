from __future__ import annotations

from typing import TYPE_CHECKING

from numba import njit

from ...core.type_functions import AllPositionals, check_node_has_inputs, check_dimension_of_inputs
from ..abstract import OneToOneNode

if TYPE_CHECKING:
    from numpy import double
    from numpy.typing import NDArray


@njit(cache=True)
def _binedges(centers: NDArray[double], edges: NDArray[double]) -> None:
    ncenters = len(centers)
    for i in range(1, ncenters):
        edges[i] = (centers[i - 1] + centers[i]) * 0.5
    edges[0] = 2.0 * centers[0] - edges[1]
    edges[ncenters] = 2.0 * centers[ncenters - 1] - edges[ncenters - 1]


class MeshToEdges(OneToOneNode):
    """
    Computes the bin edges based on the mesh (bin centers). The left-most and the right-most edges are inferred.

    inputs:
        `i`: array with mesh (N)

    outputs:
        `i`: array with edges of bins (N+1)
    """

    __slots__ = ()

    def _function(self):
        for indata, outdata in zip(self.inputs.iter_data(), self.outputs.iter_data_unsafe()):
            _binedges(indata, outdata)

    def _type_function(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        check_node_has_inputs(self, check_named=True)
        check_dimension_of_inputs(self, (AllPositionals, *self.inputs.kw.keys()), ndim=1)
        for _input, _output in zip(self.inputs, self.outputs):
            inputdd = _input.dd
            _output.dd.dtype = inputdd.dtype
            _output.dd.shape = (inputdd.shape[0] + 1,)
