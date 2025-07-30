from __future__ import annotations

from typing import TYPE_CHECKING

from numba import njit
from numpy import allclose

from ...core.node import Node
from ...core.type_functions import (
    check_dimension_of_inputs,
    check_inputs_have_same_dtype,
    check_inputs_have_same_shape,
    check_size_of_inputs,
    copy_dtype_from_inputs_to_outputs,
    evaluate_dtype_of_outputs,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from ...core.input import Input
    from ...core.output import Output


class AxisDistortionMatrixLinear(Node):
    """For a given historam and distorted X axis compute the conversion matrix.

    Distortion is assumed to be linear.
    """

    __slots__ = (
        "_edges_original",
        "_edges_target",
        "_edges_modified",
        "_result",
    )

    _edges_original: Input
    _edges_target: Input
    _edges_modified: Input
    _result: Output

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels.setdefaults(
            {
                "text": r"Bin edges distortion matrix",
            }
        )
        self._edges_original = self._add_input("EdgesOriginal", positional=False)
        self._edges_target = self._add_input("EdgesTarget", positional=False)
        self._edges_modified = self._add_input("EdgesModified", positional=False)
        self._result = self._add_output("matrix")  # output: 0

        self._functions_dict.update(
            {
                "python": self._function_python,
                "numba": self._function_numba,
            }
        )

    def _function_python(self):
        _axisdistortion_linear_python(
            self._edges_original.data,
            self._edges_target.data,
            self._edges_modified.data,
            self._result._data,
        )

    def _function_numba(self):
        _axisdistortion_linear_numba(
            self._edges_original.data,
            self._edges_target.data,
            self._edges_modified.data,
            self._result._data,
        )

    def _type_function(self) -> None:
        """A output takes this function to determine the dtype and shape."""
        names_edges = ("EdgesOriginal", "EdgesTarget", "EdgesModified")
        check_dimension_of_inputs(self, names_edges, 1)
        check_inputs_have_same_dtype(self, names_edges)
        (nedges,) = check_inputs_have_same_shape(self, names_edges)
        check_size_of_inputs(self, "EdgesOriginal", min=1)
        copy_dtype_from_inputs_to_outputs(self, "EdgesOriginal", "matrix")
        evaluate_dtype_of_outputs(self, names_edges, "matrix")

        self._result.dd.shape = (nedges - 1, nedges - 1)
        edges_original = self._edges_original.parent_output
        edges_target = self._edges_target.parent_output
        self._result.dd.axes_edges = (edges_target, edges_original)
        self.function = self._functions_dict["numba"]


def _axisdistortion_linear_python(
    edges_original: NDArray,
    edges_target: NDArray,
    edges_modified: NDArray,
    matrix: NDArray,
):
    # in general, target edges may be different (finer than original), the code should be able to handle it.
    # but currently we just check that edges are the same.
    assert edges_original is edges_target or allclose(
        edges_original, edges_target, atol=0.0, rtol=0.0
    )
    min_target = edges_target[0]
    nbinsx = edges_original.size - 1
    nbinsy = edges_target.size - 1

    matrix[:, :] = 0.0

    threshold = -1e10
    # left_axis = 0
    right_axis = 0
    idxy0, idxy1, idxy = -1, -1, 0
    # leftx_fine = threshold
    lefty_fine = threshold
    while idxy0 < 0 or lefty_fine <= threshold or lefty_fine < min_target:
        left_edge_from_x = edges_modified[idxy0 + 1] < edges_target[idxy1 + 1]
        if left_edge_from_x:
            # leftx_fine = edges_original[idxy0 + 1]
            lefty_fine = edges_modified[idxy0 + 1]
            # left_axis = 0
            if (idxy0 := idxy0 + 1) >= nbinsx:
                return
        else:
            # leftx_fine = -1
            lefty_fine = edges_target[idxy1 + 1]
            # left_axis = 1
            if (idxy1 := idxy1 + 1) >= nbinsy:
                return

    width_coarse = edges_modified[idxy0 + 1] - edges_modified[idxy0]
    while True:
        right_modified = edges_modified[idxy0 + 1]
        right_target = edges_target[idxy1 + 1]

        if right_modified < right_target:
            righty_fine = right_modified
            # rightx_fine = edges_original[idxy0 + 1]
            right_axis = 0
        else:
            righty_fine = right_target
            # rightx_fine = -1
            right_axis = 1

        while lefty_fine >= edges_target[idxy + 1]:
            if (idxy := idxy + 1) > nbinsy:
                break

        #
        # Uncomment the following lines to see the debug output
        # (you need to also uncomment all the `left_axis` lines)
        #
        # width_fine = righty_fine-lefty_fine
        # factor = width_fine/width_coarse
        # print(
        #         f"x:{leftx_fine:8.4f}→{rightx_fine:8.4f} "
        #         f"ax:{left_axis}→{right_axis} idxy:{idxy0: 4d},{idxy1: 4d} idxy: {idxy: 4d} "
        #         f"y:{lefty_fine:8.4f}→{righty_fine:8.4f}/{edges_modified[idxy0]:8.4f}→{edges_modified[idxy0+1]:8.4f}="
        #         f"{width_fine:8.4f}/{width_coarse:8.4f}={factor:8.4g} "
        # )

        matrix[idxy, idxy0] = (righty_fine - lefty_fine) / width_coarse

        if right_axis == 0:
            if (idxy0 := idxy0 + 1) >= nbinsx:
                break
            width_coarse = edges_modified[idxy0 + 1] - edges_modified[idxy0]
        elif (idxy1 := idxy1 + 1) >= nbinsx:
            break
        lefty_fine = righty_fine
        # leftx_fine = rightx_fine
        # left_axis = right_axis


_axisdistortion_linear_numba: Callable[[NDArray, NDArray, NDArray, NDArray], None] = (
    njit(cache=True)(_axisdistortion_linear_python)
)
