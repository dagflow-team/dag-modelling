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


class AxisDistortionMatrix(Node):
    """For a given histogram and distorted X axis compute the conversion
    matrix."""

    __slots__ = (
        "_edges_original",
        "_edges_target",
        "_edges_modified",
        "_edges_backward",
        "_result",
    )

    _edges_original: Input
    _edges_target: Input
    _edges_modified: Input
    _edges_backward: Input
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
        self._edges_backward = self._add_input(
            "EdgesModifiedBackwards", positional=False
        )
        self._result = self._add_output("matrix")  # output: 0

        self._functions_dict.update(
            {
                "python": self._function_python,
                "numba": self._function_numba,
            }
        )

    def _function_python(self):
        _axisdistortion_python(
            self._edges_original.data,
            self._edges_target.data,
            self._edges_modified.data,
            self._edges_backward.data,
            self._result._data,
        )

    def _function_numba(self):
        _axisdistortion_numba(
            self._edges_original.data,
            self._edges_target.data,
            self._edges_modified.data,
            self._edges_backward.data,
            self._result._data,
        )

    def _type_function(self) -> None:
        """A output takes this function to determine the dtype and shape."""
        names_edges = (
            "EdgesOriginal",
            "EdgesTarget",
            "EdgesModified",
            "EdgesModifiedBackwards",
        )
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


def _axisdistortion_python(
    edges_original: NDArray,
    edges_target: NDArray,
    edges_modified: NDArray,
    edges_backwards: NDArray,
    matrix: NDArray,
) -> None:
    # in general, target edges may be different (finer than original), the code should be able to handle it.
    # but currently we just check that edges are the same.
    assert edges_original is edges_target or allclose(
        edges_original, edges_target, atol=0.0, rtol=0.0
    )

    edges_target = edges_original
    min_original = edges_original[0]
    min_target = edges_target[0]
    nbinsx = edges_original.size - 1
    nbinsy = edges_target.size - 1

    matrix[:, :] = 0.0

    threshold = -1e10
    # left_axis = 0
    right_axis = 0
    idxx0, idxx1, idxy = -1, -1, 0
    leftx_fine, lefty_fine = threshold, threshold
    while (
        leftx_fine <= threshold or leftx_fine < min_original or lefty_fine < min_target
    ):
        left_edge_from_x = edges_original[idxx0 + 1] < edges_backwards[idxx1 + 1]
        if left_edge_from_x:
            leftx_fine, lefty_fine = (
                edges_original[idxx0 + 1],
                edges_modified[idxx0 + 1],
            )
            # left_axis = 0
            if (idxx0 := idxx0 + 1) >= nbinsx:
                return
        else:
            leftx_fine, lefty_fine = edges_backwards[idxx1 + 1], edges_target[idxx1 + 1]
            # left_axis = 1
            if (idxx1 := idxx1 + 1) >= nbinsx:
                return

    width_coarse = edges_original[idxx0 + 1] - edges_original[idxx0]
    while True:
        right_orig = edges_original[idxx0 + 1]
        right_backwards = edges_backwards[idxx1 + 1]

        if right_orig < right_backwards:
            rightx_fine = right_orig
            righty_fine = edges_modified[idxx0 + 1]
            right_axis = 0
        else:
            rightx_fine = right_backwards
            righty_fine = edges_target[idxx1 + 1]
            right_axis = 1

        while lefty_fine >= edges_target[idxy + 1]:
            if (idxy := idxy + 1) > nbinsy:
                break

        ##
        ## Uncomment the following lines to see the debug output
        ## (you need to also uncomment all the `left_axis` lines)
        ##
        # width_fine = rightx_fine-leftx_fine
        # factor = width_fine/width_coarse
        # print(
        #         f"x:{leftx_fine:8.4f}→{rightx_fine:8.4f}="
        #         f"{width_fine:8.4f}/{width_coarse:8.4f}={factor:8.4g} "
        #         f"ax:{left_axis}→{right_axis} idxx:{idxx0: 4d},{idxx1: 4d} idxy: {idxy: 4d} "
        #         f"y:{lefty_fine:8.4f}→{righty_fine:8.4f}"
        # )

        matrix[idxy, idxx0] = (rightx_fine - leftx_fine) / width_coarse

        if right_axis == 0:
            if (idxx0 := idxx0 + 1) >= nbinsx:
                break
            width_coarse = edges_original[idxx0 + 1] - edges_original[idxx0]
        elif (idxx1 := idxx1 + 1) >= nbinsx:
            break
        leftx_fine, lefty_fine = rightx_fine, righty_fine
        # left_axis = right_axis


_axisdistortion_numba: Callable[[NDArray, NDArray, NDArray, NDArray, NDArray], None] = (
    njit(cache=True)(_axisdistortion_python)
)
