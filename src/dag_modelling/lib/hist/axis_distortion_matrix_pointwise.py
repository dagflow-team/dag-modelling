from __future__ import annotations

from typing import TYPE_CHECKING

from numba import njit
from numpy import allclose, digitize, fabs

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


class AxisDistortionMatrixPointwise(Node):
    """For a given historam and distorted X axis compute the conversion matrix.

    Distortion is assumed to be linear.
    """

    __slots__ = (
        "_edges_original",
        "_edges_target",
        "_distortion_original",
        "_distortion_target",
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
        self._distortion_original = self._add_input(
            "DistortionOriginal", positional=False
        )  # X
        self._distortion_target = self._add_input(
            "DistortionTarget", positional=False
        )  # Y
        self._result = self._add_output("matrix")  # output: 0

        self._functions_dict.update(
            {
                "python": self._function_python,
                "numba": self._function_numba,
            }
        )

    def _function_python(self):
        _axisdistortion_pointwise_python(
            self._edges_original.data,
            self._edges_target.data,
            self._distortion_original.data,
            self._distortion_target.data,
            self._result._data,
        )

    def _function_numba(self):
        _axisdistortion_pointwise_numba(
            self._edges_original.data,
            self._edges_target.data,
            self._distortion_original.data,
            self._distortion_target.data,
            self._result._data,
        )

    def _type_function(self) -> None:
        """A output takes this function to determine the dtype and shape."""
        names_edges = (
            "EdgesOriginal",
            "EdgesTarget",
            "DistortionOriginal",
            "DistortionTarget",
        )
        check_dimension_of_inputs(self, names_edges, 1)
        check_inputs_have_same_dtype(self, names_edges)
        (nedges,) = check_inputs_have_same_shape(self, names_edges[:2])
        check_inputs_have_same_shape(self, names_edges[2:])
        check_size_of_inputs(self, "EdgesOriginal", min=2)
        check_size_of_inputs(self, "DistortionOriginal", min=2)
        copy_dtype_from_inputs_to_outputs(self, "EdgesOriginal", "matrix")
        evaluate_dtype_of_outputs(self, names_edges, "matrix")

        self._result.dd.shape = (nedges - 1, nedges - 1)
        edges_original = self._edges_original.parent_output
        edges_target = self._edges_target.parent_output
        self._result.dd.axes_edges = (edges_target, edges_original)
        self.function = self._functions_dict["numba"]


@njit
def _project_y_to_x_linear(
    y: float, x0: float, x1: float, y0: float, y1: float
) -> float:
    """Project Y to X linearly.

    For the case of constant height the middle x is returned. Note:
    review this decision in case of concerns.
    """
    k = (y1 - y0) / (x1 - x0)
    if k == 0:
        return (x0 + x1) * 0.5

    return (y - y0) / k + x0


def _axisdistortion_pointwise_python(
    edges_original: NDArray,
    edges_target: NDArray,
    distortion_original: NDArray,
    distortion_target: NDArray,
    matrix: NDArray,
):
    # in general, target edges may be different (finer than original), the code should be able to handle it.
    # but currently we just check that edges are the same.
    assert edges_original is edges_target or allclose(
        edges_original, edges_target, atol=0.0, rtol=0.0
    )
    matrix[:, :] = 0.0

    if (
        distortion_original[0] >= edges_original[-1]
        or distortion_original[-1] < edges_original[0]
    ):
        return

    n_bins_x = edges_original.size - 1
    n_bins_y = edges_target.size - 1

    n_points = distortion_original.size
    idx_last_point = n_points - 1
    last_x = distortion_original[idx_last_point]

    # fmt: off
    # print(                                                                                             # debug
    #     f"np {distortion_original.size} nex {edges_original.size} ney {edges_target.size}"             # debug
    #     " "                                                                                            # debug
    #     f"nbx {edges_original.size-1} nby {edges_target.size-1}"                                       # debug
    # )                                                                                                  # debug
    # fmt: on
    large_negative_number = -1e30
    large_positive_number = -large_negative_number

    idx = 0
    x0, x1 = distortion_original[idx], distortion_original[idx + 1]
    y0, y1 = distortion_target[idx], distortion_target[idx + 1]

    #
    # Iteration allowed direction:
    # - Edges:
    #   - right
    #   - up, up-right
    #   - down, down-right
    # - Curve: by index starting from 0
    # Forbidden directions:
    # - left*

    skip_incomplete_x = False
    bin_idx_y = digitize(y0, edges_target, right=False) - 1
    if bin_idx_y < 0:
        bottom_y = large_negative_number
        top_y = edges_target[0]
    elif bin_idx_y < n_bins_x:
        bottom_y = edges_target[bin_idx_y]
        top_y = edges_target[bin_idx_y + 1]
    else:
        bottom_y = edges_target[-1]
        top_y = large_positive_number

    bin_idx_x = digitize(x0, edges_original, right=False) - 1
    if bin_idx_x < 0:
        left_x = large_negative_number
        right_x = edges_original[0]
        right = large_negative_number
        left = large_negative_number

        skip_incomplete_x = True
    else:
        left_x = edges_original[bin_idx_x]
        right_x = edges_original[bin_idx_x + 1]

        right = min(right_x, x1)
        left = max(left_x, x0)

        skip_incomplete_x = (x0 != left_x) & (y0 > bottom_y) & (y0 < top_y)
    width_x_full = right_x - left_x

    # print(f"start ix {bin_idx_x} iy {bin_idx_y}")                                                      # debug

    did_advance = True
    is_inside_limits = False
    while did_advance:
        did_advance = False

        passed_x = x1 >= right_x
        passed_top_y = y1 >= top_y
        passed_bottom_y = (y1 < bottom_y) & (y1 > large_negative_number)

        assert not (
            passed_bottom_y & passed_top_y
        ), "Can not pass left and right edge on Y at the same time"

        passed_any = False

        passed_x_first = False
        passed_top_y_first = False
        passed_bottom_y_first = False

        if passed_x:
            passed_any = True
            right = right_x
            passed_x_first = True

        if passed_top_y:
            right_x_from_top_y = _project_y_to_x_linear(top_y, x0, x1, y0, y1)
            if passed_any:
                if right_x_from_top_y == right:
                    passed_top_y_first = True
                elif right_x_from_top_y <= right:
                    passed_x_first = False
                    passed_top_y_first = True
                    # passed_bottom_y_first = False

                    right = right_x_from_top_y
            else:
                passed_top_y_first = True
                right = right_x_from_top_y
                passed_any = True
        elif passed_bottom_y:
            right_x_from_bottom_y = _project_y_to_x_linear(bottom_y, x0, x1, y0, y1)
            if passed_any:
                if right_x_from_bottom_y == right:
                    passed_bottom_y_first = True
                elif right_x_from_bottom_y < right:
                    passed_x_first = False
                    # passed_top_y_first = False
                    passed_bottom_y_first = True

                    right = right_x_from_bottom_y
            else:
                passed_bottom_y_first = True
                right = right_x_from_bottom_y
                passed_any = True

        # fmt: off
        ## Uncomment the following lines to see the debug output
        # debug_is_inside_limits = (                                                                     # debug
        #     (bin_idx_x >= 0)                                                                           # debug
        #     & (bin_idx_y >= 0)                                                                         # debug
        #     & (bin_idx_y < n_bins_y)                                                                   # debug
        # )                                                                                              # debug
        # if debug_is_inside_limits:                                                                     # debug
        #     debug_dx_fine = right - left                                                               # debug
        #     debug_dx_coarse = right_x - left_x                                                         # debug
        #     debug_weight = debug_dx_fine / debug_dx_coarse                                             # debug
        # else:                                                                                          # debug
        #     debug_dx_fine = -1                                                                         # debug
        #     debug_dx_coarse = -1                                                                       # debug
        #     debug_weight = -1                                                                          # debug
        # print(                                                                                         # debug
        #     f"{debug_is_inside_limits and 'n' or 'i'} "                                                # debug
        #     f"seg {idx: 4d}: x {x0:0.2g},{x1:0.2g} → y {y0:0.2g},{y1:0.2g}"                            # debug
        #     " "                                                                                        # debug
        #     f"ex {bin_idx_x: 2d}: {left_x:0.2g}→{right_x:0.2g}"                                        # debug
        #     " "                                                                                        # debug
        #     f"ey {bin_idx_y: 2d}: {bottom_y:0.2g}→{top_y:0.2g}"                                        # debug
        #     " "                                                                                        # debug
        #     f"p{passed_any:d} "                                                                        # debug
        #     f"X{passed_x:d}{passed_x_first:d} "                                                        # debug
        #     f"Y{passed_top_y:d}{passed_top_y_first:d} "                                                # debug
        #     f"y{passed_bottom_y:d}{passed_bottom_y_first:d}"                                           # debug
        #     " "                                                                                        # debug
        #     f"fn {left:0.2g}→{right:0.2g}={debug_dx_fine:0.2g}"                                        # debug
        #     " "                                                                                        # debug
        #     f"cs {left_x:0.2g}→{right_x:0.2g}={debug_dx_coarse:0.2g}"                                  # debug
        #     " "                                                                                        # debug
        #     f"w {debug_weight}"                                                                        # debug
        #     " "                                                                                        # debug
        #     f"[{bin_idx_y}, {bin_idx_x}]"                                                              # debug
        #     " "                                                                                        # debug
        #     f"sk {skip_incomplete_x:d}"                                                                # debug
        # )                                                                                              # debug
        # fmt: on

        if not passed_any:
            idx += 1
            if idx >= idx_last_point:
                # print("break idx")                                                                     # debug
                break

            x0 = distortion_original[idx]
            y0 = distortion_target[idx]
            x1 = distortion_original[idx + 1]
            y1 = distortion_target[idx + 1]
            did_advance = True
            assert x1 > x0, "Allow only ascending x"

            continue

        is_inside_limits = (not skip_incomplete_x) & (
            (bin_idx_x >= 0) & (bin_idx_y >= 0) & (bin_idx_y < n_bins_y)
        )
        if is_inside_limits:
            width_x_partial = fabs(right - left)
            if width_x_partial != 0:
                element = width_x_partial / width_x_full

                # += takes into acount possibility to return to current element bu up/down or down/up movement
                matrix[bin_idx_y, bin_idx_x] += element

        skip_incomplete_x = False
        if left < right:
            left = right

        if passed_x_first:
            bin_idx_x += 1
            if bin_idx_x >= n_bins_x:
                # print("break idx x")                                                                   # debug
                break

            left_x = edges_original[bin_idx_x]
            did_advance = True
            if left_x > last_x:
                # print("break x")                                                                       # debug
                break

            right_x = edges_original[bin_idx_x + 1]
            width_x_full = right_x - left_x

        if passed_top_y_first:
            if bin_idx_y == n_bins_y:
                continue

            bin_idx_y += 1
            did_advance = True
            bottom_y = edges_target[bin_idx_y]
            if bin_idx_y == n_bins_y:
                top_y = large_positive_number
            else:
                top_y = edges_target[bin_idx_y + 1]
        elif passed_bottom_y_first:
            if bin_idx_y < 0:
                continue

            top_y = edges_target[bin_idx_y]
            bin_idx_y -= 1
            did_advance = True
            if bin_idx_y >= 0:
                bottom_y = edges_target[bin_idx_y]
                continue
            else:
                bottom_y = large_negative_number
    # else:                                                                                              # debug
    #     print("break due to lack of advancement")                                                      # debug


_axisdistortion_pointwise_numba: Callable[
    [NDArray, NDArray, NDArray, NDArray, NDArray], None
] = njit(cache=True)(_axisdistortion_pointwise_python)
