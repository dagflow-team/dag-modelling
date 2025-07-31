from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from numba import njit
from numpy import finfo, result_type

from ...core.exception import CalculationError, InitializationError
from ...core.node import Node
from ...core.type_functions import (
    check_dimension_of_inputs,
    check_number_of_inputs,
    copy_from_inputs_to_outputs,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ...core.input import Input
    from ...core.output import Output


@njit(cache=True)
def _is_sorted(array: NDArray) -> bool:
    previous = array[0]
    for i in range(1, len(array)):
        current = array[i]

        if current <= previous:
            return False

        previous = current

    return True


@njit(cache=True)
def _shift_last_edge_inside_right(
    fine: NDArray, idxs: NDArray, coarse: NDArray, tolerance: float
) -> None:
    overflow_idx = coarse.size
    last_edge = coarse[overflow_idx - 1]
    for i, idx in enumerate(idxs):
        if idx == overflow_idx and (fine[i] - tolerance) <= last_edge:
            idxs[i] = overflow_idx - 1


@njit(cache=True)
def _shift_last_edge_inside_left(
    fine: NDArray, idxs: NDArray, coarse: NDArray, tolerance: float
) -> None:
    first_edge = coarse[0]
    for i, idx in enumerate(idxs):
        if idx == 0 and (fine[i] + tolerance) >= first_edge:
            idxs[i] = 1


class SegmentIndex(Node):
    """
    inputs:
        `0` or `coarse`: array of the coarse x points
        `1` or  `fine`: array of the fine x points

    outputs:
        `0` or `indices`: array of the indices of the coarse segments for every fine point

    The node finds an index of the segment in the coarse array for every fine point.
    The node uses `numpy.searchsorted` method. There is an extra argument `mode`:
        `left`: `a[i-1] < v <= a[i]`
        `right`: `a[i-1] <= v < a[i]`
    """

    __slots__ = (
        "_mode",
        "_coarse",
        "_fine",
        "_indices",
        "_tolerance",
        "_tolerances",
    )

    _coarse: Input
    _fine: Input
    _indices: Output
    _tolerance: float | None
    _tolerances: dict[str, float]

    def __init__(
        self,
        *args,
        mode: Literal["left", "right"] = "right",
        tolerances: dict[str, float] = {"f": 1e-4, "d": 1e-10},
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs, allowed_kw_inputs=("coarse", "fine"))
        self._labels.setdefault("mark", "[i]")
        if mode not in {"left", "right"}:
            raise InitializationError(
                f"Argument `mode` must be 'left' or 'right', but given '{mode}'!",
                node=self,
            )
        self._mode = mode
        self._coarse = self._add_input("coarse")  # 0
        self._fine = self._add_input("fine")  # 1
        self._indices = self._add_output("indices")  # 0
        self._tolerance = None
        self._tolerances = tolerances

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def tolerance(self):
        return self._tolerance

    def _type_function(self) -> None:
        """The function to determine the dtype and shape of the ouput."""
        check_dimension_of_inputs(self, ("coarse",), 1)
        check_number_of_inputs(self, 2)
        copy_from_inputs_to_outputs(
            self, 1, 0, dtype=False, shape=True, edges=False, meshes=False
        )
        self._indices.dd.dtype = "i"

        dtype = result_type(*(inp.dd.dtype for inp in self.inputs))
        assert dtype == "d" or dtype == "f"
        self._tolerance = self._tolerances[dtype.char]

    def _function(self):
        """Uses `numpy.ndarray.searchsorted` and `numpy.ndarray.argsort`"""
        out = self._indices._data.ravel()
        coarse = self._coarse.data.ravel()
        fine = self._fine.data.ravel()
        if not _is_sorted(coarse):
            raise CalculationError(
                "Coarse array is not sorted", node=self, input=self._coarse
            )
        out[:] = coarse.searchsorted(fine, side=self.mode)
        if self.mode == "right":
            _shift_last_edge_inside_right(fine, out, coarse, self._tolerance)
        else:
            _shift_last_edge_inside_left(fine, out, coarse, self._tolerance)
