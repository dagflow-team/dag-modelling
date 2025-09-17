from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING, Literal

from numba import njit
from numpy import double, exp, integer, log

from ...core.exception import InitializationError
from ...core.node import Node
from ...core.type_functions import (
    assign_axes_from_inputs_to_outputs,
    check_dimension_of_inputs,
    check_dtype_of_inputs,
    check_node_has_inputs,
    check_number_of_inputs,
    check_shape_of_inputs,
    copy_from_inputs_to_outputs,
)

if TYPE_CHECKING:
    from typing import Callable

    from numpy.typing import NDArray

    from ...core.input import Input
    from ...core.output import Output


class ExtrapolationStrategy(IntEnum):
    constant = 0
    nearestedge = 1
    extrapolate = 2


MethodType = Literal["linear", "log", "logx", "exp", "left", "right", "nearest"]
OutOfBoundsStrategyType = Literal["constant", "nearestedge", "extrapolate"]


class InterpolatorCore(Node):
    """Interpolator core class.

    self.inputs:
        `0` or `y`: array of the `y=f(coarse)`
        `coarse`: array of the coarse x points
        `fine`: array of the fine x points
        `indices`: array of the indices of the coarse segments for every fine point.

    self.outputs:
        `0` or `result`: array of the `y≈f(fine)`

    extra arguments:
        `method`: defines an interpolation method ("linear", "log", "logx", "exp", "left", "right", "nearest");
        default: `linear`
        `underflow`: defines the underflow strategy: `constant`, `nearestedge`, or `extrapolate`;
        default: `extrapolate`
        `overflow`: defines the overflow strategy: `constant`, `nearestedge`, or `extrapolate`;
        default: `extrapolate`
        `fillvalue`: defines the filling value for the `constant` strategy;
        default: `0.0`

    The node performs interpolation of the `coarse` points with `y=f(coarse)`
    to `fine` points and calculates `y≈f(fine)`.
    """

    __slots__ = (
        "_strategies",
        "_methods",
        "_method",
        "_methodname",
        "_underflow",
        "_overflow",
        "_fillvalue",
        "_y_input",
        "_coarse_input",
        "_fine_input",
        "_indices_input",
        "_result_output",
        "_y",
        "_coarse",
        "_fine",
        "_indices",
        "_result",
    )

    _y_input: Input
    _coarse_input: Input
    _fine_input: Input
    _indices_input: Input
    _result_output: Output

    _y: NDArray
    _coarse: NDArray
    _fine: NDArray
    _indices: NDArray
    _result: NDArray

    _methods: dict[str, Callable]
    _method: Callable
    _methodname: str

    def __init__(
        self,
        *args,
        method: MethodType = "linear",
        underflow: OutOfBoundsStrategyType = "extrapolate",
        overflow: OutOfBoundsStrategyType = "extrapolate",
        fillvalue: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(
            *args, **kwargs, allowed_kw_inputs=("y", "coarse", "fine", "indices")
        )
        self._labels.setdefault("mark", "~")
        self._methodname = method
        self._methods = {
            "linear": _linear_interpolation,
            "log": _log_interpolation,
            "logx": _logx_interpolation,
            "exp": _exp_interpolation,
            "left": _left_interpolation,
            "right": _right_interpolation,
            "nearest": _nearest_interpolation,
        }
        if (mlist := self._methods.keys()) and method not in mlist:
            raise InitializationError(
                f"Argument 'method' must be in {mlist}, but given {method}!",
                node=self,
            )
        self._method = self._methods[method]
        self._strategies = {"constant": 0, "nearestedge": 1, "extrapolate": 2}
        slist = self.strategies.keys()
        if underflow not in slist:
            raise InitializationError(
                f"Argument 'underflow' must be in {slist}, but given {underflow}!",
                node=self,
            )
        if overflow not in slist:
            raise InitializationError(
                f"Argument 'overflow' must be in {slist}, but given {overflow}!",
                node=self,
            )
        self._underflow = underflow
        self._overflow = overflow
        self._fillvalue = fillvalue
        # inputs/outputs
        self._y_input = self._add_input("y")
        self._coarse_input = self._add_input("coarse", positional=False)
        self._fine_input = self._add_input("fine", positional=False)
        self._indices_input = self._add_input("indices", positional=False)
        self._result_output = self._add_output("result")

        self._functions_dict.update(
            {
                "python": self._function_python,
                "numba": self._function_numba,
            }
        )
        self.choose_function("numba")

    @property
    def methods(self) -> dict:
        return self._methods

    @property
    def strategies(self) -> dict:
        return self._strategies

    @property
    def underflow(self) -> str:
        return self._underflow

    @property
    def overflow(self) -> str:
        return self._overflow

    @property
    def fillvalue(self) -> float:
        return self._fillvalue

    def _type_function(self) -> None:
        """The function to determine the dtype and shape.

        Checks self.inputs dimension and, selects an interpolation
        algorithm, determines dtype and shape for self.outputs
        """
        check_number_of_inputs(self, 1)
        check_node_has_inputs(self, ("coarse", "y", "fine", "indices"))
        check_dimension_of_inputs(self, ("coarse", "y"), 1)
        check_dtype_of_inputs(self, "indices", dtype="i")

        ncoarse = self._coarse_input.dd.shape[0]
        if self._methodname == "left":
            check_shape_of_inputs(self, "y", (ncoarse,), (ncoarse - 1,))
        else:
            check_shape_of_inputs(self, "y", (ncoarse,))
        check_shape_of_inputs(self, "fine", self._indices_input.dd.shape)
        copy_from_inputs_to_outputs(self, "fine", "result")
        if self._fine_input.dd.dim == 1:
            assign_axes_from_inputs_to_outputs(
                self,
                "fine",
                "result",
                assign_meshes=True,
                ignore_assigned=False,
                merge_input_axes=True,
            )
        else:
            # TODO: add a choice of what axis to overwrite
            assign_axes_from_inputs_to_outputs(
                self,
                "fine",
                "result",
                assign_meshes=True,
                ignore_assigned=False,
                overwrite_assigned=True,
                ignore_inconsistent_number_of_meshes=True,
                merge_input_axes=True,
            )

    def _post_allocate(self):
        super()._post_allocate()
        self._y = self._y_input._data.ravel()
        self._coarse = self._coarse_input._data.ravel()
        self._fine = self._fine_input._data.ravel()
        self._indices = self._indices_input._data.ravel()
        self._result = self._result_output._data.ravel()

    def _function_numba(self):
        """Runs interpolation method chosen within `method` arg."""
        # TODO: inherit from OneToOneNode, loop inputs
        for callback in self._input_nodes_callbacks:
            callback()

        _interpolation_numba(
            self._method,
            self._coarse,
            self._y,
            self._fine,
            self._indices,
            self._result,
            self.strategies[self.underflow],
            self.strategies[self.overflow],
            self.fillvalue,
        )

    def _function_python(self):
        """Runs interpolation method chosen within `method` arg."""
        # TODO: inherit from OneToOneNode, loop inputs
        for callback in self._input_nodes_callbacks:
            callback()

        _interpolation_python(
            self._method,
            self._coarse,
            self._y,
            self._fine,
            self._indices,
            self._result,
            self.strategies[self.underflow],
            self.strategies[self.overflow],
            self.fillvalue,
        )


def _interpolation_python(
    method: Callable[[float, float, float, float, float], float],
    coarse: NDArray[double],
    yc: NDArray[double],
    fine: NDArray[double],
    indices: NDArray[integer],
    result: NDArray[double],
    underflow: int,
    overflow: int,
    fillvalue: float,
) -> None:
    nseg = coarse.size - 1
    has_last_y_input = coarse.size == yc.size
    for i, j in enumerate(indices):
        if j > nseg:  # overflow
            if overflow == ExtrapolationStrategy.constant:  # constant
                result[i] = fillvalue
            elif overflow == ExtrapolationStrategy.nearestedge:  # nearestedge
                result[i] = yc[nseg]
            elif has_last_y_input:  # extrapolate
                result[i] = method(
                    coarse[nseg - 1],
                    coarse[nseg],
                    yc[nseg - 1],
                    yc[nseg],
                    fine[i],
                )
            else:  # extrapolate
                result[i] = method(
                    coarse[nseg - 1],
                    coarse[nseg],
                    yc[nseg - 1],
                    yc[nseg - 1],
                    fine[i],
                )
        elif j <= 0:  # underflow
            if underflow == ExtrapolationStrategy.constant:  # constant
                result[i] = fillvalue
            elif underflow == ExtrapolationStrategy.nearestedge:  # nearestedge
                result[i] = yc[0]
            else:  # extrapolate
                result[i] = method(
                    coarse[0],
                    coarse[1],
                    yc[0],
                    yc[1],
                    fine[i],
                )
        elif has_last_y_input or j < nseg:  # interpolate
            result[i] = method(coarse[j - 1], coarse[j], yc[j - 1], yc[j], fine[i])
        else:  # interpolate
            result[i] = method(coarse[j - 1], coarse[j], yc[j - 1], yc[j - 1], fine[i])


_interpolation_numba: Callable = njit(cache=True)(_interpolation_python)


@njit(cache=True, inline="always")
def _linear_interpolation(
    coarse0: float,
    coarse1: float,
    yc0: float,
    yc1: float,
    fine: float,
) -> float:
    return yc0 + (fine - coarse0) * (yc1 - yc0) / (coarse1 - coarse0)


@njit(cache=True, inline="always")
def _log_interpolation(
    coarse0: float,
    coarse1: float,
    yc0: float,
    yc1: float,
    fine: float,
) -> float:
    return log(
        exp(yc0) + (fine - coarse0) * (exp(yc1) - exp(yc0)) / (coarse1 - coarse0)
    )


@njit(cache=True, inline="always")
def _logx_interpolation(
    coarse0: float,
    coarse1: float,
    yc0: float,
    yc1: float,
    fine: float,
) -> float:
    return yc0 + log(fine / coarse0) * (yc1 - yc0) / log(coarse1 / coarse0)


@njit(cache=True, inline="always")
def _exp_interpolation(
    coarse0: float,
    coarse1: float,
    yc0: float,
    yc1: float,
    fine: float,
) -> float:
    return yc0 * exp((fine - coarse0) * log(yc1 / yc0) / (coarse1 - coarse0))


@njit(cache=True, inline="always")
def _left_interpolation(
    coarse0: float,
    coarse1: float,
    yc0: float,
    yc1: float,
    fine: float,
) -> float:
    return yc0


@njit(cache=True, inline="always")
def _right_interpolation(
    coarse0: float,
    coarse1: float,
    yc0: float,
    yc1: float,
    fine: float,
) -> float:
    return yc1


@njit(cache=True, inline="always")
def _nearest_interpolation(
    coarse0: float,
    coarse1: float,
    yc0: float,
    yc1: float,
    fine: float,
) -> float:
    return yc0 if (fine - coarse0) <= (coarse1 - fine) else yc1
