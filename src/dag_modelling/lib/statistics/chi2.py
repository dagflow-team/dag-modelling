from __future__ import annotations

from typing import TYPE_CHECKING

from numba import njit
from numpy import empty, square, subtract
from scipy.linalg import solve_triangular

from ...core.exception import TypeFunctionError
from ...core.type_functions import (
    AllPositionals,
    check_dimension_of_inputs,
    check_inputs_are_matrix_multipliable,
    check_inputs_are_square_matrices,
    check_inputs_have_same_shape,
    check_inputs_number_is_divisible_by_N,
    evaluate_dtype_of_outputs,
)
from ..abstract import ManyToOneNode

if TYPE_CHECKING:
    from numpy.typing import NDArray


@njit(cache=True)
def _chi2_1d_add(
    data: NDArray,
    theory: NDArray,
    errors: NDArray,
    result: NDArray,
) -> None:
    res = 0.0
    for idata, itheory, ierror in zip(data, theory, errors):
        diff = (itheory - idata) / ierror
        res += diff * diff
    result[0] += res


class Chi2(ManyToOneNode):
    r"""$\chi^{2}$ node.

    inputs:
        `0` or `data`: data (1d),
        `1` or `theory`: theory (1d),
        `2` or `errors`: errors (1d or 2d).

    outputs:
        `0` or `result`: the resulting array with only one element.

    extra arguments:
        `matrix_is_lower` (bool): True if the errors is lower triangular matrix else upper.
    """

    __slots__ = (
        "_data_tuple",
        "_theory_tuple",
        "_errors_tuple",
        "_triplets_tuple",
        "_matrix_is_lower",
        "_buffer",
    )

    _data_tuple: tuple[NDArray, ...]
    _theory_tuple: tuple[NDArray, ...]
    _errors_tuple: tuple[NDArray, ...]
    _triplets_tuple: tuple[tuple[NDArray, NDArray, NDArray], ...]
    _buffer: NDArray
    _matrix_is_lower: bool

    def __init__(self, name, *args, matrix_is_lower: bool = True, **kwargs):
        super().__init__(name, *args, output_name="result", **kwargs)
        self.labels.setdefaults(
            {
                "text": r"\chi$^{2}$",
                "plot_title": r"$\chi^{2}$",
                "latex": r"$\chi^{2}$",
                "axis": r"$\chi^{2}$",
                "mark": r"χ²",
            }
        )
        self._matrix_is_lower = matrix_is_lower
        self._data_tuple = ()  # input: 0
        self._theory_tuple = ()  # input: 1
        self._errors_tuple = ()  # input: 2
        self._triplets_tuple = ()
        self._functions_dict.update({"1d": self._function_1d, "2d": self._function_2d})

    @staticmethod
    def _input_names() -> tuple[str, ...]:
        return "data", "theory", "errors"

    @property
    def matrix_is_lower(self) -> bool:
        return self._matrix_is_lower

    def _function_1d(self) -> None:
        for callback in self._input_nodes_callbacks:
            callback()

        ret = self._output_data
        ret[0] = 0.0

        for data, theory, errors in self._triplets_tuple:
            _chi2_1d_add(data, theory, errors, ret)

    def _function_2d(self) -> None:
        for callback in self._input_nodes_callbacks:
            callback()

        buffer = self._buffer
        ret = 0.0
        for data, theory, errors in self._triplets_tuple:
            # errors is triangular decomposition of covariance matrix (L)
            subtract(data, theory, out=buffer)
            solve_triangular(
                errors, buffer, lower=self.matrix_is_lower, overwrite_b=True
            )
            square(buffer, out=buffer)
            ret += buffer.sum()

        self._output_data[0] = ret

    def _type_function(self) -> None:
        """A output takes this function to determine the dtype and shape."""
        check_inputs_number_is_divisible_by_N(self, 3)

        check_dimension_of_inputs(self, slice(0, None, 3), 1)
        check_dimension_of_inputs(self, slice(1, None, 3), 1)
        check_inputs_have_same_shape(self, (0, 1))
        check_inputs_have_same_shape(self, slice(0, None, 3))
        check_inputs_have_same_shape(self, slice(1, None, 3))
        check_inputs_have_same_shape(self, slice(2, None, 3))
        errors = self.inputs[2]
        dim = errors.dd.dim
        if dim == 2:
            check_inputs_are_square_matrices(self, "errors")
            check_inputs_are_matrix_multipliable(self, "errors", "data")
        elif dim == 1:
            check_inputs_have_same_shape(self, ("data", "errors"))
        else:
            raise TypeFunctionError(
                f"errors must be 1d or 2d, but given {dim}d!",
                node=self,
                input=errors,
            )
        self.function = self._functions_dict[f"{dim}d"]

        result = self.outputs[0]
        result.dd.shape = (1,)
        evaluate_dtype_of_outputs(
            self, AllPositionals, "result"
        )  # eval dtype of result

    def _post_allocate(self) -> None:
        super()._post_allocate()
        self._data_tuple = tuple(self._input_data[::3])  # input: 0
        self._theory_tuple = tuple(self._input_data[1::3])  # input: 1
        self._errors_tuple = tuple(self._input_data[2::3])  # input: 2

        self._triplets_tuple = tuple(
            tuple(x)
            for x in zip(self._data_tuple, self._theory_tuple, self._errors_tuple)
        )  # pyright: ignore [reportAttributeAccessIssue]

        # NOTE: buffer is needed only for 2d case
        if self._errors_tuple[0].ndim == 2:
            theory = self._theory_tuple[0]
            self._buffer = empty(shape=theory.shape, dtype=self._output_data.dtype)
