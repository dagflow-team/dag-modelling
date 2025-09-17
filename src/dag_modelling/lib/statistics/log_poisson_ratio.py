from __future__ import annotations

from typing import TYPE_CHECKING

from numba import njit
from numpy import log

from ..abstract import ManyToOneNode

if TYPE_CHECKING:
    from numpy.typing import NDArray


@njit(cache=True)
def _log_poisson_ratio(
    data: NDArray,
    theory: NDArray,
) -> float:
    r"""$2\sum (theory_i - data_i + data_i * log(data_i/theory_i))$"""
    res = 0.0
    for i in range(theory.size):
        d = data[i]
        t = theory[i]
        if d > 0:
            res += t - d + d * log(d / t)
        else:
            res += t

    return 2.0 * res


class LogPoissonRatio(ManyToOneNode):
    r"""Log Poisson ratio node.

    Computes:
    $\chi^2 = -2log (P(data|theory)/P(data|data)) =
            = 2\sum (theory_i - data_i + data_i * log(data_i/theory_i))$

    inputs:
        `0` or `data`: data,
        `1` or `theory`: theory,

    outputs:
        `0` or `result`: the resulting array with only one element.
    """

    __slots__ = (
        "_data_tuple",
        "_theory_tuple",
        "_pairs_tuple",
    )

    _data_tuple: tuple[NDArray, ...]
    _theory_tuple: tuple[NDArray, ...]
    _pairs_tuple: tuple[tuple[NDArray, NDArray], ...]

    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, output_name="result", **kwargs)
        self.labels.setdefaults(
            {
                "text": r"log Poisson ratio",
                "mark": r"log(P)",
            }
        )
        self._data_tuple = ()  # input: 0
        self._theory_tuple = ()  # input: 1
        self._pairs_tuple = ()

    @staticmethod
    def _input_names() -> tuple[str, ...]:
        return "data", "theory"

    def _function(self) -> None:
        for callback in self._input_nodes_callbacks:
            callback()

        res = 0.0
        for data, theory in self._pairs_tuple:
            res += _log_poisson_ratio(data, theory)

        self._output_data[0] = res

    def _type_function(self) -> None:
        """A output takes this function to determine the dtype and shape."""
        super()._type_function()
        self.outputs[0].dd.shape = (1,)

    def _post_allocate(self) -> None:
        super()._post_allocate()
        self._data_tuple = tuple(self._input_data[::2])  # input: 0
        self._theory_tuple = tuple(self._input_data[1::2])  # input: 1

        self._pairs_tuple = tuple(
            tuple(x)
            for x in zip(
                self._data_tuple, self._theory_tuple
            )  # pyright: ignore [reportAttributeAccessIssue]
        )
