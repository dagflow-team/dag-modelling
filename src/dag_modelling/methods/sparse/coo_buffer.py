from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import zeros

from ...core.csr_array import CSRArray
from .coo import coo_to_csr, coo_to_dense
from .csr import with_headroom

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


class COOBuffer:
    """Work arrays for the entries of a matrix in the COO layout, given column by column.

    A kernel `kernel(*args, rows, cols, values)` puts the entries with non-decreasing columns
    and returns their number, counting also those that did not fit. `write()` puts them to a
    dense matrix or to `CSRArray`, a repeated cell is summed.
    """

    __slots__ = ("rows", "cols", "values", "_starts")

    rows: NDArray
    cols: NDArray
    values: NDArray
    _starts: NDArray

    def __init__(self, capacity: int, nrows: int):
        self._allocate(capacity)
        self._starts = zeros(nrows + 1, "i4")

    def _allocate(self, capacity: int) -> None:
        self.rows, self.cols = zeros(capacity, "i4"), zeros(capacity, "i4")
        self.values = zeros(capacity)

    @property
    def capacity(self) -> int:
        return self.rows.size

    # `size` and `nbytes` as an array has, for `MemoryProfiler`
    size = capacity

    @property
    def nbytes(self) -> int:
        return self.rows.nbytes + self.cols.nbytes + self.values.nbytes + self._starts.nbytes

    def fill(self, output: NDArray | CSRArray, kernel: Callable[..., int], *args) -> None:
        """Collect the entries with `kernel` and write them to `output`.

        If the entries did not fit, the arrays grow and the kernel runs again.
        """
        count = kernel(*args, self.rows, self.cols, self.values)
        if count > self.capacity:
            self._allocate(with_headroom(count))
            kernel(*args, self.rows, self.cols, self.values)
        self.write(count, output)

    def write(self, count: int, output: NDArray | CSRArray) -> None:
        """Put the first `count` entries to `output`"""
        if count > self.capacity:
            raise RuntimeError(f"COOBuffer: {count} entries, but room for {self.capacity} only")
        rows, cols, values = self.rows[:count], self.cols[:count], self.values[:count]
        if isinstance(output, CSRArray):
            if count > output.capacity:
                output.reserve(self.capacity)
            coo_to_csr(rows, cols, values, output.data, output.indices, output.indptr, self._starts)
            output.update()
        else:
            coo_to_dense(rows, cols, values, output)
