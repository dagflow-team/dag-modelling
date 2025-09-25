from __future__ import annotations

from textwrap import shorten
from time import perf_counter_ns
from typing import TYPE_CHECKING

from numpy import empty, ndarray
from numpy import sum as npsum
from pandas import DataFrame, Series

from ...core.node import Node
from .profiler import Profiler

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ...core.node import Node


# prefix `t_` - time notation
# columnt aliases for aggrigate functions
_COLUMN_ALIASES: dict[str | Callable, str] = {
    "mean": "t_single",
    "single": "t_single",
    "t_mean": "t_single",
    "median": "t_median",
    "sum": "t_total",
    "total": "t_total",
    "t_sum": "t_total",
    "std": "t_std",
    "t_count": "count",
    "min": "t_min",
    "max": "t_max",
    "var": "t_var",
}
_AGGREGATE_ALIASES: dict[str, str | Callable] = {
    "single": "mean",
    "t_single": "mean",
    "t_mean": "mean",
    "t_median": "median",
    "t_sum": "sum",
    "total": "sum",
    "t_total": "sum",
    "t_std": "std",
    "t_count": "count",
    "t_min": "min",
    "t_max": "max",
    "t_var": "var",
}

_DEFAULT_AGGREGATIONS = ("count", "single", "total", "fraction_percent")

SOURCE_COL_WIDTH = 32
SINK_COL_WIDTH = 32


class TimerProfiler(Profiler):
    """Base class for time-related profiling.

    The `"time"` column is used to store the measured values.

    It is not designed to be used directly,
    you should consider `NodeProfiler`,
    `FrameworkProfiler` or `FitSimulationProfiler`
    """

    __slots__ = ("_n_runs", "_timer")
    _n_runs: int

    def __init__(
        self,
        target_nodes: Sequence[Node] = (),
        sources: Sequence[Node] = (),
        sinks: Sequence[Node] = (),
        n_runs: int = 100,
    ):
        self._default_aggregations = _DEFAULT_AGGREGATIONS
        self._column_aliases = _COLUMN_ALIASES.copy()
        self._aggregate_aliases = _AGGREGATE_ALIASES.copy()
        self._n_runs = n_runs
        self.register_aggregate_func(
            func=self._t_percentage,
            aliases=["fraction_percent", "%_of_total", "percentage", "t_percentage"],
            column_name="fraction_percent",
        )
        super().__init__(target_nodes, sources, sinks)
        self._primary_col = "time"

    @property
    def n_runs(self) -> int:
        return self._n_runs

    @n_runs.setter
    def n_runs(self, value):
        self._n_runs = value

    @classmethod
    def _timeit_all_runs(
        cls, stmt: Callable, n_runs: int, setup: Callable | None = None
    ) -> float:
        """Estimate total time of the statement running multiple times. Use
        `time.perf_counter_ns` internally to measure time in nanoseconds and
        then convert it to seconds by dividing the total time by 1 billion.

        Args:
            stmt (Callable): The function/method whose execution time is measured.
            n_runs (int): The number of executions.
            setup (Callable | None, optional): Preliminary actions that are executed
            before each function measurement.

        Returns:
            float: Execution time in seconds for `n_runs`.
        """
        timer = perf_counter_ns
        total_nanoseconds = 0
        for _ in range(n_runs):
            if setup is not None:
                setup()
            t_0 = timer()
            stmt()
            t_1 = timer()
            total_nanoseconds += t_1 - t_0
        return total_nanoseconds / 1e9

    @classmethod
    def _timeit_each_run(
        cls, stmt: Callable, n_runs: int, setup: Callable | None = None
    ) -> ndarray:
        """Estimate execution time of the statement for each run. Similar to
        `self.estimate_runs`, but returns a `numpy.ndarray` where each value is
        the execution time of the i-th run.

        Args:
            stmt (Callable): The function/method whose execution time is measured.
            n_runs (int): The number of executions.
            setup (Callable | None, optional): Preliminary actions that are executed
            before each function measurement.

        Returns:
            ndarray: Execution time in seconds of `stmt` for each run with the shape `(n_runs, )`.
        """
        timer = perf_counter_ns
        nanoseconds = empty(n_runs)
        for i in range(n_runs):
            if setup is not None:
                setup()
            t_0 = timer()
            stmt()
            t_1 = timer()
            nanoseconds[i] = t_1 - t_0
        return nanoseconds / 1e9

    def _t_percentage(self, _s: Series) -> Series:
        """User-defined aggregate function to calculate the percentage of group
        given as `pandas.Series`."""
        total = self._total_estimations_time()
        if total == 0:
            raise ZeroDivisionError('The total calculated "time" is zero!')
        return Series({"%_of_total": npsum(_s) * 100 / total})

    def _total_estimations_time(self):
        return self._estimations_table["time"].sum()

    def _compute_average(self, df: DataFrame) -> DataFrame:
        """Normalize `time` by `self.n_runs`"""
        for c in df.columns:
            if c.startswith("t_") or c == "time":
                df[c] /= self._n_runs
        return df

    def _shorten_names(self, nodes, max_length) -> str:
        """Get a string representation of names of the `nodes`, truncated to
        not exceed `max_length`.

        Note: This implementation is generally faster than directly applying
        `shorten(str([n.name for n in nodes]), max_length)`.
        """
        names = []
        names_sum_length = 0
        for node in nodes:
            if names_sum_length > max_length:
                break
            names.append(node.name)
            names_sum_length += len(node.name)
        return shorten(", ".join(names), max_length)

    def _shorten_sources_sinks(self) -> tuple[str, str]:
        """Get a short string representation of `sources` and `sinks` names
        with truncation by `SOURCE_COL_WIDTH`, `SINK_COL_WIDTH` lengths.

        Return pair (short names of __sources__, short names of
        __sinks__)
        """
        return (
            self._shorten_names(self._sources, SOURCE_COL_WIDTH),
            self._shorten_names(self._sinks, SINK_COL_WIDTH),
        )
