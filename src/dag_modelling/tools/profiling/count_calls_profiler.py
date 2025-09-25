from __future__ import annotations

from typing import TYPE_CHECKING

from pandas import DataFrame

from .profiler import Profiler

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ...core.node import Node

# columnt aliases for aggregate functions
_COLUMN_ALIASES: dict[str | Callable, str] = {
    "mean": "single",
    "count": "node_count",
}

# The same names are needed to show possible aggregation functions in case of error
_AGGREGATE_ALIASES: dict[str, str | Callable] = {
    "single": "mean",
    "mean": "mean",
    "median": "median",
    "sum": "sum",
    "std": "std",
    "node_count": "count",
    "min": "min",
    "max": "max",
    "var": "var",
}


class CountCallsProfiler(Profiler):
    """Profiling class for estimating number of calls of each node after model
    fit."""

    __slots__ = ()

    def __init__(
        self,
        target_nodes: Sequence[Node] = (),
        *,
        sources: Sequence[Node] = (),
        sinks: Sequence[Node] = (),
    ):
        super().__init__(target_nodes, sources, sinks)
        self._primary_col = "calls"
        self._default_aggregations = ("count", "mean", "sum")
        self._aggregate_aliases = _AGGREGATE_ALIASES.copy()
        self._column_aliases = _COLUMN_ALIASES.copy()

    def estimate_calls(self) -> CountCallsProfiler:
        self._estimations_table = DataFrame(
            {
                "node": (str(node) for node in self._target_nodes),
                "type": (type(node).__name__ for node in self._target_nodes),
                "name": (node.name for node in self._target_nodes),
                "calls": (node.n_calls for node in self._target_nodes),
            }
        )
        return self

    def make_report(
        self,
        *,
        group_by: str | Sequence[str] | None = "type",
        aggregations: Sequence[str] | None = None,
        sort_by: str | None = None,
    ) -> DataFrame:
        return super().make_report(group_by=group_by, aggregations=aggregations, sort_by=sort_by)

    def calls_by_node(self):
        if not hasattr(self, "_estimations_table"):
            self.estimate_calls()
        count = len(self._target_nodes)
        calls = self._estimations_table["calls"].sum()
        return calls / count

    def print_report(
        self,
        *,
        rows: int | None = 100,
        group_by: str | Sequence[str] | None = "type",
        aggregations: Sequence[str] | None = None,
        sort_by: str | None = None,
    ) -> DataFrame:
        report = self.make_report(group_by=group_by, aggregations=aggregations, sort_by=sort_by)
        print(
            f"\nCounts of calls profiling {hex(id(self))}, "
            f"sort by: `{sort_by or 'default sorting'}`, "
            f"group by: `{group_by or 'no grouping'}`"
        )
        super()._print_table(report, rows)
        print(f"Mean count of calls by node: \t{self.calls_by_node():.2f}")
        return report
