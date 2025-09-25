from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import mean, ndarray
from pandas import DataFrame, Series

from .timer_profiler import TimerProfiler

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...core.node import Node


# it is possible to group by two columns
_ALLOWED_GROUPBY = (
    ("source nodes", "sink nodes"),
    "source nodes",
    "sink nodes",
)


class FrameworkProfiler(TimerProfiler):
    """Profiler class used to estimate the interaction time between nodes (i.e.
    "framework" time)

    The basic idea: replace the calculating functions of a node
    with empty stubs, while allowing the graph to be executed as usual
    """

    __slots__ = "_replaced_fcns"

    def __init__(
        self,
        target_nodes: Sequence[Node] = (),
        *,
        sources: Sequence[Node] = (),
        sinks: Sequence[Node] = (),
        n_runs=100,
    ):
        super().__init__(target_nodes, sources, sinks, n_runs)
        self._allowed_groupby = _ALLOWED_GROUPBY
        self.register_aggregate_func(
            func=self._t_single_node,
            aliases=[
                "t_node_average",
                "node_average",
            ],
            column_name="t_node_average",
        )
        self._default_aggregations = ("count", "single", "sum", "t_node_average")
        self._replaced_fcns = {}
        if not (self._sources and self._sinks):
            self._reveal_source_sink()

    def _t_single_node(self, _s: Series) -> Series:
        """Return mean framework time normilized by one node.

        This as also an example of user-defined aggregate function
        """
        return Series({"t_node_average": mean(_s) / len(self._target_nodes)})

    def _taint_nodes(self):
        for node in self._target_nodes:
            node.taint()

    @staticmethod
    def function_stub(node: Node):
        """An empty function stub of the Node that touches parent nodes to
        start a recursive execution of a graph (without computations)"""
        for input in node.inputs.iter_all():
            input.touch()

    def _set_functions_empty(self):
        for node in self._target_nodes:
            self._replaced_fcns[node] = node.function
            # __get__ - a way to bind method to an instance
            node.function = self.function_stub.__get__(node)

    def _restore_functions(self):
        for node in self._target_nodes:
            node.function = self._replaced_fcns[node]
        self._replaced_fcns = {}

    def _estimate_framework_time(self) -> ndarray:
        self._set_functions_empty()

        def evaluate_graph():
            for sink_node in self._sinks:
                sink_node.eval()

        evaluate_graph()  # touch all dependent nodes before estimations
        results = self._timeit_each_run(
            stmt=evaluate_graph,
            n_runs=self._n_runs,
            setup=self._taint_nodes,
        )
        self._restore_functions()
        self._taint_nodes()
        return results

    def estimate_framework_time(self) -> FrameworkProfiler:
        results = self._estimate_framework_time()
        sources_col, sinks_col = self._shorten_sources_sinks()
        self._estimations_table = DataFrame(
            {"source nodes": sources_col, "sink nodes": sinks_col, "time": results}
        )
        return self

    def make_report(
        self,
        *,
        group_by: str | Sequence[str] | None = ("source nodes", "sink nodes"),
        aggregations: Sequence[str] | None = None,
        sort_by: str | None = None,
    ) -> DataFrame:
        return super().make_report(group_by=group_by, aggregations=aggregations, sort_by=sort_by)

    def print_report(
        self,
        *,
        rows: int | None = 100,
        group_by: str | Sequence[str] | None = ("source nodes", "sink nodes"),
        aggregations: Sequence[str] | None = None,
        sort_by: str | None = None,
    ) -> DataFrame:
        report = self.make_report(group_by=group_by, aggregations=aggregations, sort_by=sort_by)
        print(
            f"\nFramework Profiling {hex(id(self))}, "
            f"n_runs for given subgraph: {self._n_runs},\n"
            f"nodes in subgraph: {len(self._target_nodes)}, "
            f"sources: {len(self._sources)}, sinks: {len(self._sinks)},\n"
            f"sort by: `{sort_by or 'default sorting'}`, "
            f"group by: `{group_by or 'no grouping'}`"
        )
        super()._print_table(report, rows)
        return report
