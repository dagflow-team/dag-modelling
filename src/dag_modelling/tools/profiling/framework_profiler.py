from __future__ import annotations

from typing import TYPE_CHECKING

from pandas import DataFrame, Series

from .timer_profiler import TimerProfiler

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Callable

    from numpy.typing import NDArray

    from ...core.node import Node


# it is possible to group by two columns
_ALLOWED_GROUPBY = (
    ("source nodes", "sink nodes", "n_nodes"),
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

    _replaced_fcns: dict[str, Callable]

    def __init__(
        self,
        target_nodes: Sequence[Node] = (),
        *,
        sources: Sequence[Node] = (),
        sinks: Sequence[Node] = (),
        n_runs=100,
        **kwargs,
    ):
        super().__init__(target_nodes, sources, sinks, n_runs, **kwargs)
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

    def _t_single_node(self, _s: Series) -> Series:
        """Return mean framework time normilized by one node.

        This as also an example of user-defined aggregate function
        """
        return Series({"t_node_average": mean(_s) / len(self._target_nodes)})

    def _taint_nodes(self):
        for node in self._sources:
            node.taint()

    @staticmethod
    def function_stub(node: Node):
        """An empty function stub of the Node that touches parent nodes to
        start a recursive execution of a graph (without computations)"""
        for input in node.inputs.iter_all():
            input.touch()

    @staticmethod
    def function_stub_with_threshold(node: Node, taint_threshold: int):
        """An empty function stub of the Node that touches parent nodes to
        start a recursive execution of a graph (without computations)"""
        for input in node.inputs.iter_all():
            input.touch_above_threshold(taint_threshold)

    def _set_functions_empty(self, *, with_threshold: bool = False):
        stub = with_threshold and self.function_stub_with_threshold or self.function_stub
        for node in self._target_nodes:
            self._replaced_fcns[node] = node.function
            # __get__ - a way to bind method to an instance
            node.function = stub.__get__(node)

    def _restore_functions(self):
        for node in self._target_nodes:
            node.function = self._replaced_fcns[node]
        self._replaced_fcns = {}

    def _estimate_framework_time_individual(self) -> NDArray:
        self._set_functions_empty()

        def evaluate_graph():
            for target_node in self._target_nodes:
                target_node.eval()

        evaluate_graph()  # touch all dependent nodes before estimations
        results = self._timeit_each_run(
            stmt=evaluate_graph, n_runs=self._n_runs, setup=self._taint_nodes, timer=self._timer
        )
        self._restore_functions()
        self._taint_nodes()
        return results

    def _estimate_taint_time(self) -> float:
        self._set_functions_empty(with_threshold=True)
        taint_threshold = max(source.tainted_value for source in self._sources) + 1

        def taint():
            nonlocal taint_threshold
            for node in self._sources:
                node.taint_below_threshold(taint_threshold)
            taint_threshold += 1

        result = self._timeit_all_runs(stmt=taint, n_runs=self._n_runs, timer=self._timer)
        self._restore_functions()
        self._taint_nodes()
        return result

    def _estimate_taint_and_touch_time(self) -> float:
        self._set_functions_empty(with_threshold=True)
        taint_threshold = max(source.tainted_value for source in self._sources) + 1

        def taint_and_eval():
            nonlocal taint_threshold
            for node in self._sources:
                node.taint_below_threshold(taint_threshold)
            for node in self._sinks:
                node.touch_above_threshold(taint_threshold)
            taint_threshold += 1

        result = self._timeit_all_runs(stmt=taint_and_eval, n_runs=self._n_runs, timer=self._timer)
        self._restore_functions()
        self._taint_nodes()
        return result

    def estimate_framework_time(self) -> FrameworkProfiler:
        self._start_process_time()
        t_taint = self._estimate_taint_time()
        t_taint_and_touch = self._estimate_taint_and_touch_time()
        results_individual = self._estimate_framework_time_individual()
        sources_col, sinks_col = self._shorten_sources_sinks()

        t_ind_total = results_individual.sum()
        t_ind_single = t_ind_total / self._n_runs
        t_ind_single_std = results_individual.std() / self._n_runs

        t_taint /= self._n_runs
        t_taint_and_touch /= self._n_runs

        n_nodes = len(self._target_nodes)
        t_ind_node = t_ind_single / n_nodes
        self._estimations_table = DataFrame(
            {
                "source nodes": [sources_col],
                "sink nodes": sinks_col,
                "t_taint_step": t_taint,
                "t_taint_node": t_taint / n_nodes,
                "t_step": t_taint_and_touch,
                "t_node": t_taint_and_touch / n_nodes,
                "t_ind_step": t_ind_single,
                "t_ind_step_std": t_ind_single_std,
                "t_ind_node": t_ind_node,
                "n_runs": self._n_runs,
                "n_nodes": n_nodes,
            }
        ).infer_objects()
        self._stop_process_time()
        return self

    def make_report(
        self,
        *,
        group_by: str | Sequence[str] | None = None,
        aggregations: Sequence[str] | None = None,
        sort_by: str | None = None,
    ) -> DataFrame:
        assert (group_by, aggregations, sort_by) == (None, None, None) # FIXME
        return self._estimations_table.copy()

    def print_report(
        self,
        *,
        rows: int | None = 100,
        group_by: str | Sequence[str] | None = None,
        aggregations: Sequence[str] | None = None,
        sort_by: str | None = None,
    ) -> DataFrame:
        assert (group_by, aggregations, sort_by) == (None, None, None) # FIXME
        report = self.make_report()
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
