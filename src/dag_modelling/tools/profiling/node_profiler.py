from __future__ import annotations

from typing import TYPE_CHECKING

from pandas import DataFrame

from .timer_profiler import TimerProfiler

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...core.node import Node

_ALLOWED_GROUPBY = ("node", "type", "name")


class NodeProfiler(TimerProfiler):
    """Profiling class for estimating the time of individual nodes."""

    __slots__ = ()

    def __init__(
        self,
        target_nodes: Sequence[Node] = (),
        *,
        sources: Sequence[Node] = (),
        sinks: Sequence[Node] = (),
        filter_types: Sequence[type[Node] | str] | None = None,
        n_runs: int = 10_000,
    ):
        """Initialize a NodeProfiler.

        Parameters
        ----------
        target_nodes : Sequence[Node], optional
            The collection of nodes to profile. If `filter_types` is provided,
            only nodes matching the specified types or type names will be retained.
        sources : Sequence[Node], optional
            A collection of nodes used as starting points to find the subgraph of target_nodes,
            if target_nodes is not specified.
        sinks : Sequence[Node], optional
            A collection of nodes used as ending points to find the subgraph of target nodes,
            if target_nodes is not specified.
        filter_types : Sequence[type[Node] | str], optional
            A list of node classes or their class names. When specified, `target_nodes`
            will be filtered to include only nodes matching these types.
        n_runs : int, default=10000
            The number of times each node's function will be executed.
        """
        if filter_types:
            target_nodes = self._filter_nodes(target_nodes, filter_types)
        super().__init__(target_nodes, sources, sinks, n_runs)
        self._allowed_groupby = _ALLOWED_GROUPBY

    def _filter_nodes(self, nodes: Sequence[Node], node_types: Sequence[type[Node] | str]):
        type_names_to_accept = set(f if isinstance(f, str) else type(f).__name__ for f in node_types)

        nodes_accepted = []
        nodes_rejected_types = set()
        nodes_accepted_types = set()

        for node in nodes:
            if (node_type:=type(node).__name__) in type_names_to_accept:
                nodes_accepted.append(node)
                nodes_accepted_types.add(node_type)
            else:
                nodes_rejected_types.add(node_type)

        nodes_missing_types = type_names_to_accept.difference(nodes_accepted_types)

        print(f"Node types to accept ({len(type_names_to_accept)}): {', '.join(sorted(type_names_to_accept))}")
        print(f"Accepted node types ({len(nodes_accepted_types)}): {', '.join(sorted(nodes_accepted_types))}")
        print(f"Rejected node types ({len(nodes_rejected_types)}): {', '.join(sorted(nodes_rejected_types))}")
        print(f"Node types not seen ({len(nodes_missing_types)}): {', '.join(sorted(nodes_missing_types))}")

        return nodes_accepted

    @classmethod
    def estimate_node(cls, node: Node, n_runs: int = 10_000):
        for input in node.inputs.iter_all():
            input.touch()
        return cls._timeit_all_runs(stmt=node.function, n_runs=n_runs)

    def estimate_target_nodes(self) -> NodeProfiler:
        """Estimate all nodes in `self.target_nodes`.

        Return the current `NodeProfiler` instance.
        """
        self._estimations_table = DataFrame(
            {
                "node": (str(node) for node in self._target_nodes),
                "type": (type(node).__name__ for node in self._target_nodes),
                "name": (node.name for node in self._target_nodes),
                "time": (self.estimate_node(node, self._n_runs) for node in self._target_nodes),
            }
        )
        return self

    def make_report(
        self,
        *,
        group_by: str | Sequence[str] | None = "type",
        aggregations: Sequence[str] | None = None,
        sort_by: str | None = None,
        average_by_runs: bool = True,
    ) -> DataFrame:
        report = super().make_report(group_by=group_by, aggregations=aggregations, sort_by=sort_by)
        if average_by_runs:
            return self._compute_average(report)
        return report

    def _print_total_time(self):
        total = self._total_estimations_time()
        print(f"total estimations time / n_runs: {total / self._n_runs:.9f} sec.")
        print(f"total estimations time: {total:.6f} sec.")

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
            f"\nNode Profiling {hex(id(self))}, "
            f"n_runs for each node: {self._n_runs}\n"
            f"nodes in subgraph: {len(self._target_nodes)}\n"
            f"sources: {len(self._sources)}, sinks: {len(self._sinks)},\n"
            f"sort by: `{sort_by or 'default sorting'}`, "
            f"group by: `{group_by or 'no grouping'}`"
        )
        super()._print_table(report, rows)
        self._print_total_time()
        return report
