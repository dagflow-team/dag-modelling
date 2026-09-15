from __future__ import annotations

from typing import TYPE_CHECKING, override

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
        keep_types: Sequence[type[Node] | str] = (),
        filter_frozen: bool = False,
        filter_not_tainted: bool = False,
        n_runs: int = 10_000,
        **kwargs,
    ):
        """Initialize a NodeProfiler.

        Parameters
        ----------
        target_nodes : Sequence[Node], optional
            The collection of nodes to profile. If `keep_types` is provided,
            only nodes matching the specified types or type names will be retained.
        sources : Sequence[Node], optional
            A collection of nodes used as starting points to find the subgraph of target_nodes,
            if target_nodes is not specified.
        sinks : Sequence[Node], optional
            A collection of nodes used as ending points to find the subgraph of target nodes,
            if target_nodes is not specified.
        keep_types : Sequence[type[Node] | str], optional
            A list of node classes or their class names. When specified, `target_nodes`
            will be filtered to include only nodes matching these types.
        filter_frozen : bool, default=False
            Filter out frozen nodes
        filter_not_tainted : bool, default=False
            Filter out not tainted nodes
        n_runs : int, default=10000
            The number of times each node's function will be executed.
        """
        if keep_types or filter_frozen or filter_not_tainted:
            target_nodes = self._filter_nodes(
                target_nodes,
                keep_types=keep_types,
                filter_frozen=filter_frozen,
                filter_not_tainted=filter_not_tainted,
            )
        super().__init__(target_nodes, sources, sinks, n_runs, **kwargs)
        self._allowed_groupby = _ALLOWED_GROUPBY

    def _filter_nodes(
        self,
        nodes: Sequence[Node],
        *,
        keep_types: Sequence[type[Node] | str],
        filter_frozen: bool = False,
        filter_not_tainted: bool = False,
    ):
        nodes_accepted = nodes
        type_names_to_accept = set(f if isinstance(f, str) else f.__name__ for f in keep_types)
        if keep_types:
            nodes_accepted = []

            nodes_rejected = []
            nodes_rejected_types = set()
            nodes_accepted_types = set()
            for node in nodes:
                if (node_type := type(node).__name__) in type_names_to_accept:
                    nodes_accepted.append(node)
                    nodes_accepted_types.add(node_type)
                else:
                    nodes_rejected_types.add(node_type)
                    nodes_rejected.append(node)

            nodes_missing_types = type_names_to_accept.difference(nodes_accepted_types)

            print(
                f"Node types to accept ({len(type_names_to_accept)}): {', '.join(sorted(type_names_to_accept)) if type_names_to_accept else 'all'}"
            )
            print(
                f"Rejected nodes by filter ({len(nodes_rejected)}): {', '.join(sorted(nodes_rejected_types))}"
            )
            print(
                f"Node types not seen ({len(nodes_missing_types)}): {', '.join(sorted(nodes_missing_types))}"
            )

        if filter_not_tainted:
            nodes_accepted_tainted = []
            nodes_removed_not_tainted = []
            for node in nodes_accepted:
                if node.tainted:
                    nodes_accepted_tainted.append(node)
                else:
                    nodes_removed_not_tainted.append(node)

            types_removed_not_tainted = {type(node).__name__ for node in nodes_removed_not_tainted}
            nodes_accepted = nodes_accepted_tainted

            print(
                f"Nodes removed as not tainted ({len(nodes_removed_not_tainted)}): {', '.join(sorted(types_removed_not_tainted))}"
            )

        if filter_frozen:
            nodes_accepted_not_frozen = []
            nodes_frozen = []
            for node in nodes_accepted:
                if node.frozen:
                    nodes_frozen.append(node)
                else:
                    nodes_accepted_not_frozen.append(node)

            types_frozen = {type(node).__name__ for node in nodes_frozen}
            nodes_accepted = nodes_accepted_not_frozen

            print(
                f"Nodes removed as frozen ({len(nodes_frozen)}): {', '.join(sorted(types_frozen))}"
            )

        nodes_accepted_types = {type(node).__name__ for node in nodes_accepted}
        print(f"Accepted nodes ({len(nodes_accepted)}): {', '.join(sorted(nodes_accepted_types))}")

        return nodes_accepted

    @classmethod
    def estimate_node(cls, node: Node, n_runs: int = 10_000, **kwargs):
        for input in node.inputs.iter_all():
            input.touch()
        return cls._timeit_all_runs(stmt=node.function, n_runs=n_runs, **kwargs)

    def estimate_node_input_array_elements(self, node: Node) -> int:
        return sum(input.dd.size for input in node.inputs.iter_all())

    def estimate_target_nodes(self) -> NodeProfiler:
        """Estimate all nodes in `self.target_nodes`.

        Return the current `NodeProfiler` instance.
        """
        self._start_process_time()
        columns = ("node", "type", "name", "path", "time", "n_input_array_elements")
        index = range(len(self._target_nodes))
        self._estimations_table = (df := DataFrame(columns=columns, index=index))
        for i, node in enumerate(self._target_nodes):
            node_i = str(node)
            type_i = type(node).__name__
            name_i = node.name
            path_i = node.labels.path
            time_i = self.estimate_node(node, self._n_runs, timer=self._timer) / self._n_runs
            n_input_array_elements = self.estimate_node_input_array_elements(node)
            df.iloc[i] = node_i, type_i, name_i, path_i, time_i, n_input_array_elements
        self._estimations_table = df.infer_objects()

        self._stop_process_time()

        return self

    @override
    def make_report(
        self,
        *,
        group_by: str | Sequence[str] | None = None,
        aggregations: Sequence[str] | None = None,
        sort_by: str | None = None,
    ) -> DataFrame:
        assert (group_by, aggregations, sort_by) == (None, None, None) # FIXME

        estimations = self._estimations_table
        estimations_grouped = estimations.groupby(by="type", as_index=False)
        summary = estimations_grouped.aggregate(  # pyright: ignore [reportCallIssue]
            **{
                "n_nodes": ("type", "count"),
                "t_step": ("time", "sum"),
                "n_input_array_elements_average": ("n_input_array_elements", "mean"),
                "n_input_array_elements_total": ("n_input_array_elements", "sum"),
            }
        ).sort_values("t_step", ascending=False, ignore_index=True)
        summary["fraction_step_percent"] = summary["t_step"] / summary["t_step"].sum()
        summary["t_node_average"] = summary["t_step"] / summary["n_nodes"]

        return summary.infer_objects()

    def _print_total_time(self):
        total = self._total_estimations_time() * self._n_runs
        print(f"total estimations time / n_runs: {total / self._n_runs:.9f} sec.")
        print(f"total estimations time: {total:.6f} sec.")

    @override
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
