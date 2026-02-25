from __future__ import annotations

from typing import TYPE_CHECKING

from pandas import DataFrame

if TYPE_CHECKING:
    from collections.abc import Sequence, Callable
    from ...core.node import Node
    from ...core.input import Input
    from ...core.output import Output

from .profiler import Profiler

# prefix `s_` - size notation
# columnt aliases for aggregate functions
_COLUMN_ALIASES: dict[str | Callable, str] = {
    "mean": "size_single",
    "single": "size_single",
    "size_mean": "size_single",
    "median": "size_median",
    "sum": "size_sum",
    "std": "size_std",
    "count": "node_count",
    "min": "size_min",
    "max": "size_max",
    "var": "size_var",
}
_AGGREGATE_ALIASES: dict[str, str | Callable] = {
    "single": "mean",
    "size_single": "mean",
    "size_mean": "mean",
    "size_median": "median",
    "size_sum": "sum",
    "size_std": "std",
    "node_count": "count",
    "size_min": "min",
    "size_max": "max",
    "size_var": "var",
}

_DEFAULT_AGGREGATIONS = ("count", "sum")


class MemoryProfiler(Profiler):
    """Memory Profiler.

    It helps to estimate the size of edges of the given nodes. All
    estimations are performed in bytes and only those edges that own
    memory (size != 0) are taken into account.
    """

    __slots__ = ()

    def __init__(
        self,
        target_nodes: Sequence[Node] = (),
        *,
        sources: Sequence[Node] = (),
        sinks: Sequence[Node] = (),
    ):
        self._default_aggregations = _DEFAULT_AGGREGATIONS
        self._column_aliases = _COLUMN_ALIASES.copy()
        self._aggregate_aliases = _AGGREGATE_ALIASES.copy()
        self._primary_col = "n_bytes_total"
        super().__init__(target_nodes, sources, sinks)

    def _touch_nodes(self):
        for node in self._target_nodes:
            node.eval()

    @classmethod
    def estimate_node(
        cls, node
    ) -> tuple[dict[Input, tuple[int, int]], dict[Output, tuple[int, int]]]:
        """Return `dict` of sizes for each Input/Output of given `node`"""
        inputs = {}
        outputs = {}
        inp: Input
        out: Output
        for inp in node.inputs.iter_all():
            if inp.has_data and inp.owns_buffer:
                inputs[inp] = inp._own_data.nbytes, inp._own_data.size
            else:
                inputs[inp] = 0, 0
        for out in node.outputs.iter_all():
            # If there is an _allocating_input,
            #  the `out.data` refers to the child `Input` data.
            # However if there is no `_allocating_input`
            #  and owns_buffer=False (and `out.data` is not `None` of course)
            #  then it means there is allocated memory for this Output.
            if out.has_data and (out.owns_buffer or out._allocating_input is None):
                outputs[out] = out._data.nbytes, out._data.size
            else:
                outputs[out] = 0, 0
        return inputs, outputs

    def estimate_target_nodes(self, touch=False):
        """Estimate size of edges of all `self.target_nodes`. Only those edges
        that own memory (size != 0) are taken into account.

        Return current `MemoryProfiler` instance.
        """
        self._start_perf_counter()
        if touch:
            self._touch_nodes()

        records = {col: [] for col in (
            "node",
            "type",
            "n_inputs",
            "n_outputs",
            "n_elements",
            "n_bytes",
        )}

        for node in self._target_nodes:
            inputs, outputs = self.estimate_node(node)
            # 0 size is not counted as an edge of the node
            n_bytes_inputs = tuple(size[0] for _, size in inputs.items() if size[0])
            n_bytes_outputs = tuple(size[0] for _, size in outputs.items() if size[0])
            n_bytes = sum(n_bytes_inputs) + sum(n_bytes_outputs)

            size_inputs = tuple(size[1] for _, size in inputs.items() if size[1])
            size_outputs = tuple(size[1] for _, size in outputs.items() if size[1])
            size = sum(size_inputs) + sum(size_outputs)

            records["node"].append(str(node))
            records["type"].append(type(node).__name__)
            records["n_inputs"].append(len(inputs))
            records["n_outputs"].append(len(inputs))
            records["n_elements"].append(size)
            records["n_bytes"].append(n_bytes)

        self._estimations_table = DataFrame(records)
        self._stop_perf_counter()
        return self

    @property
    def total_size(self):
        """Return size of all edges of '_target_nodes' in bytes."""
        if not hasattr(self, "_estimations_table"):
            return None
        return self._estimations_table["n_bytes"].sum()

    def make_report(
        self,
        *,
        group_by: str | Sequence[str] | None = None,
        aggregations: Sequence[str] | None = None,
        sort_by: str | None = None,
    ) -> DataFrame:
        assert (group_by, aggregations, sort_by) == (None, None, None)

        estimations = self._estimations_table
        estimations_grouped = estimations.groupby(by="type", as_index=False)
        summary = estimations_grouped.aggregate(  # pyright: ignore [reportCallIssue]
            **{
                "n_nodes": ("type", "count"),
                "n_inputs_total": ("n_inputs", "sum"),
                "n_inputs_average": ("n_inputs", "mean"),
                "n_outputs_total": ("n_outputs", "sum"),
                "n_outputs_average": ("n_outputs", "mean"),
                "n_elements_total": ("n_elements", "sum"),
                "n_elements_average": ("n_elements", "mean"),
                "n_bytes_total": ("n_bytes", "sum"),
                "n_bytes_average": ("n_bytes", "mean"),
            }
        ).sort_values("n_bytes_total", ascending=False, ignore_index=True)

        return summary

    def _present_in_units(self, value, float_bytes=False, separator="\n\t") -> str:
        """Convert the `value` in bytes to kilobytes, and megabytes.

        Return formatted string, where the values separated by `separator`:
        """
        return separator.join(
            (
                f"{value:.{1 if float_bytes else 0}f} bytes",
                f"{value / 2**10:.1f} KB",
                f"{value / 2**20:.1f} MB",
            )
        )

    def print_report(
        self,
        *,
        rows: int | None = 100,
        group_by: str | Sequence[str] | None = None,
        aggregations: Sequence[str] | None = None,
        sort_by: str | None = None,
    ) -> DataFrame:
        assert (group_by, aggregations, sort_by) == (None, None, None)
        report = self.make_report()
        print(
            f"\nMemory Profiling {hex(id(self))}, "
            f"nodes in subgraph: {len(self._target_nodes)}\n"
            f"sort by: `{sort_by or 'default sorting'}`, "
            f"group by: `{group_by or 'no grouping'}`"
        )
        # Limit float formatting to one decimal place,
        #  since we are working with bytes and higher precision is unnecessary
        self._print_table(report, rows, float_fmt=".1f")
        size_bytes = self.total_size
        print("TOTAL SIZE:")
        print(f"\t{self._present_in_units(size_bytes)}")
        print("TOTAL SIZE / node count:")
        if size_bytes is not None:
            s_node = size_bytes / len(self._target_nodes)
            print(f"\t{self._present_in_units(s_node, float_bytes=True)}")
        else:
            print(f"\t[undefined]")
        return report
