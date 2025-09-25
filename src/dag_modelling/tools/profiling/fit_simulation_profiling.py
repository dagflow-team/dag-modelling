from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from numpy import sum as npsum
from pandas import DataFrame, Series

from ...core.node import Node
from .timer_profiler import TimerProfiler

_ALLOWED_GROUPBY = (("parameters", "endpoints", "eval mode"),)


class FitSimulationProfiler(TimerProfiler):
    """Profiler class for estimating the time of model fitting. The fitting is
    simulated by touching different nodes (depending on the `mode`).

    NOTE: This class inherits from `TimerProfiler` and uses `source_nodes`
    as tweakable parameters to imitate model fit process.
    """

    __slots__ = ("_fit_step", "_mode", "_n_derivative_points")

    def __init__(
        self,
        mode: Literal["parameter-wise", "simultaneous"] = "parameter-wise",
        *,
        parameters: Sequence[Node] = (),
        endpoints: Sequence[Node] = (),
        n_runs: int = 10_000,
        n_derivative_points: int = 4,
    ):
        """Initializes the FitSimulationProfiler, which simulates model fit.

        There are two different `mode`:
           - `"parameter-wise"` - calculate n points for the parameter
           and then make a fit step for all the parameters.
           - `"simultaneous"` - fits all parameters together,
           which is similar to backpropogation in real tasks.

        The `parameters` and `endpoints` must each contain at least one node.

        The `n_derivative_points` specifies the number of points for derivative estimation
        for `mode="parameter-wise"`. Defaults to `4`.
        When the user sets `mode="simultaneous"` it is assumed that `n_derivative_points=0`
        """
        if not parameters or not endpoints:
            raise ValueError("There must be at least one parameter and at least one endpoint")
        super().__init__(sources=parameters, sinks=endpoints, n_runs=n_runs)
        if mode == "parameter-wise":
            self._fit_step = self._separate_step
            # TODO: add tests
            if n_derivative_points < 2:
                raise ValueError("Number of derivative points cannot be less than 2")
            self._n_derivative_points = n_derivative_points
            self._default_aggregations = ("step", "call", "total", "n_steps", "n_calls")
        elif mode == "simultaneous":
            self._fit_step = self._together_step
            self._n_derivative_points = 0
            self._default_aggregations = ("call", "total", "n_calls")
        else:
            raise ValueError(f"Unknown mode: {mode}")
        self._mode = mode
        self._allowed_groupby = _ALLOWED_GROUPBY

        # rename 't_single' (which is just a mean) to 't_step' for this profiling
        # TODO: perhaps not the most beautiful way to accomplish this
        for col_name in ("mean", "t_mean", "single"):
            self._column_aliases[col_name] = "t_step"
        for alias in ("single", "t_single"):
            self._aggregate_aliases[alias.replace("single", "step")] = "mean"

        # rename 'count' to 'n_steps'
        for col_name in ("count", "t_count"):
            self._column_aliases[col_name] = "n_steps"
        for alias in ("steps", "n_steps"):
            self._aggregate_aliases[alias] = "count"

        self.register_aggregate_func(
            func=self._n_calls, aliases=("n_calls",), column_name="n_calls"
        )
        self.register_aggregate_func(
            func=self._t_call,
            aliases=("call", "t_call"),
            column_name="t_call",
        )

    @property
    def mode(self):
        return self._mode

    @property
    def _parameters(self):
        """Alias for `self._sources`"""
        return self._sources

    @property
    def _endpoints(self):
        """Alias for `self._sinks`"""
        return self._sinks

    def _together_step(self):
        for parameter in self._sources:
            parameter.taint()
        self.__call_endpoints()

    def _separate_step(self):
        for parameter in self._sources:
            # simulate finding derivative by N points
            for _ in range(self._n_derivative_points):
                parameter.taint()
                self.__call_endpoints()
            # simulate reverting to the initial state
            parameter.taint()

        # make a step for all params
        for parameter in self._sources:
            parameter.taint()
        self.__call_endpoints()

    def __call_endpoints(self):
        for endpoint in self._sinks:
            endpoint.touch()

    def _touch_model_nodes(self):
        for node in self._target_nodes:
            node.touch()

    def estimate_fit(self) -> FitSimulationProfiler:
        self._touch_model_nodes()
        results = self._timeit_each_run(self._fit_step, n_runs=self.n_runs)
        source_short_names, sink_short_names = self._shorten_sources_sinks()
        self._estimations_table = DataFrame(
            {
                "parameters": source_short_names,
                "endpoints": sink_short_names,
                "eval mode": self._mode,
                "time": results,
            }
        )
        return self

    def _n_calls(self, _s: Series) -> Series:
        """User-defined aggregate function.

        Return number of calls for each "point" in derivative estimation
        for given group
        """
        # TODO: add tests
        return Series({"n_calls": (self._n_derivative_points + 1) * len(_s.index)})

    def _t_call(self, _s: Series) -> Series:
        """User-defined aggregate function.

        Return [total time] divided by [number of calls for each point
        in derivative computation + 1].
        """
        if len(_s.index) == 0:
            raise ZeroDivisionError("An empty group is received for t_call computation!")
        return Series({"t_call": npsum(_s) / ((self._n_derivative_points + 1) * len(_s.index))})

    def make_report(
        self,
        *,
        group_by: str | Sequence[str] | None = ("parameters", "endpoints", "eval mode"),
        aggregations: Sequence[str] | None = None,
        sort_by: str | None = None,
    ) -> DataFrame:
        return super().make_report(group_by=group_by, aggregations=aggregations, sort_by=sort_by)

    def print_report(
        self,
        *,
        rows: int | None = 100,
        group_by: str | Sequence[str] | None = ("parameters", "endpoints", "eval mode"),
        aggregations: Sequence[str] | None = None,
        sort_by: str | None = None,
    ) -> DataFrame:
        report = self.make_report(group_by=group_by, aggregations=aggregations, sort_by=sort_by)
        print(
            f"\nFit simulation Profiling {hex(id(self))}, "
            f"fit steps (n_runs): {self._n_runs},\n"
            f"nodes in subgraph: {len(self._target_nodes)}, "
            f"parameters: {len(self._sources)}, endpoints: {len(self._sinks)},\n"
            f"eval mode: {self.mode}, "
            f"{f'derivative points: {self._n_derivative_points}' if self._mode == 'parameter-wise' else ''}\n"
            f"sort by: `{sort_by or 'default sorting'}`, "
            f"group by: `{group_by or 'no grouping'}`"
        )
        super()._print_table(report, rows)
        return report
