"""Benchmark Cython optimization on real Daya Bay model.

Three profiling modes:
1. FitSimulation (simultaneous) — full iteration time
2. Framework — pure framework time (no computations)
3. cProfile — detailed Python call statistics
"""

import cProfile
import time

from dayabay_data_official import get_path_data
from dgm_dayabay_dev.models import load_model

from _bench_utils import banner, print_cprofile_stats
from dag_modelling.tools.graph_walker import get_subgraph_nodes_between
from dag_modelling.tools.profiling import FitSimulationProfiler, FrameworkProfiler

# ── Load model ──────────────────────────────────────────────────────────────

print("Loading Daya Bay v2 model...")
t0 = time.time()
model = load_model("v2", path_data=get_path_data(), close=True, strict=True, process_labels=False)
storage = model.storage
print(f"Model loaded in {time.time() - t0:.1f}s")

# ── Extract parameters and endpoint ─────────────────────────────────────────

params = [
    s._value_node
    for key, s in storage["parameters.variable"].walkitems()
    if "spectrum_uncertainty" not in ".".join(key)
]

stat_node = storage["outputs.statistic.full.pull.chi2cnp"].node

print(f"Parameters: {len(params)}")
print(f"Endpoint: {stat_node}")

# ── Initial evaluation ──────────────────────────────────────────────────────

print("Running initial evaluation...")
result = stat_node.outputs[0].data
print(f"chi2cnp = {result}")

# ── Subgraph ────────────────────────────────────────────────────────────────

subgraph = list(get_subgraph_nodes_between(sources=params, sinks=[stat_node]))
print(f"Nodes in subgraph: {len(subgraph)}")

# ── 1. FitSimulation profiling (simultaneous) ──────────────────────────────

banner("1. FIT SIMULATION PROFILING (simultaneous)")

fit_profiler = FitSimulationProfiler(
    mode="simultaneous",
    parameters=params,
    endpoints=[stat_node],
    n_runs=100,
)
fit_profiler.estimate_fit()
fit_report = fit_profiler.print_report()

# ── 2. Framework profiling ──────────────────────────────────────────────────

banner("2. FRAMEWORK PROFILING (stub functions)")

fw_profiler = FrameworkProfiler(
    target_nodes=subgraph,
    sources=params,
    sinks=[stat_node],
    n_runs=100,
)
fw_profiler.estimate_framework_time()
fw_report = fw_profiler.print_report()

# ── 3. cProfile ─────────────────────────────────────────────────────────────

banner("3. cPROFILE (100 iterations)")


def one_iteration():
    for p in params:
        p.taint()
    stat_node.touch()


# Warmup
for _ in range(5):
    one_iteration()

# Profile
profiler = cProfile.Profile()
profiler.enable()
for _ in range(100):
    one_iteration()
profiler.disable()

stats = print_cprofile_stats(profiler, sort_by="cumulative", n=30)
total_calls = stats.total_calls

# ── Summary ─────────────────────────────────────────────────────────────────

banner("SUMMARY")

fit_mean = fit_report["t_call"].iloc[0]
fw_mean = fw_report["t_single"].iloc[0]
pct = fw_mean / fit_mean * 100

print(f"  Nodes in subgraph:              {len(subgraph)}")
print(f"  Parameters:                     {len(params)}")
print(f"  Iteration time (FitSimulation): {fit_mean * 1000:.3f} ms")
print(f"  Framework time (Framework):     {fw_mean * 1000:.3f} ms")
print(f"  Framework fraction:             {pct:.1f}%")
print(f"  Python function calls (100 it): {total_calls}")
print(f"  Python function calls (1 iter): {total_calls // 100}")
