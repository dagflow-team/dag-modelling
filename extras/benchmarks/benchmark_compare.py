#!/usr/bin/env python

"""Compare Cython vs pure Python on Daya Bay model.

Runs benchmark N times for each variant, swapping .so files between runs.
Each run is a separate subprocess to avoid import caching.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev

# extras/benchmarks/benchmark_compare.py -> repo root is three levels up.
SO_DIR = Path(__file__).parent.parent.parent / "src" / "dag_modelling" / "core"
BACKUP_DIR = Path("/tmp/cython_backup_compare")
SO_PATTERN = "*.cpython-*.so"

N_REPEATS = 5

BENCH_SCRIPT = r"""
import json, time
from dayabay_data_official import get_path_data
from dgm_dayabay_dev.models import load_model
from dag_modelling.tools.graph_walker import get_subgraph_nodes_between
from dag_modelling.tools.profiling import FitSimulationProfiler, FrameworkProfiler

model = load_model("v2", path_data=get_path_data(), close=True, strict=True, process_labels=False)
storage = model.storage
params = [s._value_node for key, s in storage["parameters.variable"].walkitems()
          if "spectrum_uncertainty" not in ".".join(key)]
stat_node = storage["outputs.statistic.full.pull.chi2cnp"].node
stat_node.outputs[0].data

subgraph = list(get_subgraph_nodes_between(sources=params, sinks=[stat_node]))

fp = FitSimulationProfiler(mode="simultaneous", parameters=params, endpoints=[stat_node], n_runs=100)
fp.estimate_fit()
fr = fp.print_report()

fwp = FrameworkProfiler(target_nodes=subgraph, sources=params, sinks=[stat_node], n_runs=100)
fwp.estimate_framework_time()
fwr = fwp.print_report()

print("RESULT:" + json.dumps({"t_iter": fr["t_call"].iloc[0], "t_fw": fwr["t_single"].iloc[0]}))
"""


def so_files():
    return list(SO_DIR.glob(SO_PATTERN))


def disable_cython():
    BACKUP_DIR.mkdir(exist_ok=True)
    for f in so_files():
        shutil.move(str(f), str(BACKUP_DIR / f.name))


def enable_cython():
    for f in BACKUP_DIR.glob(SO_PATTERN):
        shutil.move(str(f), str(SO_DIR / f.name))


def run_once():
    result = subprocess.run(
        [sys.executable, "-c", BENCH_SCRIPT],
        capture_output=True, text=True, timeout=300,
    )
    for line in result.stdout.splitlines():
        if line.startswith("RESULT:"):
            return json.loads(line.removeprefix("RESULT:"))
    print("STDERR:", result.stderr[-500:] if result.stderr else "(empty)")
    raise RuntimeError("Benchmark subprocess failed")


def run_series(label, n):
    results = []
    for i in range(n):
        print(f"  {label} run {i+1}/{n}...", end=" ", flush=True)
        r = run_once()
        print(f"t_iter={r['t_iter']*1000:.2f}ms  t_fw={r['t_fw']*1000:.2f}ms")
        results.append(r)
    return results


def fmt(values_s, unit="ms"):
    values = [v * 1000 for v in values_s]
    m = mean(values)
    s = stdev(values) if len(values) > 1 else 0
    return f"{m:.3f} ± {s:.3f} {unit}"


if __name__ == "__main__":
    print(f"=== Cython vs Python comparison ({N_REPEATS} runs each) ===\n")

    print("WITH Cython:")
    assert so_files(), "No .so files found!"
    cython_results = run_series("cython", N_REPEATS)

    print("\nWITHOUT Cython:")
    disable_cython()
    try:
        python_results = run_series("python", N_REPEATS)
    finally:
        enable_cython()
        print("\n(.so files restored)")

    cy_iter = [r["t_iter"] for r in cython_results]
    cy_fw = [r["t_fw"] for r in cython_results]
    py_iter = [r["t_iter"] for r in python_results]
    py_fw = [r["t_fw"] for r in python_results]

    print(f"\n{'':30s} {'Cython':>20s}  {'Python':>20s}  {'Speedup':>10s}")
    print("-" * 85)
    print(f"{'Iteration time (ms)':30s} {fmt(cy_iter):>20s}  {fmt(py_iter):>20s}  {mean(py_iter)/mean(cy_iter):>9.2f}x")
    print(f"{'Framework time (ms)':30s} {fmt(cy_fw):>20s}  {fmt(py_fw):>20s}  {mean(py_fw)/mean(cy_fw):>9.2f}x")
