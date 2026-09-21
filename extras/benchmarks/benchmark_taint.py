#!/usr/bin/env python3
"""Benchmark for taint/eval cascade in dag-modelling.

Measures the real hot path: set() → taint cascade → touch() → eval cascade.
"""
from __future__ import annotations

import time

from numpy import arange, zeros

from _bench_utils import banner, print_cprofile_stats
from dag_modelling.core.graph import Graph
from dag_modelling.lib.arithmetic import Product, Sum
from dag_modelling.lib.common import Array


def bench(name: str, setup, body, n_warmup: int = 1000, n_iter: int = 50_000):
    """setup() builds the fixture state (ctx) once; body(ctx) is timed n_iter
    times against that same state (after n_warmup untimed calls)."""
    ctx = setup()
    for _ in range(n_warmup):
        body(ctx)

    t0 = time.perf_counter()
    for _ in range(n_iter):
        body(ctx)
    dt = time.perf_counter() - t0

    us = dt / n_iter * 1e6
    print(f"  {name:40s}  {us:8.2f} µs/iter  ({n_iter} iters, {dt:.3f}s total)")
    return us


# ---------------------------------------------------------------------------
# Scenario 1: Linear chain  (Array → Sum → Sum → ... → Sum)
# ---------------------------------------------------------------------------
def setup_chain(n: int = 100):
    def _setup():
        array = arange(3, dtype="d")
        with Graph(close_on_exit=True) as graph:
            src = Array("src", array)
            prev = src
            for i in range(n):
                s = Sum(f"s_{i}")
                prev >> s
                prev = s
            last = prev
        # Initial eval (untaint graph)
        last.outputs["result"].touch()
        return {
            "src": src,
            "last": last,
            "data": src.outputs["array"].data.copy(),
        }
    return _setup


def body_set_touch(ctx):
    """Full cycle: set → taint → touch → eval."""
    ctx["src"].outputs["array"].set(ctx["data"])
    ctx["last"].outputs["result"].touch()


# ---------------------------------------------------------------------------
# Scenario 2: Wide DAG  (fan-out + intermediate + fan-in)
#
# Topology (inspired by Daya Bay):
#   N_PARAMS parameter Arrays
#     ↓ (each feeds into BRANCH_WIDTH Sum nodes)
#   N_PARAMS × BRANCH_WIDTH intermediate Sums
#     ↓ (groups of BRANCH_WIDTH merge into one Sum each)
#   N_PARAMS merge Sums
#     ↓ (all merge into final Product)
#   1 final Product
#
# Total nodes: N_PARAMS + N_PARAMS*BRANCH_WIDTH + N_PARAMS + 1
# ---------------------------------------------------------------------------
def setup_wide_dag(n_params: int = 6, branch_width: int = 8):
    def _setup():
        array = arange(3, dtype="d")
        with Graph(close_on_exit=True) as graph:
            sources = [Array(f"par_{i}", array) for i in range(n_params)]

            intermediates = []
            for i, src in enumerate(sources):
                layer = []
                for j in range(branch_width):
                    const = Array(f"const_{i}_{j}", array)
                    s = Sum(f"mid_{i}_{j}")
                    (src, const) >> s
                    layer.append(s)
                intermediates.append(layer)

            merges = []
            for i, layer in enumerate(intermediates):
                m = Sum(f"merge_{i}")
                for node in layer:
                    node >> m
                merges.append(m)

            final = Product("final")
            for m in merges:
                m >> final

        final.outputs["result"].touch()

        return {
            "sources": sources,
            "final": final,
            "data": array.copy(),
        }
    return _setup


def body_wide_set_touch(ctx):
    """Change one param → taint → eval."""
    ctx["sources"][0].outputs["array"].set(ctx["data"])
    ctx["final"].outputs["result"].touch()


def body_wide_set_all_touch(ctx):
    """Change all params → taint → eval."""
    for src in ctx["sources"]:
        src.outputs["array"].set(ctx["data"])
    ctx["final"].outputs["result"].touch()


# ---------------------------------------------------------------------------
# Scenario 3: Profiling — measure taint-only vs touch-only vs numpy-only
# ---------------------------------------------------------------------------
def setup_chain_profiling(n: int = 100):
    """Setup for profiling: separate taint and eval phases."""
    def _setup():
        array = arange(3, dtype="d")
        with Graph(close_on_exit=True) as graph:
            src = Array("src", array)
            prev = src
            for i in range(n):
                s = Sum(f"s_{i}")
                prev >> s
                prev = s
            last = prev
        last.outputs["result"].touch()
        return {
            "src": src,
            "last": last,
            "data": src.outputs["array"].data.copy(),
        }
    return _setup


def body_touch_only(ctx):
    """Eval only: touch on tainted graph."""
    ctx["last"].outputs["result"].touch()


# ---------------------------------------------------------------------------
# Scenario 4: Pure numpy baseline (no DAG overhead)
# ---------------------------------------------------------------------------
def setup_numpy_baseline(n: int = 100):
    """Numpy baseline: n additions (no DAG)."""
    def _setup():
        arrays = [arange(3, dtype="d") for _ in range(n + 1)]
        return {"arrays": arrays, "result": zeros(3, dtype="d")}
    return _setup


def body_numpy_baseline(ctx):
    """Copy + n additions (no DAG)."""
    arrays = ctx["arrays"]
    result = ctx["result"]
    result[:] = arrays[0]
    for a in arrays[1:]:
        result += a


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    banner("DAG Taint/Eval Cascade Benchmark", width=75)
    print()
    print("Each iteration = set(data) + touch(result)  [= full fitting step]")
    print()

    for n in (20, 50, 100):
        print(f"--- Numpy baseline: {n} additions ---")
        bench(
            f"numpy({n}) copy+add",
            setup_numpy_baseline(n),
            body_numpy_baseline,
        )
    print()

    for n in (20, 50, 100):
        print(f"--- Linear chain: {n} nodes ---")
        bench(
            f"chain({n}) set+touch",
            setup_chain(n),
            body_set_touch,
        )
    print()

    for n_params, bw in [(6, 8), (6, 16), (10, 8)]:
        n_total = n_params + n_params * bw + n_params * bw + n_params + 1
        print(f"--- Wide DAG: {n_params} params × {bw} branches (~{n_total} nodes) ---")
        bench(
            f"wide({n_params}×{bw}) set_1+touch",
            setup_wide_dag(n_params, bw),
            body_wide_set_touch,
        )
        bench(
            f"wide({n_params}×{bw}) set_all+touch",
            setup_wide_dag(n_params, bw),
            body_wide_set_all_touch,
        )
    print()

    banner("PROFILING: taint vs eval breakdown (chain 100)", width=75)
    print()

    # Full set+touch for reference
    t_full = bench(
        "chain(100) set+touch (reference)",
        setup_chain_profiling(100),
        body_set_touch,
    )

    # Touch-only on already-untainted graph (should be near-zero: early return)
    t_noop = bench(
        "chain(100) touch (untainted=noop)",
        setup_chain_profiling(100),
        body_touch_only,
        n_iter=100_000,
    )

    # Interleave set/touch to measure taint vs eval
    def setup_split():
        base = setup_chain_profiling(100)()
        base["t_set"] = 0.0
        base["t_touch"] = 0.0
        return base

    n_split = 50_000
    ctx_split = setup_split()
    # Warmup
    for _ in range(1000):
        ctx_split["src"].outputs["array"].set(ctx_split["data"])
        ctx_split["last"].outputs["result"].touch()

    # NOTE: perf_counter() overhead (~1 µs) included
    t_set_total = 0.0
    t_touch_total = 0.0
    for _ in range(n_split):
        t0 = time.perf_counter()
        ctx_split["src"].outputs["array"].set(ctx_split["data"])
        t1 = time.perf_counter()
        ctx_split["last"].outputs["result"].touch()
        t2 = time.perf_counter()
        t_set_total += t1 - t0
        t_touch_total += t2 - t1

    us_set = t_set_total / n_split * 1e6
    us_touch = t_touch_total / n_split * 1e6
    print(f"  {'chain(100) set() only [taint cascade]':40s}  {us_set:8.2f} µs/iter")
    print(f"  {'chain(100) touch() only [eval cascade]':40s}  {us_touch:8.2f} µs/iter")
    print(f"  {'sum':40s}  {us_set + us_touch:8.2f} µs/iter")
    print(f"  {'taint / total':40s}  {us_set / (us_set + us_touch) * 100:8.1f} %")
    print(f"  {'eval  / total':40s}  {us_touch / (us_set + us_touch) * 100:8.1f} %")

    # NOTE: cProfile inflates timings; use for relative comparison only
    banner(
        "cProfile: chain(100) set+touch × 10000 iters  (absolute times inflated)",
        width=75,
    )
    import cProfile

    ctx_prof = setup_chain_profiling(100)()
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(10_000):
        ctx_prof["src"].outputs["array"].set(ctx_prof["data"])
        ctx_prof["last"].outputs["result"].touch()
    pr.disable()

    print_cprofile_stats(pr, sort_by="tottime", n=20)


if __name__ == "__main__":
    main()
