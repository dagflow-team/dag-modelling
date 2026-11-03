#!/usr/bin/env python
"""Benchmark the CSR layout against dense matrices, 240 to 4000 bins:
1. AxisDistortionMatrix and VectorMatrixProduct, the chain of a fit iteration
2. the same with AxisDistortionMatrixPointwise
3. static RebinMatrix with a wide last bin and VectorMatrixProduct
"""

# isort: off
from __future__ import annotations
from dag_modelling.tools import disable_implicit_numpy_multithreading  # noqa: F401

# isort: on

from numpy import arange, concatenate, linspace

from _bench_utils import banner, best_time
from dag_modelling.core.graph import Graph
from dag_modelling.lib.common import Array
from dag_modelling.lib.hist import (
    AxisDistortionMatrix,
    AxisDistortionMatrixPointwise,
    RebinMatrix,
)
from dag_modelling.lib.linalg import VectorMatrixProduct

SCALE = 1.03
NBINS = (240, 500, 1000, 2000, 4000)
GROUP = 4  # old bins per new bin, except the last one


def grouped_bins(nbins: int) -> int:
    """Old bins grouped by `GROUP`, the last third of the range is one new bin"""
    return 2 * nbins // 3 // GROUP * GROUP


def build(nbins: int, layout: str):
    edges = linspace(0.0, 10.0, nbins + 1)
    with Graph(close_on_exit=True):
        array_edges = Array("Edges", edges, mode="fill")
        array_modified = Array("Edges modified", edges * SCALE, mode="fill")
        array_backward = Array("Edges backward", edges / SCALE, mode="fill")
        array_vector = Array("Vector", arange(1.0, nbins + 1), mode="fill")

        matrix = AxisDistortionMatrix("matrix", layout=layout)
        array_edges >> matrix.inputs["EdgesOriginal"]
        array_edges >> matrix.inputs["EdgesTarget"]
        array_modified >> matrix.inputs["EdgesModified"]
        array_backward >> matrix.inputs["EdgesModifiedBackwards"]

        product = VectorMatrixProduct("product", mode="column")
        matrix.outputs["matrix"] >> product.inputs["matrix"]
        array_vector >> product
    product.get_data()
    return array_modified, matrix, product


def build_pointwise(nbins: int, layout: str):
    """The curve has 4 points per bin, as the LSNL curves of Daya Bay"""
    edges = linspace(0.0, 10.0, nbins + 1)
    x = linspace(0.0, 10.0, 4 * nbins + 1)
    with Graph(close_on_exit=True):
        array_edges = Array("Edges", edges, mode="fill")
        array_x = Array("Curve X", x, mode="fill")
        array_y = Array("Curve Y", x * SCALE, mode="fill")
        array_vector = Array("Vector", arange(1.0, nbins + 1), mode="fill")

        matrix = AxisDistortionMatrixPointwise("matrix", layout=layout)
        array_edges >> matrix.inputs["EdgesOriginal"]
        array_edges >> matrix.inputs["EdgesTarget"]
        array_x >> matrix.inputs["DistortionOriginal"]
        array_y >> matrix.inputs["DistortionTarget"]

        product = VectorMatrixProduct("product", mode="column")
        matrix.outputs["matrix"] >> product.inputs["matrix"]
        array_vector >> product
    product.get_data()
    return array_y, matrix, product


def build_rebin(nbins: int, layout: str):
    edges_old = linspace(0.0, 1.0, nbins + 1)
    stop = grouped_bins(nbins) + 1
    edges_new = concatenate((edges_old[:stop:GROUP], edges_old[-1:]))
    with Graph(close_on_exit=True):
        array_vector = Array("Vector", arange(1.0, nbins + 1), mode="fill")
        rebin = RebinMatrix("rebin", layout=layout)
        Array("Edges old", edges_old, mode="fill") >> rebin.inputs["edges_old"]
        Array("Edges new", edges_new, mode="fill") >> rebin.inputs["edges_new"]

        product = VectorMatrixProduct("product", mode="column")
        rebin.outputs["matrix"] >> product.inputs["matrix"]
        array_vector >> product
    product.get_data()
    return array_vector, product


def repetitions(nbins: int) -> int:
    return max(20, int(2e6 / nbins**1.5) + 20)


def iteration_time(source, product, number: int) -> float:
    def iteration():
        source.taint()
        product.touch()

    return best_time(iteration, number)


def measure(builder, nbins: int, layout: str, number: int) -> tuple[float, float, float, float]:
    source, matrix, product = builder(nbins, layout)
    t_iter = iteration_time(source, product, number)
    t_producer = best_time(matrix._evaluator.function, number)
    t_consumer = best_time(product._evaluator.function, number)
    mbytes = matrix.outputs["matrix"]._data.nbytes / 1e6
    return t_iter, t_producer, t_consumer, mbytes


def compare_chains(title: str, builder) -> None:
    banner(f"CSR vs DENSE: {title}")
    print(f"distortion: uniform scale x{SCALE}\n")
    header = (
        f"{'nbins':>7} | {'dense iter':>11}{'prod':>9}{'cons':>9}{'MB':>8} | "
        f"{'csr iter':>10}{'prod':>9}{'cons':>9}{'MB':>8} | "
        f"{'iter':>7}{'prod':>7}{'cons':>7}{'mem':>7}"
    )
    print(header)
    print("-" * len(header))
    for nbins in NBINS:
        number = repetitions(nbins)
        d_iter, d_prod, d_cons, d_mb = measure(builder, nbins, "dense", number)
        c_iter, c_prod, c_cons, c_mb = measure(builder, nbins, "sparse-csr", number)
        print(
            f"{nbins:>7} | {d_iter * 1e3:11.3f}{d_prod * 1e3:9.3f}{d_cons * 1e3:9.3f}{d_mb:8.2f} | "
            f"{c_iter * 1e3:10.3f}{c_prod * 1e3:9.3f}{c_cons * 1e3:9.3f}{c_mb:8.3f} | "
            f"{d_iter / c_iter:7.1f}{d_prod / c_prod:7.1f}{d_cons / c_cons:7.1f}{d_mb / c_mb:7.0f}"
        )
    print("\ntimes in ms, last four columns are dense/csr ratios")


def main() -> None:
    compare_chains("distortion matrix chain", build)
    compare_chains("pointwise distortion matrix chain", build_pointwise)

    banner("CSR vs DENSE: static rebinning onto wide bins")
    print("the last third of the range is one new bin\n")
    header = f"{'nbins':>7} | {'dense':>9} | {'csr':>9} | {'ratio':>7}"
    print(header)
    print("-" * len(header))
    for nbins in NBINS:
        number = repetitions(nbins)
        t_dense, t_csr = (
            iteration_time(*build_rebin(nbins, layout), number)
            for layout in ("dense", "sparse-csr")
        )
        print(f"{nbins:>7} | {t_dense * 1e3:9.4f} | {t_csr * 1e3:9.4f} | {t_dense / t_csr:7.1f}")
    print("\ntimes in ms, the last column is the dense/csr ratio")


if __name__ == "__main__":
    main()
