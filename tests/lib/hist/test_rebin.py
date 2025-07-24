from matplotlib import pyplot as plt
from numpy import allclose, finfo, linspace, matmul
from numpy.typing import NDArray
from pytest import mark, raises

from dag_modelling.core.graph import Graph
from dag_modelling.lib.hist import Rebin, RebinMatrix
from dag_modelling.lib.common import Array
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.plot.plot import plot_array_1d_hist


def partial_sum(y_old: NDArray, stride: int) -> list:
    psum = []
    i = 0
    while i < y_old.size:
        psum.append(y_old[i : i + stride].sum())
        i += stride
    return psum


@mark.parametrize("dtype", ("d", "f"))
@mark.parametrize("start", (0, 1))
@mark.parametrize("stride", (2, 4))
@mark.parametrize("mode", ("python", "numba"))
@mark.parametrize("nclones", (0, 2))
def test_Rebin(
    testname: str, start: int, stride: int, dtype: str, mode: str, nclones: int
):
    n = 21
    edges_old = linspace(0.0, 2.0, n, dtype=dtype)
    edges_new = edges_old[start::stride]
    y_old_list = [
        linspace(3.0, 0.0, n - 1, dtype=dtype),
        linspace(2.0, 0.0, n - 1, dtype=dtype),
    ]

    atol = finfo(dtype).resolution * 10
    with Graph(close_on_exit=True) as graph:
        EdgesOld = Array("edges_old", edges_old, mode="fill")
        EdgesNew = Array("edges_new", edges_new, mode="fill")
        Y = Array("Y", y_old_list[0], mode="fill")
        Y2 = Array("Y2", y_old_list[1], mode="fill")
        metanode = Rebin(mode=mode, atol=atol)

        EdgesOld >> metanode.inputs["edges_old"]
        EdgesNew >> metanode.inputs["edges_new"]

        # for iclone in range(nclones):
        #     EdgesOld_i = Array(f"edges_old_{iclone+1}", edges_old, mode="fill")
        #     EdgesOld_i >> metanode

        #     for matrix in metanode._RebinMatrixList:
        #         EdgesOld_i >> matrix

        Y >> metanode()
        Y2 >> metanode()
        # metanode.print()
        # metanode._RebinMatrixList[0].print()

    mat = metanode.outputs["matrix"].data
    # NOTE: Asserts below are only for current edges_new! For other binning it may not coincide!
    if not start:
        assert (mat.sum(axis=0) == 1).all()
        assert mat.sum(axis=0).sum() == n - 1

    for i, y_old in enumerate(y_old_list):
        y_new = metanode.outputs[i].data
        y_res = matmul(mat, y_old)
        assert allclose(y_res, y_new, atol=atol, rtol=0)
        y_check = partial_sum(y_old[start:], stride)
        assert allclose(y_check[: len(y_new)], y_new, atol=atol, rtol=0)

    # plots
    plot_array_1d_hist(
        array=y_old_list[0],
        edges=edges_old,
        color="blue",
        label="old edges 1",
        linewidth=2,
    )
    plot_array_1d_hist(
        array=y_old_list[1],
        edges=edges_old,
        color="orange",
        label="old edges 2",
        linewidth=2,
    )
    plot_array_1d_hist(
        array=metanode.outputs[0].data,
        edges=edges_new,
        color="blue",
        linestyle="-.",
        label="new edges 1",
        linewidth=2,
    )
    plot_array_1d_hist(
        array=metanode.outputs[1].data,
        edges=edges_new,
        color="orange",
        linestyle="-.",
        label="new edges 2",
        linewidth=2,
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(fontsize="x-large")
    plt.savefig(f"output/{testname}-plot.png")
    plt.close()

    savegraph(graph, f"output/{testname}-graph.png")


@mark.parametrize(
    "edges_new",
    (
        linspace(-1.0, 2.0, 21),
        linspace(0.0, 2.1, 21),
        linspace(0.0, 2.0, 41),
        linspace(0.0, 2.0, 10),
    ),
)
@mark.parametrize("mode", ("python", "numba"))
def test_RebinMatrix_wrong_edges_new(edges_new, mode):
    edges_old = linspace(0.0, 2.0, 21)
    with Graph(close_on_exit=True):
        EdgesOld = Array("edges_old", edges_old, mode="fill")
        EdgesNew = Array("edges_new", edges_new, mode="fill")
        mat = RebinMatrix("Rebin Matrix", mode=mode)
        EdgesOld >> mat("edges_old")
        EdgesNew >> mat("edges_new")
    with raises(Exception):
        mat.get_data()


@mark.parametrize("mode", ("python", "numba"))
def test_RebinMatrix_wrong_edges_new(mode):
    edges_old = linspace(0.0, 2.0, 21)
    edges_new = edges_old[0::2]
    edges_clone = edges_old.copy()
    edges_clone[0] -= 1
    with Graph(close_on_exit=True):
        EdgesOld = Array("edges_old", edges_old, mode="fill")
        EdgesNew = Array("edges_new", edges_new, mode="fill")
        EdgesClone = Array("edges_clone", edges_clone, mode="fill")
        mat = RebinMatrix("Rebin Matrix", mode=mode)
        EdgesOld >> mat("edges_old")
        EdgesNew >> mat("edges_new")
        EdgesClone >> mat

    # mat.print()
    with raises(RuntimeError):
        mat.get_data()
