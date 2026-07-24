from matplotlib import pyplot as plt
from numpy import linspace, ma
from pytest import mark

from dag_modelling.core.graph import Graph
from dag_modelling.lib.common import Array
from dag_modelling.lib.hist.hist_smear_erf import HistSmearErf
from dag_modelling.plot.plot import add_colorbar


@mark.parametrize("shape,dtype", (((10, 10), "d"), ((10, 5), "f"), ((5, 10), "d"), ((5, 1), "f")))
def test_HistSmearErf(
    shape: tuple[int, int],
    dtype: str,
    test_name: str,
    output_path: str,
):
    nbins_in, nbins_out = shape
    edges_in =  linspace(0, nbins_in, nbins_in + 1, dtype=dtype)
    edges_out = linspace(0, nbins_out, nbins_out + 1, dtype=dtype)

    relsigma = 0.1 / (edges_in[1:] + edges_in[:-1])

    with Graph(close_on_exit=True) as graph:
        EdgesIn = Array("EdgesIn", edges_in, mode="fill")
        EdgesOut = Array("EdgesOut", edges_out, mode="fill")
        RelSigma = Array("Relative resolution", relsigma, mode="fill")

        mat = HistSmearErf("Smearing Erf")

        EdgesIn >> mat.inputs["Edges"]
        EdgesOut >> mat.inputs["EdgesOut"]
        RelSigma >> mat.inputs["RelSigma"]

    res = mat.get_data()

    plt.figure()
    ax = plt.subplot(111, xlabel="x", ylabel="y")
    ax.set_aspect("equal")
    ax.minorticks_on()
    ax.vlines(edges_in, edges_out[0], edges_out[-1], linestyle="dashed", color="gray", alpha=0.5)
    ax.hlines(edges_out, edges_in[0], edges_in[-2], linestyle="dashed", color="gray", alpha=0.5)

    res_m = ma.array(res, mask=(res == 0))
    cmbl = ax.matshow(res_m, vmin=0, vmax=1)  # , extent=[edges[0], edges[-1], edges[-1], edges[0]])
    add_colorbar(cmbl)

    plt.savefig(f"{output_path}/{test_name}-plot.pdf")

    # TODO: add tests
