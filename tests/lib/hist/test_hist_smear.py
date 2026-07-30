from typing import Literal

from matplotlib import pyplot as plt
from numpy import allclose, fabs, finfo, full_like, linspace, logical_and, ma, where
from numpy.typing import NDArray
from pytest import mark

from dag_modelling.core.graph import Graph
from dag_modelling.lib.common import Array
from dag_modelling.lib.hist import HistSmearNormalMatrixBC, HistSmearNormalMatrixIntegral
from dag_modelling.plot.plot import add_colorbar


@mark.parametrize(
    "dtype",
    (
        "d",
        # "f"
    ),
)
@mark.parametrize("nbins", (100,))
@mark.parametrize("mode", ("integral", "bincenter"))
@mark.parametrize(
    "sigma_type,sigma",
    (
        ("relative", 0.1),
        ("absolute", 0.1),
        ("relative", 0.5),
        ("absolute", 0.5),
    ),
)
def test_HistSmearErf(
    nbins: int,
    mode: Literal["integral", "bincenter"],
    sigma_type: Literal["relative", "absolute"],
    sigma: float,
    dtype: str,
    test_name: str,
    output_path: str,
):
    edges = linspace(0, 10, nbins + 1, dtype=dtype)

    centers = 0.5 * (edges[1:] + edges[:-1])
    match sigma_type:
        case "relative":
            sigma_rel = full_like(centers, sigma)
            sigma_abs = centers*sigma_rel
        case "absolute":
            sigma_abs = full_like(centers, sigma)
            sigma_rel = sigma / centers

        case _:
            raise RuntimeError()

    with Graph(close_on_exit=True) as graph:
        EdgesIn = Array("EdgesIn", edges, mode="fill")
        EdgesOut = Array("EdgesOut", edges, mode="fill")
        RelSigma = Array("Relative resolution", sigma_rel, mode="fill")

        if mode == "integral":
            mat = HistSmearNormalMatrixIntegral(f"Smearing: {mode}")
        elif mode == "bincenter":
            mat = HistSmearNormalMatrixBC(f"Smearing: {mode}")

        EdgesIn >> mat.inputs["Edges"]
        EdgesOut >> mat.inputs["EdgesOut"]
        RelSigma >> mat.inputs["RelSigma"]

    res = mat.get_data()
    res_sum = res.sum(axis=0)
    res_sum1 = res.sum(axis=0)
    threshold = finfo(dtype).resolution

    res_m = ma.array(res, mask=(fabs(res) < threshold))

    plt.figure()
    ax = plt.subplot(
        111,
        xlabel="true",
        ylabel="smeared",
        title=f"Energy resolution: {mode}, σ={sigma} ({sigma_type})",
    )
    ax.set_aspect("equal")
    ax.minorticks_on()
    # ax.vlines(edges, edges[0], edges[-2], linestyle="solid", color="gray", alpha=0.4)
    # ax.hlines(edges, edges[0], edges[-2], linestyle="solid", color="gray", alpha=0.4)

    cmbl = ax.matshow(res_m, vmin=0, vmax=1, extent=[edges[0], edges[-1], edges[-1], edges[0]])
    add_colorbar(cmbl)

    ax.plot([edges[0], edges[-1]], [edges[0], edges[-1]], color="red", alpha=0.5)

    plt.savefig(f"{output_path}/{test_name}-{mode}-{dtype}-{nbins}-{sigma_type}-{sigma!s}-plot.pdf")

    plt.figure()
    ax = plt.subplot(
        111,
        xlabel="true",
        ylabel="smeared",
        title=f"Energy resolution: {mode}, σ={sigma} ({sigma_type})",
    )
    ax.set_aspect("equal")
    ax.minorticks_on()
    # ax.vlines(edges, edges[0], edges[-2], linestyle="solid", color="gray", alpha=0.4)
    # ax.hlines(edges, edges[0], edges[-2], linestyle="solid", color="gray", alpha=0.4)

    cmbl = ax.matshow(
        res_m, vmin=1.0e-4, vmax=1, norm="log", extent=[edges[0], edges[-1], edges[-1], edges[0]]
    )
    add_colorbar(cmbl)

    ax.plot([edges[0], edges[-1]], [edges[0], edges[-1]], color="red", alpha=0.5)

    plt.savefig(f"{output_path}/{test_name}-{mode}-{dtype}-{nbins}-{sigma_type}-{sigma!s}-log.pdf")

    plt.figure()
    ax = plt.subplot(
        111,
        xlabel="true",
        ylabel="smeared",
        title=f"Energy resolution: {mode}, σ={sigma} ({sigma_type})",
    )
    ax.minorticks_on()
    ax.grid()

    for icol in (1, 10, 20, 40, 60, 80, 99):
        slc = res[:, icol]
        ax.stairs(slc, edges, label=f"[{edges[icol]:.2f}, {edges[icol+1]:.2f}], σ={sigma_abs[icol]:.2f}")
    ax.legend()

    plt.savefig(
        f"{output_path}/{test_name}-{mode}-{dtype}-{nbins}-{sigma_type}-{sigma!s}-slices.pdf"
    )

    mask_no_underflow = fabs(res[0, :]) < threshold
    mask_no_overflow = fabs(res[-1, :]) < threshold
    mask_no_spill = logical_and(mask_no_overflow, mask_no_underflow)
    print(f"masked sum {mode} {sigma_type} {sigma}", res_sum[mask_no_spill])
    print("sum rows", res_sum)
    print("sum cols", res_sum1)

    plt.show()

    # assert is_continuous(mask_no_spill)
    # assert all(res_sum < 1.0 + threshold)
    # assert allclose(res_sum[mask_no_spill], 1.0, atol=threshold)


def is_continuous(mask: NDArray) -> bool:
    idx = where(mask)[0]
    return len(idx) == 0 or (idx[-1] - idx[0] + 1 == len(idx))
