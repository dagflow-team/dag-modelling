from matplotlib import pyplot as plt
from numpy import allclose, array, digitize, linspace, ma, poly1d, polyfit
from numpy.typing import ArrayLike
from pytest import mark

from dag_modelling.core.graph import Graph
from dag_modelling.lib.hist import AxisDistortionMatrixPointwise
from dag_modelling.lib.common import Array
from dag_modelling.plot.plot import add_colorbar


@mark.parametrize("xoffset", (0, -0.5, +0.5, +5, +10))
@mark.parametrize("yoffset", (0, -0.5, +0.5, +5, +10))
@mark.parametrize("inverse", (False, True))
@mark.parametrize("nsegments", (10, 21, 4))
@mark.parametrize("dtype", ("d", "f"))
def test_AxisDistortionMatrixPointwise(
    dtype: str,
    nsegments: int,
    inverse: bool,
    xoffset: float | int,
    yoffset: float | int,
    testname: str,
):
    nbins = 10
    edges = linspace(0, nbins, nbins + 1, dtype=dtype)
    x_fine = linspace(edges[0], edges[-1], nsegments + 1, dtype=dtype)
    y_fine = x_fine + yoffset
    x_fine = x_fine + xoffset
    if inverse:
        y_fine[:] = y_fine[::-1]

    # print(x_fine)
    # print(y_fine)

    with Graph(close_on_exit=True) as graph:
        Edges = Array("Edges", edges, mode="fill")
        EdgesModified = Array("Edges modified", edges, mode="fill")

        Distortion = Array("Distortion", x_fine, mode="fill")
        DistortionModified = Array("Distortion modified", y_fine, mode="fill")

        mat = AxisDistortionMatrixPointwise("LSNL matrix (pointwise)")

        Edges >> mat.inputs["EdgesOriginal"]
        EdgesModified >> mat.inputs["EdgesTarget"]

        Distortion >> mat.inputs["DistortionOriginal"]
        DistortionModified >> mat.inputs["DistortionTarget"]

    res = mat.get_data()

    plt.figure()
    ax = plt.subplot(111, xlabel="x", ylabel="y", title=f"Δx={xoffset}, Δy={yoffset}")
    ax.set_aspect("equal")
    # ax.grid()
    ax.minorticks_on()
    ax.vlines(edges, edges[0], edges[-1], linestyle="dashed", color="gray", alpha=0.5)
    ax.hlines(edges, edges[0], edges[-1], linestyle="dashed", color="gray", alpha=0.5)

    res_m = ma.array(res, mask=(res == 0))
    cmbl = ax.matshow(
        res_m, vmin=0, vmax=1, extent=[edges[0], edges[-1], edges[-1], edges[0]]
    )
    add_colorbar(cmbl)

    ax.plot(x_fine, y_fine, "+-", color="magenta")

    if inverse:
        y_top = edges[-1] + xoffset + yoffset
        x_right = y_top

        x_left = xoffset + yoffset
        y_bottom = x_left
    else:
        y_bottom = yoffset - xoffset
        x_left = -y_bottom
        x_right = edges[-1] + x_left
        y_top = edges[-1] + y_bottom

    ax.axvline(x_left, color="magenta", linestyle="dotted", alpha=0.5)
    ax.axvline(x_right, color="magenta", linestyle="dashed", alpha=0.5)
    ax.axhline(y_bottom, color="magenta", linestyle="dotted", alpha=0.5)
    ax.axhline(y_top, color="magenta", linestyle="dashed", alpha=0.5)

    # print(f"{x_left=} {x_right=} {y_bottom=} {y_top=}")

    plt.savefig(f"output/{testname}-plot.pdf")

    ressum = res.sum(axis=0)
    # print("Obtained matrix sum:\n", ressum)

    # plt.show()

    # TODO: test on contents
    assert (0.0 <= res).all() and (res <= 1.0).all()
    assert (0.0 <= ressum).all() and (ressum <= 1.0).all()

    xfirst = max(x_left, edges[0], x_fine[0])
    xlast = min(x_right, edges[-1], x_fine[-1])
    # if inverse:
    #     yfirst = max(y_bottom, edges[0], y_fine[-1])
    #     ylast = min(y_top, edges[-1], y_fine[0])
    # else:
    #     yfirst = max(y_bottom, edges[0], y_fine[0])
    #     ylast = min(y_top, edges[-1], y_fine[-1])

    ixfirst = digitize(xfirst, edges, right=False) - 1
    ixlast = digitize(xlast, edges, right=False) - 1

    if xfirst > edges[0] and xfirst % 1 > 0:
        ixfirst += 1

    # print(f"Check {ixfirst}:{ixlast}, x {xfirst} {xlast} y {yfirst} {ylast}")
    if ixlast > ixfirst:
        check = ressum[ixfirst:ixlast]
        # print(f"Size: {check.size}")
        # print(check, (check == 1.0).all())

        if dtype == "d":
            assert (check == 1.0).all()
        else:
            assert allclose(check, 1.0, atol=1.0e-7)
    else:
        assert (res == 0.0).all()

    plt.close()


@mark.parametrize("xoffset", (0, -0.5, +0.5, +5, +10))
@mark.parametrize("yoffset", (0, -0.5, +0.5, +5, +10))
@mark.parametrize("inverse", (False, True))
@mark.parametrize(
    "poly_points",
    (
        [0, 5, 5, 5, 10],
        [0, 6, 5, 4, 10],
        [0, 8, 5, 2, 10],
        [0, 15, 5, -5, 10],
        [0, 0, 0, 0, 0],
    ),
)
@mark.parametrize("nsegments", (10, 21, 4))
@mark.parametrize("dtype", ("d", "f"))
def test_AxisDistortionMatrixPointwise_pol3(
    dtype: str,
    nsegments: int,
    inverse: bool,
    xoffset: float | int,
    yoffset: float | int,
    poly_points: ArrayLike,
    testname: str,
):
    nbins = 10
    edges = linspace(0, nbins, nbins + 1, dtype=dtype)

    poly_points = array(poly_points)
    poly_coeffs = polyfit([0, 3, 5, 7, 10], poly_points, deg=3)
    poly = poly1d(poly_coeffs)
    x_fine = linspace(edges[0], edges[-1], nsegments + 1, dtype=dtype)
    y_fine = (poly(x_fine) + yoffset).astype(dtype)
    x_fine += xoffset
    if inverse:
        y_fine[:] = y_fine[::-1]

    # print(x_fine)
    # print(y_fine)

    with Graph(close_on_exit=True) as graph:
        Edges = Array("Edges", edges, mode="fill")
        EdgesModified = Array("Edges modified", edges, mode="fill")

        Distortion = Array("Distortion", x_fine, mode="fill")
        DistortionModified = Array("Distortion modified", y_fine, mode="fill")

        mat = AxisDistortionMatrixPointwise("LSNL matrix (pointwise)")

        Edges >> mat.inputs["EdgesOriginal"]
        EdgesModified >> mat.inputs["EdgesTarget"]

        Distortion >> mat.inputs["DistortionOriginal"]
        DistortionModified >> mat.inputs["DistortionTarget"]

    res = mat.get_data()

    plt.figure()
    ax = plt.subplot(111, xlabel="x", ylabel="y", title="")
    ax.set_aspect("equal")
    # ax.grid()
    ax.minorticks_on()
    ax.vlines(edges, edges[0], edges[-1], linestyle="dashed", color="gray", alpha=0.5)
    ax.hlines(edges, edges[0], edges[-1], linestyle="dashed", color="gray", alpha=0.5)

    res_m = ma.array(res, mask=(res == 0))
    cmbl = ax.matshow(
        res_m, vmin=0, vmax=1, extent=[edges[0], edges[-1], edges[-1], edges[0]]
    )
    add_colorbar(cmbl)

    ax.plot(x_fine, y_fine, "+-", color="magenta")
    plt.savefig(f"output/{testname}-plot.pdf")

    ressum = res.sum(axis=0)
    # print("Obtained matrix sum:\n", ressum)

    # plt.show()

    assert (0.0 <= res).all()
    assert (0.0 <= ressum).all()
    if dtype == "d":
        assert (res <= 1.0).all()
        assert (ressum <= 1.0).all()
    else:
        assert (res <= 1.0 + 2e-7).all()
        assert (ressum <= 1.0 + 2e-7).all()

    plt.close()
