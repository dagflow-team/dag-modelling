#!/usr/bin/env python

from matplotlib import pyplot as plt
from numpy import allclose, arange, digitize, fabs, finfo, geomspace, ndarray, zeros
from pytest import mark

from dag_modelling.core.graph import Graph
from dag_modelling.lib.physics import EnergyResolution
from dag_modelling.lib.common import Array
from dag_modelling.plot.graphviz import savegraph

parnames = ("a_nonuniform", "b_stat", "c_noise")


@mark.parametrize("input_binning", ["equal", "variable"])
@mark.parametrize(
    "Energy_set",
    [
        [[1.025], [3.025], [6.025], [9.025]],
        [[1.025, 5.025, 9.025]],
        [[6.025, 7.025, 8.025, 8.825]],
    ],
)
def test_HistSmearNormalMatrixBC_v01(input_binning, debug_graph, Energy_set, test_name, output_path: str):
    def singularities(values, Edges):
        indices = digitize(values, Edges) - 1
        phist = zeros(Edges.size - 1)
        phist[indices] = 1.0
        return phist

    if input_binning == "equal":
        binwidth_in = 0.05
        Edges_in = arange(0.0, 12.0001, binwidth_in)
    else:
        Edges_in = geomspace(1.0, 12.0, 200)
        binwidth_in = Edges_in[1:] - Edges_in[:-1]

    wvals = [0.016, 0.081, 0.026]

    def RelSigma(e):
        a, b, c = wvals
        return (a**2 + (b**2) / e + (c / e) ** 2) ** 0.5 if all(e != 0) else 0

    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        edges = Array("Edges", Edges_in, mode="fill")
        a, b, c = tuple(
            Array(name, [val], mark=name) for name, val in zip(parnames, wvals)
        )

        ereses = []
        for i, energies in enumerate(Energy_set):
            phist_in = singularities(energies, Edges_in)
            Array(f"Energy_{i}", phist_in, edges=[edges.outputs["array"]], mode="fill")
            eres = EnergyResolution()
            for name, inp in zip(parnames, (a, b, c)):
                inp >> eres.inputs[name]
            edges >> eres.inputs["Edges"]
            edges >> eres.inputs["EdgesOut"]
            ereses.append(eres)
    savegraph(graph, f"{output_path}/{test_name}.png")

    for i, eres in enumerate(ereses):
        centers_in = eres.outputs["Energy"].data
        rs_calc = eres.inputs["RelSigma"].data
        rs_template = RelSigma(centers_in)
        assert allclose(rs_template, rs_calc, atol=finfo("d").resolution, rtol=0)
        mat = eres.outputs["SmearMatrix"]
        mat_edges = mat.dd.axes_edges
        assert mat_edges[0] is mat_edges[1]
        assert mat_edges[0] is edges.outputs[0]
        check_smearing_projection(mat.data)

        plt.figure()
        plt.grid()
        plt.plot(centers_in, mat.data @ centers_in, "+")
        plt.xlabel("Energy")
        plt.savefig(f"{output_path}/{test_name}_{i}.png")
        plt.close()


def check_smearing_projection(mat: ndarray, *, check_assert: bool = True) -> None:
    threshold = 1.0e-8
    ones = mat.sum(axis=0)
    zeros = fabs(ones - 1.0)

    istart, iend = None, None
    for istart, val in enumerate(zeros):
        if val < threshold:
            break
    for iend, val in enumerate(reversed(zeros)):
        if val < threshold:
            iend = -iend
            break
    if iend == 0:
        iend = None

    zerossub = zeros[istart:iend]
    if check_assert:
        assert (zerossub < threshold).all()
    return ones
