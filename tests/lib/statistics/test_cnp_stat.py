from numpy import allclose, arange, finfo, sqrt
from pytest import mark

from dag_modelling.core.graph import Graph
from dag_modelling.lib.common import Array
from dag_modelling.lib.statistics import CNPStat
from dag_modelling.plot.graphviz import savegraph


def test_Chi2CNPStat_v01(debug_graph, test_name):
    n = 10
    start = 10
    offset = 1.0
    dataa = arange(start, start + n, dtype="d") + 1
    theorya = dataa + offset

    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        data = Array("data", dataa, mark="Data", mode="fill")
        theory = Array("theory", theorya, mark="Theory", mode="fill")

        cnp = CNPStat(name="cnp", label="CNP stat. uncertainty")

        (data, theory) >> cnp
        chi2 = Chi2(name="chi2", label="chi2")
        data >> chi2("data")
        theory >> chi2("theory")
        cnp.outputs[0] >> chi2("errors")
    res = chi2.outputs["result"].data[0]
    res_expected = ((1.0 / dataa + 2.0 / theorya) * (theorya - dataa) ** 2).sum() / 3.0
    assert allclose(res, res_expected, atol=finfo("d").resolution)

    savegraph(graph, f"output/{test_name}.png")


@mark.parametrize("mode", (None, "uncertainty", "variance"))
def test_Chi2CNPStat_v01(mode: str | None, debug_graph, test_name):
    n = 10
    start = 10
    offset = 1.0
    dataa = arange(start, start + n, dtype="d") + 1
    theorya = dataa + offset

    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        data = Array("data", dataa, mark="Data", mode="fill")
        theory = Array("theory", theorya, mark="Theory", mode="fill")

        if mode:
            cnp = CNPStat(name="cnp", mode=mode, label=f"CNP stat. {mode}")
        else:
            cnp = CNPStat(name="cnp", label=f"CNP stat. {mode}")
        (data, theory) >> cnp

    res = cnp.outputs["result"].data
    if mode == "uncertainty" or mode is None:
        res_expected = sqrt(3.0 / (1 / dataa + 2 / theorya))
    elif mode == "variance":
        res_expected = 3.0 / (1 / dataa + 2 / theorya)

    assert allclose(res, res_expected, atol=finfo("d").resolution)

    savegraph(graph, f"output/{test_name}.png")
