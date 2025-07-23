from numpy import allclose, arange, array, tril
from pytest import mark

from dag_modelling.core.graph import Graph
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.lib.common import Array
from dag_modelling.lib.statistics import CovmatrixFromCormatrix


@mark.parametrize("dtype", ("d", "f"))
def test_CovmatrixFromCormatrix_00(testname, debug_graph, dtype):
    inSigma = arange(1.0, 4.0, dtype=dtype)
    inC = array(
        [
            [1.0, 0.5, 0.0],
            [0.5, 1.0, 0.9],
            [0.0, 0.9, 1.0],
        ],
        dtype=dtype,
    )
    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        matrix = Array("matrix", inC, mode="fill")
        sigma = Array("sigma", inSigma, mode="fill")
        cov = CovmatrixFromCormatrix("covariance")

        sigma >> cov.inputs["sigma"]
        matrix >> cov

    inV = inC * inSigma[:, None] * inSigma[None, :]
    V = cov.get_data()

    assert allclose(inV, V, atol=0, rtol=0)
    assert allclose(tril(V), tril(V.T), atol=0, rtol=0)

    savegraph(graph, f"output/{testname}.png", show=["all"])
