from pytest import mark
from numpy import allclose, square

from dag_modelling.core.exception import CriticalError
from dag_modelling.core.graph import Graph
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.lib.common import Array
from dag_modelling.parameters import GaussianParameters, Parameter


@mark.parametrize("mode", ("single", "uncorr", "cov", "cov1d"))
def test_parameters_00(mode) -> None:
    names = list("abc")
    value_in = [1.1, 1.8, 5.0]
    central_in = [1.0, 2.0, 3.0]
    sigma_in = [1.0, 0.5, 2.0]
    corrs_in = [-0.1, 0.5, -0.9]  # 01, 02, 12
    variance_in = square(sigma_in)
    zeros_in = [0.0, 0.0, 0.0]

    if mode == "single":
        value_in = value_in[:1]
        central_in = central_in[:1]
        sigma_in = sigma_in[:1]
        zeros_in = zeros_in[:1]

    with Graph(debug=False, close_on_exit=False) as graph:
        value = Array("variable", value_in, mode="store_weak", mark="v")
        central = Array("central", central_in, mark="v₀")

        if mode in ("single", "uncorr", "cor"):
            sigma = Array("sigma", sigma_in, mark="σ")

        if mode in ("single", "uncorr"):
            gp = GaussianParameters(names, value, central, sigma=sigma)
        elif mode == "cov":
            covariance = Array(
                "covariance",
                [
                    [
                        variance_in[0],
                        corrs_in[0] * sigma_in[0] * sigma_in[1],
                        corrs_in[1] * sigma_in[0] * sigma_in[2],
                    ],
                    [
                        corrs_in[0] * sigma_in[0] * sigma_in[1],
                        variance_in[1],
                        corrs_in[2] * sigma_in[1] * sigma_in[2],
                    ],
                    [
                        corrs_in[1] * sigma_in[0] * sigma_in[2],
                        corrs_in[2] * sigma_in[1] * sigma_in[2],
                        variance_in[2],
                    ],
                ],
                mark="V",
            )
            gp = GaussianParameters(names, value, central, covariance=covariance)
        elif mode == "cov1d":
            covariance = Array("covariance", variance_in, mark="diag(V)")
            gp = GaussianParameters(names, value, central, covariance=covariance)
        elif mode == "cor":
            correlation = Array(
                "correlation",
                [
                    [1.0, corrs_in[0], corrs_in[1]],
                    [corrs_in[0], 1.0, corrs_in[2]],
                    [corrs_in[1], corrs_in[2], 1.0],
                ],
                mark="C",
            )
            gp = GaussianParameters(names, value, central, sigma=sigma, correlation=correlation)
        else:
            raise RuntimeError(f"Invalid mode {mode}")

    try:
        graph.close()
    except CriticalError as error:
        savegraph(graph, f"output/test_parameters_00_{mode}.png")
        raise error

    value_out0 = gp.value.data.copy()
    normvalue_out0 = gp.constraint.normvalue.data
    assert allclose(value_in, value_out0, atol=0, rtol=0)
    assert all(normvalue_out0 != 0)

    gp.constraint.normvalue.set(zeros_in)
    value_out1 = gp.value.data
    normvalue_out1 = gp.constraint.normvalue.data
    assert allclose(central_in, value_out1, atol=0, rtol=0)
    assert allclose(normvalue_out1, 0.0, atol=0, rtol=0)

    gp.value.set(value_out0)
    value_out2 = gp.value.data
    normvalue_out2 = gp.constraint.normvalue.data
    assert allclose(value_in, value_out2, atol=0, rtol=0)
    assert allclose(normvalue_out2, normvalue_out0, atol=0, rtol=0)

    savegraph(graph, f"output/test_parameters_00_{mode}.png", show=["all"])
    savegraph(graph, f"output/test_parameters_00_{mode}.pdf", show=["all"])


def test_parameters_01():
    vals = [1.0, 2.0, 3.0]
    val_init = 0.0
    with Graph(debug=False, close_on_exit=False):
        value = Array("variable", [val_init])
        par = Parameter(parent=None, value_output=value._output)

    for val in vals:
        assert par.push(val) == par.value == val
    vals_pop = [2.0, 1.0, val_init]
    for val in vals_pop:
        assert par.pop() == val

    with par:
        par.value = 2.0
        assert par.value == 2.0
        assert par._stack[-1] == val_init
        par.value = 4.0
        assert par.value == 4.0
    assert par.value == val_init
