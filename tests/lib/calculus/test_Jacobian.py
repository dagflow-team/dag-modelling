from numpy import allclose, arange, array, diag, finfo, ones
from pytest import mark

from dagflow.core.graph import Graph
from dagflow.plot.graphviz import savegraph
from dagflow.lib.arithmetic import Product, Sum
from dagflow.lib.common import Array, Concatenation, Copy
from dagflow.lib.calculus import Jacobian
from dagflow.lib.calculus.jacobian import compute_jacobian
from dagflow.lib.linalg import LinearFunction
from dagflow.parameters import GaussianParameter


@mark.parametrize("dtype", ("d", "f"))
def test_Jacobian_01(dtype, testname):
    """
    Test of the identity Jacobian.
    y_i = a_i, so the Jacobian is [dy/da_i]_i = 1 and [dy/da_i]_j = 0 when j != i
    """
    size = 10
    scale = 0.1
    values = arange(size, dtype=dtype)
    sigmas = ones(size, dtype=dtype)

    with Graph(close_on_exit=True) as graph:
        valueslist = [Array(f"arr_{i:02d}", [val], mode="fill") for i, val in enumerate(values)]
        sigmaslist = [Array(f"sigma_{i:02d}", [sigma], mode="fill") for i, sigma in enumerate(sigmas)]
        pars = [
            GaussianParameter(
                parent=None,
                value_output=value._output,
                central_output=value._output,
                normvalue_output=value._output,
                sigma_output=sigma._output,
            )
            for value, sigma in zip(valueslist, sigmaslist)
        ]
        jac = Jacobian(
            "Jacobian",
            scale=scale,
            parameters=pars,
        )
        parsconcat = Concatenation("parameters")
        valueslist >> parsconcat
        parsconcat >> jac

        jac2 = Copy("jac_copy")
        jac >> jac2

    assert jac.tainted is True
    assert jac2.tainted is True
    res = jac.outputs[0].data[:]
    jac2.touch()
    assert jac.tainted is False
    assert jac2.tainted is False

    factors = {
        "d": 30,
        "f": 10,
    }
    assert allclose(diag(res), 1, atol=factors[dtype] * finfo(dtype).resolution, rtol=0)

    jac.taint()
    assert jac.tainted is False
    assert jac2.tainted is False
    new_res = jac.outputs[0].data
    assert allclose(new_res, res, atol=factors[dtype] * finfo(dtype).resolution, rtol=0)

    jac.unfreeze()
    assert jac.tainted is True
    assert jac2.tainted is True
    new_res = jac.outputs[0].data
    assert allclose(new_res, res, atol=factors[dtype] * finfo(dtype).resolution, rtol=0)

    jac.compute()
    assert jac.tainted is False
    assert jac2.tainted is True
    new_res = jac.outputs[0].data
    new_res2 = jac2.outputs[0].data
    assert allclose(new_res, res, atol=factors[dtype] * finfo(dtype).resolution, rtol=0)
    assert allclose(new_res2, res, atol=factors[dtype] * finfo(dtype).resolution, rtol=0)
    assert jac.tainted is False
    assert jac2.tainted is False

    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("dtype", ("d", "f"))
def test_Jacobian_02(dtype, testname):
    """
    Test of the linear function input.
    y = a x + b, so the Jacobian must be [dy/da, dy/db]_i = [x_i, 1]
    """
    size = 10
    scale = 0.1

    a = 2.3
    b = -3.2
    x = arange(size, dtype=dtype)
    sigmas = (0.5, 2.0)

    with Graph(close_on_exit=True) as graph:
        X = Array("x", x, mode="fill")
        A = Array("a", array([a], dtype=dtype), mode="fill")
        B = Array("b", array([b], dtype=dtype), mode="fill")
        Y = LinearFunction("f(x)", label="f(x)=ax+b")
        A >> Y("a")
        B >> Y("b")
        X >> Y

        sigmaslist = [Array(f"sigma_{i:02d}", [sigma], mode="fill") for i, sigma in enumerate(sigmas)]
        pars = [
            GaussianParameter(
                parent=None,
                value_output=value._output,
                central_output=value._output,
                normvalue_output=value._output,
                sigma_output=sigma._output,
            )
            for value, sigma in zip((A, B), sigmaslist)
        ]
        jac = Jacobian(
            "Jacobian",
            scale=scale,
            parameters=pars,
        )
        Y >> jac

    res = jac.outputs[0].data
    factors = {
        "d": [100, 20],
        "f": [100, 10],
    }
    assert allclose(res[:, 1], 1, atol=factors[dtype][1] * finfo(dtype).resolution, rtol=0)
    assert allclose(res[:, 0], x, atol=factors[dtype][0] * finfo(dtype).resolution, rtol=0)

    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("dtype", ("d", "f"))
def test_Jacobian_03(dtype, testname):
    """
    Test of the linear function input.
    y = a*a*x + b*x + c, so the Jacobian must be [dy/da, dy/db, dy/dc]_i = [2*a*x_i, x_i, 1]
    """
    size = 10
    scale = 0.1

    a = 2.3
    b = -3.2
    c = 1.1
    x = arange(size, dtype=dtype)
    sigmas = (0.5, 2.0, 3.0)

    with Graph(close_on_exit=True) as graph:
        X = Array("x", x, mode="fill")
        A = Array("a", array([a], dtype=dtype), mode="fill")
        B = Array("b", array([b], dtype=dtype), mode="fill")
        C = Array("c", array([c], dtype=dtype), mode="fill")

        first = Product.from_args("a²x", A, A, X)
        second = Product.from_args("bx", B, X)
        Y = Sum.from_args("f(x)=a²x+bx+c", first, second, C)

        sigmaslist = [Array(f"sigma_{i:02d}", [sigma], mode="fill") for i, sigma in enumerate(sigmas)]
        pars = [
            GaussianParameter(
                parent=None,
                value_output=value._output,
                central_output=value._output,
                normvalue_output=value._output,
                sigma_output=sigma._output,
            )
            for value, sigma in zip((A, B, C), sigmaslist)
        ]
        jac = Jacobian(
            "Jacobian",
            scale=scale,
            parameters=pars,
        )
        Y >> jac

    dax = 2 * a * x

    output = jac.outputs[0]
    res = output.data
    factors = {
        "d": [2000, 100, 20],
        "f": [300, 20, 10],
    }
    assert allclose(res[:, 2], 1, atol=factors[dtype][2] * finfo(dtype).resolution, rtol=0)
    assert allclose(res[:, 1], x, atol=factors[dtype][1] * finfo(dtype).resolution, rtol=0)
    assert allclose(res[:, 0], dax, atol=factors[dtype][0] * finfo(dtype).resolution, rtol=0)

    assert not jac.tainted
    output.set(-1.0)
    assert jac.frozen
    jac.compute()
    assert not jac.tainted

    res = jac.outputs[0].data
    assert allclose(res[:, 2], 1, atol=factors[dtype][2] * finfo(dtype).resolution, rtol=0)
    assert allclose(res[:, 1], x, atol=factors[dtype][1] * finfo(dtype).resolution, rtol=0)
    assert allclose(res[:, 0], dax, atol=factors[dtype][0] * finfo(dtype).resolution, rtol=0)

    jac2 = compute_jacobian(Y.outputs[0], pars, scale=scale)
    assert allclose(res, jac2, atol=0, rtol=0)

    savegraph(graph, f"output/{testname}.png", show="full")
