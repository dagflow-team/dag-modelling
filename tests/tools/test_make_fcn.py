from numpy import arange
from pytest import mark, raises

from dag_modelling.core import Graph, NodeStorage
from dag_modelling.lib.common import Array
from dag_modelling.lib.linalg import LinearFunction
from dag_modelling.parameters import Parameters
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.tools.make_fcn import make_fcn


@mark.parametrize(
    "par_dict,pass_dict",
    (
        ({}, False),
        ({"a": 2.0, "b.IDX1": 3.0}, True),
        ({"a": 2.0}, False),
        ({"b.IDX1": 3.0}, True),
    ),
)
@mark.parametrize("pass_output", (False, True))
def test_make_fcn_safe(test_name, par_dict, pass_dict, pass_output, output_path: str):
    n = 10
    x = arange(n, dtype="d")
    vals_in = [1.0, 2.0]
    names = ("a", "b")

    with Graph(close_on_exit=True) as graph:
        pars = Parameters.from_numbers(value=vals_in, names=names)
        storage = NodeStorage(
            {"parameters": {"all": {"a": pars._pars[0], "b": {"IDX1": pars._pars[1]}}}}
        )
        f = LinearFunction("ax+b")
        A, B = pars._pars
        A >> f("a")
        B >> f("b")
        Array("x", x, mode="fill") >> f

    res0 = f.outputs[0].data
    if pass_dict:
        parameters = {parname: storage[f"parameters.all.{parname}"] for parname in par_dict.keys()}
    else:
        parameters = [storage[f"parameters.all.{parname}"] for parname in par_dict.keys()]
    LF = make_fcn(
        f.outputs[0] if pass_output else f,
        parameters=parameters,
        safe=True,
    )
    res1 = LF()

    if par_dict:
        with raises(KeyError):
            LF(non_existent=12)

        with raises(RuntimeError):
            LF(*[value for value in par_dict.values()], -1)

    if pass_dict:
        res2 = LF(
            **{parname: parvalue for parname, parvalue in zip(parameters.keys(), par_dict.values())}
        )
    else:
        res2 = LF(*[value for value in par_dict.values()])

    # pars equal old values
    assert A.value == vals_in[0]
    assert B.value == vals_in[1]
    # new result is the same as the result of LF
    assert all(res0 == (vals_in[0] * x + vals_in[1]))
    assert all(res1 == res0)
    assert all(res2 == par_dict.get("a", vals_in[0]) * x + par_dict.get("b.IDX1", vals_in[1]))

    savegraph(graph, f"{output_path}/{test_name}.png")


@mark.parametrize(
    "par_dict,pass_dict",
    (
        ({}, False),
        ({"parameters.all.a": 2.0, "parameters.all.b.IDX1": 3.0}, True),
        ({"parameters.all.a": 2.0}, False),
        ({"parameters.all.b.IDX1": 3.0}, True),
    ),
)
@mark.parametrize("pass_output", (False, True))
def test_make_fcn_nonsafe(test_name, par_dict, pass_dict, pass_output, output_path: str):
    n = 10
    x = arange(n, dtype="d")
    vals_in = [1.0, 2.0]
    names = ("a", "b.IDX1")

    with Graph(close_on_exit=True) as graph:
        pars = Parameters.from_numbers(value=vals_in, names=names)
        storage = NodeStorage(
            {"parameters": {"all": {"a": pars._pars[0], "b": {"IDX1": pars._pars[1]}}}}
        )
        f = LinearFunction("ax+b")
        A, B = pars._pars
        A >> f("a")
        B >> f("b")
        Array("x", x, mode="fill") >> f

    if pass_dict:
        parameters = {parname: storage[parname] for parname in par_dict.keys()}
    else:
        parameters = [storage[parname] for parname in par_dict.keys()]
    res0 = f.outputs[0].data
    res0c = res0.copy()
    LF = make_fcn(
        f.outputs[0] if pass_output else f,
        parameters=parameters,
        safe=False,
    )

    if par_dict:
        with raises(KeyError):
            LF(non_existent=12)

        with raises(RuntimeError):
            LF(*[value for value in par_dict.values()], -1)

    res1 = LF(*[value for value in par_dict.values()])
    res2 = LF()

    # pars equal new values
    assert A.value == par_dict.get("parameters.all.a", vals_in[0])
    assert B.value == par_dict.get("parameters.all.b.IDX1", vals_in[1])
    # new result is the same as the result of LF
    assert all(res0c == (vals_in[0] * x + vals_in[1]))
    assert all(
        res1
        == (
            par_dict.get("parameters.all.a", vals_in[0]) * x
            + par_dict.get("parameters.all.b.IDX1", vals_in[1])
        )
    )
    assert all(res1 == res0)
    assert all(res1 == res2)

    savegraph(graph, f"{output_path}/{test_name}.png")
