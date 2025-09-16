from numpy import arange
from pytest import mark, raises

from dag_modelling.tools.make_fcn import make_fcn
from dag_modelling.core import NodeStorage, Graph
from dag_modelling.lib.common import Array
from dag_modelling.lib.linalg import LinearFunction
from dag_modelling.parameters import Parameters
from dag_modelling.plot.graphviz import savegraph


@mark.parametrize("par_names,mapper", (
    ({}, None),
    ({"a": 2.0, "b.IDX1": 3.0}, {"b.IDX1": "par1"}),
    ({"a": 2.0}, {"a": "par0"}),
    ({"b.IDX1": 3.0}, {"a": "par0"}),
))
@mark.parametrize("pass_output", (False, True))
def test_make_fcn_safe(test_name, par_names, mapper, pass_output, output_path: str):
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
    LF = make_fcn(
        f.outputs[0] if pass_output else f,
        storage["parameters.all"],
        par_names=par_names,
        mapper=mapper,
        safe=True,
    )

    if par_names:
        with raises(KeyError):
            LF(non_existent=12)

        with raises(RuntimeError):
            LF(*[value for value in par_names.values()], non_existent=12)

    res1 = LF(*[value for value in par_names.values()])
    res2 = LF()

    if mapper:
        res3 = LF(**{mapper.get(parname, parname): parvalue for parname, parvalue in par_names.items()})
        assert all(res1 == res3)

    # pars equal inital values
    assert A.value == vals_in[0]
    assert B.value == vals_in[1]
    # new result differs from the result of LF
    assert all(res1 == (par_names.get("a", vals_in[0]) * x + par_names.get("b.IDX1", vals_in[1])))
    assert all(res0 == (vals_in[0] * x + vals_in[1]))
    assert all(res2 == res0)

    savegraph(graph, f"{output_path}/{test_name}.png")


@mark.parametrize("par_names,mapper", (
    ({}, None),
    ({"parameters.all.a": 2.0, "parameters.all.b.IDX1": 3.0}, {"parameters.all.b.IDX1": "par1"}),
    ({"parameters.all.a": 2.0}, {"parameters.all.a": "par0"}),
    ({"parameters.all.b.IDX1": 3.0}, {"parameters.all.a": "par0"}),
))
@mark.parametrize("pass_output", (False, True))
def test_make_fcn_nonsafe(test_name, par_names, mapper, pass_output, output_path: str):
    n = 10
    x = arange(n, dtype="d")
    vals_in = [1.0, 2.0]
    val_new = 3.0
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

    res0 = f.outputs[0].data
    res0c = res0.copy()
    LF = make_fcn(
        f.outputs[0] if pass_output else f,
        storage,
        mapper=mapper,
        par_names=par_names,
        safe=False,
    )

    if par_names:
        with raises(KeyError):
            LF(non_existent=12)

        with raises(RuntimeError):
            LF(*[value for value in par_names.values()], non_existent=12)

    res1 = LF(*[value for value in par_names.values()])
    res2 = LF()

    if mapper:
        res3 = LF(**{mapper.get(parname, parname): parvalue for parname, parvalue in par_names.items()})
        assert all(res1 == res3)

    # pars equal new values
    assert A.value == par_names.get("parameters.all.a", vals_in[0])
    assert B.value == par_names.get("parameters.all.b.IDX1", vals_in[1])
    # new result is the same as the result of LF
    assert all(res0c == (vals_in[0] * x + vals_in[1]))
    assert all(res1 == (par_names.get("parameters.all.a", vals_in[0]) * x + par_names.get("parameters.all.b.IDX1", vals_in[1])))
    assert all(res1 == res0)
    assert all(res1 == res2)

    savegraph(graph, f"{output_path}/{test_name}.png")
