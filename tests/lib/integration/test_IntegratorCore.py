from matplotlib.pyplot import close, subplots
from numpy import allclose, linspace, meshgrid, pi, vectorize
from pytest import mark, raises

from dagflow.core.exception import CalculationError, CriticalError, TypeFunctionError
from dagflow.core.graph import Graph
from dagflow.lib.abstract import ManyToOneNode, OneToOneNode
from dagflow.lib.common import Array
from dagflow.lib.integration import IntegratorCore, IntegratorSampler
from dagflow.lib.trigonometry import Cos, Sin
from dagflow.plot.graphviz import savegraph
from dagflow.plot.plot import plot_auto


@mark.parametrize("align", ("left", "center", "right"))
def test_IntegratorCore_rect_center(align, debug_graph, testname):
    with Graph(debug=debug_graph, close_on_exit=True) as graph:
        npoints = 10
        edges = Array("edges", linspace(0, pi, npoints + 1), mode="fill")
        orders_x = Array("orders_x", [1000] * npoints, edges=edges["array"], mode="store")
        A = Array("A", edges._data[:-1], mode="fill")
        B = Array("B", edges._data[1:], mode="fill")
        sampler = IntegratorSampler("sampler", mode="rect", align=align)
        integrator = IntegratorCore("integrator")
        cosf = Cos("cos")
        sinf = Sin("sin")
        orders_x >> sampler("orders_x")
        sampler.outputs["x"] >> cosf
        A >> sinf
        B >> sinf
        sampler.outputs["weights"] >> integrator("weights")
        cosf.outputs[0] >> integrator
        orders_x >> integrator("orders_x")
    ires = integrator.outputs[0].data  # goes first: ensure integrator triggers calculations
    res = sinf.outputs[1].data - sinf.outputs[0].data
    assert allclose(ires, res, atol=1e-4)
    integrator.taint()
    integrator.touch()
    assert allclose(integrator.outputs[0].data, res, atol=1e-4)
    assert integrator.outputs[0].dd.axes_edges == [edges["array"]]
    savegraph(graph, f"output/{testname}.png")


def test_IntegratorCore_trap(debug_graph, testname):
    with Graph(debug=debug_graph, close_on_exit=True) as graph:
        npoints = 10
        edges = Array("edges", linspace(0, pi, npoints + 1), mode="fill")
        orders_x = Array("orders_x", [1000] * npoints, edges=edges["array"], mode="store")
        A = Array("A", edges._data[:-1], mode="fill")
        B = Array("B", edges._data[1:], mode="fill")
        sampler = IntegratorSampler("sampler", mode="trap")
        integrator = IntegratorCore("integrator")
        cosf = Cos("cos")
        sinf = Sin("sin")
        orders_x >> sampler("orders_x")
        sampler.outputs["x"] >> cosf
        A >> sinf
        B >> sinf
        sampler.outputs["weights"] >> integrator("weights")
        cosf.outputs[0] >> integrator
        orders_x >> integrator("orders_x")
    ires = integrator.outputs[0].data  # goes first: ensure integrator triggers calculations
    res = sinf.outputs[1].data - sinf.outputs[0].data
    assert allclose(ires, res, atol=1e-2)
    assert integrator.outputs[0].dd.axes_edges == [edges["array"]]
    savegraph(graph, f"output/{testname}.png")


def f0(x: float) -> float:
    return 4 * x**3 + 3 * x**2 + 2 * x - 1


def fres(x: float) -> float:
    return x**4 + x**3 + x**2 - x


vecF0 = vectorize(f0)
vecFres = vectorize(fres)


class Polynomial0(OneToOneNode):
    def _function(self):
        for indata, outdata in zip(self.inputs.iter_data(), self.outputs.iter_data_unsafe()):
            outdata[:] = vecF0(indata)


class PolynomialRes(OneToOneNode):
    def _function(self):
        for indata, outdata in zip(self.inputs.iter_data(), self.outputs.iter_data_unsafe()):
            outdata[:] = vecFres(indata)


def test_IntegratorCore_gl1d(debug_graph, testname):
    with Graph(debug=debug_graph, close_on_exit=True) as graph:
        npoints = 10
        edges = Array("edges", linspace(0, 10, npoints + 1), mode="fill")
        orders_x = Array("orders_x", [2] * npoints, edges=edges["array"], mode="store")
        A = Array("A", edges._data[:-1], mode="fill")
        B = Array("B", edges._data[1:], mode="fill")
        sampler = IntegratorSampler("sampler", mode="gl")
        integrator = IntegratorCore("integrator")
        poly0 = Polynomial0("poly0")
        polyres = PolynomialRes("polyres")
        orders_x >> sampler("orders_x")
        sampler.outputs["x"] >> poly0
        A >> polyres
        B >> polyres
        sampler.outputs["weights"] >> integrator("weights")
        poly0.outputs[0] >> integrator
        orders_x >> integrator("orders_x")
    ires = integrator.outputs[0].data  # goes first: ensure integrator triggers calculations
    res = polyres.outputs[1].data - polyres.outputs[0].data
    assert allclose(ires, res, atol=1e-10)
    assert integrator.outputs[0].dd.axes_edges == [edges["array"]]
    savegraph(graph, f"output/{testname}.png")


def test_IntegratorCore_gl2d(debug_graph, testname):
    class Polynomial1(ManyToOneNode):
        scale = 1.0

        def _function(self):
            self.outputs["result"]._data[:] = (
                self.scale * vecF0(self.inputs[1].data) * vecF0(self.inputs[0].data)
            )

    with Graph(debug=debug_graph, close_on_exit=True) as graph:
        npointsX, npointsY = 10, 20
        edgesX = Array(
            "edgesX",
            linspace(0, 10, npointsX + 1),
            label={"axis": "Label for axis X"},
        )
        edgesY = Array(
            "edgesY",
            linspace(2, 12, npointsY + 1),
            label={"axis": "Label for axis Y"},
        )
        orders_x = Array("orders_x", [2] * npointsX, edges=edgesX["array"], mode="store")
        orders_y = Array("orders_y", [2] * npointsY, edges=edgesY["array"], mode="store")
        x0, y0 = meshgrid(edgesX._data[:-1], edgesY._data[:-1], indexing="ij")
        x1, y1 = meshgrid(edgesX._data[1:], edgesY._data[1:], indexing="ij")
        X0, X1 = Array("X0", x0, mode="fill"), Array("X1", x1)
        Y0, Y1 = Array("Y0", y0, mode="fill"), Array("Y1", y1)
        sampler = IntegratorSampler("sampler", mode="gl2d")
        integrator = IntegratorCore(
            "integrator",
            label={
                "plot_title": f"IntegratorCore test: {testname}",
                "axis": "integral",
            },
        )
        poly0 = Polynomial1("poly0")
        polyres = PolynomialRes("polyres")
        orders_x >> sampler("orders_x")
        orders_y >> sampler("orders_y")
        sampler.outputs["x"] >> poly0
        sampler.outputs["y"] >> poly0
        X0 >> polyres
        X1 >> polyres
        Y0 >> polyres
        Y1 >> polyres
        sampler.outputs["weights"] >> integrator("weights")
        poly0.outputs[0] >> integrator
        orders_x >> integrator("orders_x")
        orders_y >> integrator("orders_y")

    ires = integrator.outputs[0].data  # goes first: ensure integrator triggers calculations
    res = (polyres.outputs[1].data - polyres.outputs[0].data) * (
        polyres.outputs[3].data - polyres.outputs[2].data
    )
    assert allclose(ires, res, atol=1e-10)
    integrator.taint()
    integrator.touch()
    assert allclose(integrator.outputs[0].data, res, atol=1e-10)
    assert integrator.outputs[0].dd.axes_edges == [
        edgesX["array"],
        edgesY["array"],
    ]

    subplots(1, 1, subplot_kw={"projection": "3d"})
    plot_auto(integrator, plotoptions="bar3d", colorbar=True)

    subplots(1, 1)
    plot_auto(integrator, plotoptions="pcolormesh", colorbar=True)

    subplots(1, 1)
    plot_auto(integrator, plotoptions="pcolor", colorbar=True)

    subplots(1, 1)
    plot_auto(integrator, plotoptions="pcolorfast", colorbar=True)

    subplots(1, 1)
    plot_auto(integrator, plotoptions="imshow", colorbar=True)

    subplots(1, 1)
    plot_auto(integrator, plotoptions="matshow", colorbar=True)

    subplots(1, 1)
    plot_auto(integrator, plotoptions="matshow", extent=None, colorbar=True)

    for _ in range(7):
        close()

    poly0.scale = 2.2
    poly0.taint()
    assert allclose(integrator.outputs[0].data, res * poly0.scale, atol=1e-10)


    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("dropdim", (True, False))
def test_IntegratorCore_gl2to1d_x(debug_graph, testname, dropdim):
    class Polynomial21(ManyToOneNode):
        def _function(self):
            self.outputs["result"]._data[:] = vecF0(self.inputs[1].data) * vecF0(
                self.inputs[0].data
            )

    with Graph(debug=debug_graph, close_on_exit=True) as graph:
        npointsX = 20
        edgesX = Array(
            "edgesX",
            linspace(2, 12, npointsX + 1),
            label={"axis": "Label for axis X"},
        )
        edgesY = Array(
            "edgesY",
            [0, 1],
            label={"axis": "Label for axis Y"},
        )
        orders_x = Array("orders_x", [2] * npointsX, edges=edgesX["array"], mode="store")
        orders_y = Array("orders_y", [2], edges=edgesY["array"], mode="store")
        x0, y0 = meshgrid(edgesX._data[:-1], edgesY._data[:-1], indexing="ij")
        x1, y1 = meshgrid(edgesX._data[1:], edgesY._data[1:], indexing="ij")
        X0, X1 = Array("X0", x0, mode="fill"), Array("X1", x1)
        Y0, Y1 = Array("Y0", y0, mode="fill"), Array("Y1", y1)
        sampler = IntegratorSampler("sampler", mode="gl2d")
        integrator = IntegratorCore(
            "integrator",
            dropdim=dropdim,
            label={
                "plot_title": f"IntegratorCore test: {testname}",
                "axis": "integral",
            },
        )
        poly0 = Polynomial21("poly0")
        polyres = PolynomialRes("polyres")
        orders_x >> sampler("orders_x")
        orders_y >> sampler("orders_y")
        sampler.outputs["x"] >> poly0
        sampler.outputs["y"] >> poly0
        X0 >> polyres
        X1 >> polyres
        Y0 >> polyres
        Y1 >> polyres
        sampler.outputs["weights"] >> integrator("weights")
        poly0.outputs[0] >> integrator
        orders_x >> integrator("orders_x")
        orders_y >> integrator("orders_y")
    ires = integrator.outputs[0].data  # goes first: ensure integrator triggers calculations
    if dropdim:
        res = (polyres.outputs[1].data.T - polyres.outputs[0].data.T) * (
            polyres.outputs[3].data.T - polyres.outputs[2].data.T
        )[0]
        edges = [edgesX["array"]]
    else:
        res = (polyres.outputs[1].data - polyres.outputs[0].data) * (
            polyres.outputs[3].data - polyres.outputs[2].data
        )
        edges = [edgesX["array"], edgesY["array"]]
    assert allclose(ires, res, atol=1e-10)
    assert integrator.outputs[0].dd.axes_edges == edges

    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("dropdim", (True, False))
def test_IntegratorCore_gl2to1d_y(debug_graph, testname, dropdim):
    class Polynomial21(ManyToOneNode):
        def _function(self):
            self.outputs["result"]._data[:] = vecF0(self.inputs[1].data) * vecF0(
                self.inputs[0].data
            )

    with Graph(debug=debug_graph, close_on_exit=True) as graph:
        npointsY = 20
        edgesX = Array("edgesX", [0, 1], label={"axis": "Label for axis X"}, mode="fill")
        edgesY = Array(
            "edgesY",
            linspace(2, 12, npointsY + 1),
            label={"axis": "Label for axis Y"},
        )
        orders_x = Array("orders_x", [2], edges=edgesX["array"], mode="store")
        orders_y = Array("orders_y", [2] * npointsY, edges=edgesY["array"], mode="store")
        x0, y0 = meshgrid(edgesX._data[:-1], edgesY._data[:-1], indexing="ij")
        x1, y1 = meshgrid(edgesX._data[1:], edgesY._data[1:], indexing="ij")
        X0, X1 = Array("X0", x0, mode="fill"), Array("X1", x1)
        Y0, Y1 = Array("Y0", y0, mode="fill"), Array("Y1", y1)
        sampler = IntegratorSampler("sampler", mode="gl2d")
        integrator = IntegratorCore(
            "integrator",
            dropdim=dropdim,
            label={
                "plot_title": f"IntegratorCore test: {testname}",
                "axis": "integral",
            },
        )
        poly0 = Polynomial21("poly0")
        polyres = PolynomialRes("polyres")
        orders_x >> sampler("orders_x")
        orders_y >> sampler("orders_y")
        sampler.outputs["x"] >> poly0
        sampler.outputs["y"] >> poly0
        X0 >> polyres
        X1 >> polyres
        Y0 >> polyres
        Y1 >> polyres
        sampler.outputs["weights"] >> integrator("weights")
        poly0.outputs[0] >> integrator
        orders_x >> integrator("orders_x")
        orders_y >> integrator("orders_y")
    ires = integrator.outputs[0].data  # goes first: ensure integrator triggers calculations
    if dropdim:
        res = (polyres.outputs[1].data - polyres.outputs[0].data) * (
            polyres.outputs[3].data - polyres.outputs[2].data
        )[0]
        edges = [edgesY["array"]]
    else:
        res = (polyres.outputs[1].data - polyres.outputs[0].data) * (
            polyres.outputs[3].data - polyres.outputs[2].data
        )
        edges = [edgesX["array"], edgesY["array"]]
    assert allclose(ires, res, atol=1e-10)
    assert integrator.outputs[0].dd.axes_edges == edges

    savegraph(graph, f"output/{testname}.png")


def test_IntegratorCore_orders_0(debug_graph):
    """test wrong ordersX: edges not given"""
    arr = [1.0, 2.0, 3.0]
    with Graph(debug=debug_graph):
        arr1 = Array("array", arr, mode="fill")
        weights = Array("weights", arr, mode="fill")
        orders_x = Array("orders_x", [1, 2, 3], mode="fill")
        integrator = IntegratorCore("integrator")
        arr1 >> integrator
        weights >> integrator("weights")
        orders_x >> integrator("orders_x")

    with raises(CalculationError):
        integrator.close()


def test_IntegratorSampler_orders_0(debug_graph):
    """test wrong ordersX: edges not given"""
    arr = [1.0, 2.0, 3.0]
    with Graph(debug=debug_graph) as graph:
        edges = Array("edges", linspace(0, pi, len(arr) + 1), mode="fill")
        arr1 = Array("array", arr, mode="fill")
        weights = Array("weights", arr, mode="fill")
        orders_x_0 = Array("orders_x", [1, 2, 3], mode="store", edges=edges["array"])
        sampler = IntegratorSampler("sampler", mode="trap")
        integrator = IntegratorCore("integrator")
        arr1 >> integrator
        weights >> integrator("weights")
        orders_x_0 >> integrator("orders_x")
        orders_x_0 >> sampler("orders_x")

    with raises(CriticalError):
        orders_x_0.outputs["array"].set(3)


def test_IntegratorSampler_orders_1(debug_graph):
    """test wrong ordersX: edges not given"""
    arr = [1.0, 2.0, 3.0]
    with Graph(debug=debug_graph) as graph:
        edges = Array("edges", linspace(0, pi, len(arr) + 1), mode="fill")
        arr1 = Array("array", arr, mode="fill")
        weights = Array("weights", arr, mode="fill")
        orders_x_0 = Array("orders_x", [1, 2, 3], mode="store")
        orders_x_1 = Array("orders_x", [1, 2, 3], mode="fill", edges=edges["array"])
        sampler = IntegratorSampler("sampler", mode="trap")
        integrator = IntegratorCore("integrator")
        arr1 >> integrator
        weights >> integrator("weights")
        orders_x_0 >> integrator("orders_x")
        orders_x_1 >> sampler("orders_x")

    with raises(CalculationError):
        graph.close()


def test_IntegratorCore_edges_0(debug_graph):
    """test wrong ordersX: edges not given"""
    arr = [1.0, 2.0, 3.0]
    with Graph(debug=debug_graph):
        arr1 = Array("array", arr, mode="fill")
        weights = Array("weights", arr, mode="fill")
        orders_x = Array("orders_x", [1, 2, 3], mode="store")
        integrator = IntegratorCore("integrator")
        arr1 >> integrator
        weights >> integrator("weights")
        orders_x >> integrator("orders_x")

    with raises(TypeFunctionError):
        integrator.close()


def test_IntegratorCore_edges_1(debug_graph):
    """test wrong ordersX: edges is wrong"""
    arr = [1.0, 2.0, 3.0]
    with Graph(debug=debug_graph, close_on_exit=False):
        edges = Array("edges", [0.0, 1.0, 2.0], mode="fill")
        with raises(TypeFunctionError):
            arr1 = Array("array", arr, edges=edges["array"], mode="store")
        edges = Array("edges", [0.0, 1.0, 2.0, 3.0], mode="fill")
        arr1 = Array("array", arr, edges=edges["array"], mode="fill")
        weights = Array("weights", arr, mode="fill")
        orders_x = Array("orders_x", [1, 2, 3], mode="store")
        integrator = IntegratorCore("integrator")
        arr1 >> integrator
        weights >> integrator("weights")
        orders_x >> integrator("orders_x")

    with raises(TypeFunctionError):
        integrator.close()


def test_IntegratorCore_02(debug_graph):
    """test wrong ordersX: sum(ordersX) != shape"""
    arr = [1.0, 2.0, 3.0]
    with Graph(debug=debug_graph):
        edges = Array("edges", [0.0, 1.0, 2.0, 3.0], mode="fill")
        arr1 = Array("array", arr, edges=edges["array"], mode="fill")
        weights = Array("weights", arr, edges=edges["array"], mode="fill")
        orders_x = Array("orders_x", [1, 2, 3], edges=edges["array"], mode="store")
        integrator = IntegratorCore("integrator")
        arr1 >> integrator
        weights >> integrator("weights")
        orders_x >> integrator("orders_x")
    with raises(TypeFunctionError):
        integrator.close()


def test_IntegratorCore_03(debug_graph):
    """test wrong ordersX: sum(ordersX[i]) != shape[i]"""
    arr = [1.0, 2.0, 3.0]
    with Graph(debug=debug_graph, close_on_exit=False):
        edgesX = Array("edgesX", [-1.0, 0.0, 1.0], mode="fill")
        edgesY = Array("edgesY", [-2.0, -1, 0.0, 1.0], mode="fill")
        arr1 = Array("array", [arr, arr], edges=[edgesX["array"], edgesY["array"]], mode="fill")
        weights = Array(
            "weights", [arr, arr], edges=[edgesX["array"], edgesY["array"]], mode="fill"
        )
        orders_x = Array("orders_x", [1, 3], edges=edgesX["array"], mode="store")
        orders_y = Array("orders_y", [1, 0, 0], edges=edgesY["array"], mode="store")
        integrator = IntegratorCore("integrator")
        arr1 >> integrator
        weights >> integrator("weights")
        orders_x >> integrator("orders_x")
        orders_y >> integrator("orders_y")
    with raises(TypeFunctionError):
        integrator.close()


def test_IntegratorCore_04(debug_graph):
    """Test wrong shape."""
    with Graph(debug=debug_graph, close_on_exit=False):
        arr1 = Array("array", [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], mode="fill")
        arr2 = Array("array", [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]], mode="fill")
        weights = Array("weights", [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], mode="fill")
        orders_x = Array("orders_x", [0, 2], mode="store")
        orders_y = Array("orders_y", [1, 1, 1, 3], mode="store")
        integrator = IntegratorCore("integrator")
        arr1 >> integrator
        arr2 >> integrator
        weights >> integrator("weights")
        orders_x >> integrator("orders_x")
        orders_y >> integrator("orders_y")
    with raises(TypeFunctionError):
        integrator.close()
