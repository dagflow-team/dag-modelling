from random import choice
from string import ascii_lowercase

from numpy import linspace
from numpy.random import seed, shuffle
from pytest import mark, raises

from dagflow.core.exception import InitializationError
from dagflow.core.graph import Graph
from dagflow.lib.common import Array
from dagflow.lib.interpolation import SegmentIndex
from dagflow.plot.graphviz import savegraph


@mark.parametrize("mode", ("left", "right"))
@mark.parametrize("offset", (0, -1.0e-11, +1.0e-11, -0.5, 0.5, -7, 7))
@mark.parametrize("dtype", ("d", "f"))
def test_segmentIndex_01(debug_graph, testname, mode, offset: float | int, dtype: str):
    seed(10)
    nc, nf = 10, 100
    ne = nc + 1
    with Graph(debug=debug_graph, close_on_exit=True) as graph:
        coarseX = linspace(0, 10, nc + 1, dtype=dtype)
        fineX = linspace(0, 10, nf + 1, dtype=dtype) + offset
        shuffle(fineX)
        coarse = Array("coarse", coarseX, mode="fill")
        fine = Array("fine", fineX, mode="fill")
        segmentIndex = SegmentIndex("segmentIndex", mode=mode)
        (coarse, fine) >> segmentIndex

    expect = coarseX.searchsorted(fineX, side=mode)

    tolerance = segmentIndex.tolerance
    if mode == "right":
        mask = (expect == ne) * ((fineX - tolerance) <= coarseX[-1])
        expect[mask] = nc
    else:
        mask = (expect == 0) * ((fineX + tolerance) >= coarseX[0])
        expect[mask] = 1

    if dtype == "f":
        assert tolerance == 1.0e-4
    elif dtype == "d":
        assert tolerance == 1.0e-10

    res = segmentIndex.outputs[0].data
    assert all(res == expect)
    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("mode", ("left", "right"))
@mark.parametrize("offset", (0, -1.0e-11, +1.0e-11, -0.5, 0.5))
def test_segmentIndex_02(debug_graph, testname, mode, offset):
    seed(10)
    nc, nf = 10, 100
    ne = nc + 1
    coarseX = linspace(0, 10, nc + 1)
    fineX0 = linspace(0, 10, nf) + offset
    shuffle(fineX0)
    fineX = fineX0.reshape(4, nf // 4)

    with Graph(debug=debug_graph, close_on_exit=True) as graph:
        coarse = Array("coarse", coarseX, mode="fill")
        fine = Array("fine", fineX, mode="fill")
        segmentIndex = SegmentIndex("segmentIndex", mode=mode)
        (coarse, fine) >> segmentIndex
    expect = coarseX.searchsorted(fineX.ravel(), side=mode)

    tolerance = segmentIndex.tolerance
    if mode == "right":
        mask = (expect == ne) * ((fineX.ravel() - tolerance) <= coarseX[-1])
        expect[mask] = nc
    else:
        mask = (expect == 0) * ((fineX.ravel() + tolerance) >= coarseX[0])
        expect[mask] = 1

    res = segmentIndex.outputs[0].data.ravel()
    assert all(res == expect)
    savegraph(graph, f"output/{testname}.png")


@mark.parametrize("mode", ("left", "right"))
def test_segmentIndex_03(debug_graph, testname, mode):
    seed(10)
    nc, nf = 10, 100
    ne = nc + 1
    coarseX = linspace(0, 10, nc + 1)
    fineX0 = linspace(0, 10, nf)
    shuffle(fineX0)
    fineX = fineX0.reshape(4, nf // 4)
    with Graph(debug=debug_graph, close_on_exit=True) as graph:
        coarse = Array("coarse", coarseX, mode="fill")
        fine = Array("fine", fineX, mode="fill")
        segmentIndex = SegmentIndex("segmentIndex", mode=mode)
        (coarse, fine) >> segmentIndex

    expect = coarseX.searchsorted(fineX.ravel(), side=mode)

    tolerance = segmentIndex.tolerance
    if mode == "right":
        mask = (expect == ne) * ((fineX.ravel() - tolerance) <= coarseX[-1])
        expect[mask] = nc
    else:
        mask = (expect == 0) * ((fineX.ravel() + tolerance) >= coarseX[0])
        expect[mask] = 1

    res = segmentIndex.outputs[0].data.ravel()
    assert all(res == expect)
    savegraph(graph, f"output/{testname}.png")


def test_segmentIndex_exception(debug_graph):
    with Graph(debug=debug_graph, close_on_exit=False):
        with raises(InitializationError):
            SegmentIndex(
                "segmentIndex",
                mode="".join(choice(ascii_lowercase) for _ in range(5)),
            )
