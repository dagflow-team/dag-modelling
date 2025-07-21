from numpy import allclose, geomspace, linspace
from pytest import mark

from dagflow.core.graph import Graph
from dagflow.plot.graphviz import savegraph
from dagflow.lib.common import Array, Cache, Copy


@mark.parametrize("dtype", ("d", "f"))
def test_cache_copy(testname, debug_graph, dtype):
    array_in_1 = linspace(-10, 10, 101, dtype=dtype)
    array_in_2 = geomspace(0.00001, 10, 101, dtype=dtype)

    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        array1 = Array("arr1", array_in_1, mode="fill")
        array2 = Array("arr2", array_in_2, mode="fill")

        cache = Cache("cache")
        array1 >> cache
        array2 >> cache

        copy = Copy("copy")
        array1 >> copy
        array2 >> copy

        copy2 = Copy("copy 2")
        cache.outputs[0] >> copy2

    assert cache.tainted == True
    assert copy.tainted == True
    assert copy2.tainted == True
    assert all(output.dd.dtype == dtype for output in cache.outputs)
    assert all(output.dd.dtype == dtype for output in copy.outputs)
    assert allclose(cache.outputs[0].data, array_in_1, rtol=0, atol=0)
    assert allclose(copy.outputs[0].data, array_in_1, rtol=0, atol=0)
    assert allclose(copy2.outputs[0].data, array_in_1, rtol=0, atol=0)
    assert cache.tainted == False
    assert copy.tainted == False
    assert copy2.tainted == False
    assert allclose(cache.outputs[1].data, array_in_2, rtol=0, atol=0)
    assert allclose(copy.outputs[1].data, array_in_2, rtol=0, atol=0)

    array_in_1b = array_in_1 * 2
    array1.outputs[0].set(array_in_1b)
    assert cache.tainted == False
    assert copy.tainted == True
    assert copy2.tainted == False
    assert allclose(cache.outputs[0].data, array_in_1, rtol=0, atol=0)
    assert allclose(cache.outputs[1].data, array_in_2, rtol=0, atol=0)
    assert allclose(copy.outputs[0].data, array_in_1b, rtol=0, atol=0)
    assert allclose(copy2.outputs[0].data, array_in_1, rtol=0, atol=0)
    assert copy.tainted == False
    assert copy2.tainted == False
    assert allclose(copy.outputs[1].data, array_in_2, rtol=0, atol=0)

    cache.recache()
    assert cache.tainted == False
    assert copy.tainted == False
    assert copy2.tainted == True
    assert allclose(cache.outputs[0].data, array_in_1b, rtol=0, atol=0)
    assert allclose(cache.outputs[1].data, array_in_2, rtol=0, atol=0)
    assert allclose(copy.outputs[0].data, array_in_1b, rtol=0, atol=0)
    assert allclose(copy.outputs[1].data, array_in_2, rtol=0, atol=0)
    assert allclose(copy2.outputs[0].data, array_in_1b, rtol=0, atol=0)
    assert cache.tainted == False
    assert copy.tainted == False
    assert copy2.tainted == False

    savegraph(graph, f"output/{testname}.png")
