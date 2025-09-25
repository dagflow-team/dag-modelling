from numpy import allclose, arange, diag
from pytest import mark

from dag_modelling.core.graph import Graph
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.lib.common import Array
from dag_modelling.lib.linalg import MatrixProductAB


@mark.parametrize("dtype", ("d", "f"))
@mark.parametrize("diag_left", (False, True))
@mark.parametrize("diag_right", (False, True))
def test_MatrixProductAB(dtype: str, diag_left: bool, diag_right: bool, output_path: str):
    size = 3
    left = (in_left := arange(1, size * (size + 1) + 1, dtype=dtype).reshape(size + 1, size))
    right = (in_right := arange(1, size * (size + 1) + 1, dtype=dtype).reshape(size, size + 1))

    if diag_left:
        left = diag(in_left[:size, :size])
        in_left = diag(left)
    if diag_right:
        right = diag(in_right[:size, :size])
        in_right = diag(right)

    with Graph(close_on_exit=True) as graph:
        array_left = Array("Left", left, mode="fill")
        array_right = Array("Right", right, mode="fill")

        prod = MatrixProductAB("MatrixProductAB")
        array_left >> prod.inputs["left"]
        array_right >> prod.inputs["right"]

    desired = in_left @ in_right
    if diag_left and diag_right:
        desired = diag(desired)

    actual = prod.get_data()
    assert allclose(desired, actual, atol=0, rtol=0)
    assert (diag_left and diag_right) == (len(actual.shape) == 1)

    sleft = diag_left and "diag" or "block"
    sright = diag_right and "diag" or "block"
    ograph = f"{output_path}/test_MatrixProductAB_{dtype}_{sleft}_{sright}.png"
    print(f"Write graph: {ograph}")
    savegraph(graph, ograph)
