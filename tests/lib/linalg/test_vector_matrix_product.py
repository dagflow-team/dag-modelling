from numpy import allclose, arange, diag
from pytest import mark

from dag_modelling.core.graph import Graph
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.lib.common import Array
from dag_modelling.lib.linalg import VectorMatrixProduct


@mark.parametrize("dtype", ("d", "f"))
@mark.parametrize("diag_matrix", (False, True))
@mark.parametrize("mode", ("row", "column"))
def test_VectorMatrixProduct(dtype: str, diag_matrix: bool, mode: str, output_path: str):
    size = 4
    is_column = mode == "column"

    matrix = (in_matrix := arange(1, size * (size + 1) + 1, dtype=dtype).reshape(size, size + 1))
    edgesX = arange(0, size + 1)
    edgesY = arange(0, size + 2)

    if diag_matrix:
        matrix = diag(in_matrix[:size, :size])
        in_matrix = diag(matrix)

    if is_column:
        vector = arange(1, matrix.shape[-1] + 1, dtype=dtype)
        column = vector[:, None]
        desired = (in_matrix @ column).ravel()
    else:
        vector = arange(1, matrix.shape[0] + 1, dtype=dtype)
        row = vector[None, :]
        desired = (row @ in_matrix).ravel()

    with Graph(close_on_exit=True) as graph:
        array_vector = Array("Vector", vector, mode="fill")
        array_edgesX = Array("edgesX", edgesX, mode="fill")
        array_edgesY = Array("edgesY", edgesY, mode="fill")
        edges = (
            (array_edgesX.outputs["array"], array_edgesY.outputs["array"])
            if matrix.ndim == 2
            else (array_edgesX.outputs["array"],)
        )
        array_matrix = Array("Matrix", matrix, edges=edges, mode="fill")

        prod = VectorMatrixProduct("VectorMatrixProduct", mode=mode)
        array_matrix >> prod.inputs["matrix"]
        array_vector >> prod

    actual = prod.get_data()
    assert allclose(desired, actual, atol=0, rtol=0)
    assert len(actual.shape) == 1

    right_edges = edges[0] if matrix.ndim == 1 or is_column else edges[1]
    assert prod.outputs[0].dd.axes_edges[0] == right_edges

    smatrix = diag_matrix and "diag" or "block"
    ograph = f"{output_path}/test_VectorMatrixProduct_{dtype}_{smatrix}_{mode}.png"
    print(f"Write graph: {ograph}")
    savegraph(graph, ograph)
