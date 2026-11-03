from matplotlib import pyplot as plt
from numpy import arange, array, array_equal, count_nonzero, diag, eye, linspace, load
from pytest import mark, raises
from scipy.sparse import csr_array

from dag_modelling.core.csr_array import CSRArray
from dag_modelling.core.data_descriptor import DataDescriptor
from dag_modelling.core.exception import TypeFunctionError
from dag_modelling.core.graph import Graph
from dag_modelling.lib.arithmetic import Sum
from dag_modelling.lib.common import Array, Copy, View
from dag_modelling.lib.hist import AxisDistortionMatrix
from dag_modelling.lib.linalg import MatrixProductDDt, VectorMatrixProduct
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.plot.plot import plot_auto
from dag_modelling.tools.save_matrices import save_matrices

SIZE = 6
MATRIX = diag(arange(1.0, SIZE + 1)) + diag(arange(1.0, SIZE), -1)


def test_DataDescriptor_layout():
    dd = DataDescriptor("d", (3, 5))
    assert dd.layout == "dense"
    assert dd.capacity is None
    dense = dd.allocate()
    assert dense.shape == (3, 5) and not dense.any()

    dd.layout, dd.capacity = "sparse-csr", 4
    sparse = dd.allocate()
    assert isinstance(sparse, CSRArray) and sparse.shape == (3, 5) and sparse.capacity == 4
    assert dd.consistent_with(sparse) and not dd.consistent_with(dense)
    assert "[sparse-csr]" in str(dd)
    dd.layout = "dense"
    assert dd.consistent_with(dense) and not dd.consistent_with(sparse)


def test_CSRArray():
    desired = array([[1.0, 0.0, 2.0], [0.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    reference = csr_array(desired)
    matrix = CSRArray.allocate(desired.shape, "d", 2)
    assert matrix.nnz == matrix.size == 0 and matrix.capacity == 2 and matrix.dtype == "d"
    assert matrix.nbytes == 2 * 8 + 2 * 4 + 4 * 4

    nnz = reference.nnz
    matrix.reserve(nnz)
    assert matrix.capacity == nnz
    matrix.data[:nnz] = reference.data
    matrix.indices[:nnz] = reference.indices
    matrix.indptr[:] = reference.indptr
    view = matrix.matrix
    assert matrix.nnz == matrix.size == view.nnz == nnz and view.data.size == 0
    matrix.update()
    assert view.data.size == view.indices.size == nnz
    assert array_equal(view.toarray(), desired) and array_equal(matrix.toarray(), desired)
    for array_ro, base in zip(
        (view.data, view.indices, view.indptr), (matrix.data, matrix.indices, matrix.indptr)
    ):
        assert array_ro.base is base and not array_ro.flags.writeable

    # the arrays grow keeping the entries, the view stays the same object
    matrix.reserve(10)
    assert matrix.capacity == 10 and matrix.matrix is view
    assert view.data.base is matrix.data and array_equal(view.toarray(), desired)


def connect_sum(matrix):
    matrix >> Sum("sum")


def connect_view(matrix):
    matrix >> View("view")


def connect_ddt(matrix):
    matrix >> MatrixProductDDt("DDt")


def connect_matrix_as_vector(matrix):
    # the CSR layout is allowed for the `matrix` input only
    product = VectorMatrixProduct("product", mode="column")
    Array("Dense", eye(SIZE)) >> product.inputs["matrix"]
    matrix >> product


@mark.parametrize("connect", (connect_sum, connect_view, connect_ddt, connect_matrix_as_vector))
def test_NodeStructure_layout(connect, debug_graph):
    with raises(TypeFunctionError, match="does not support the 'sparse-csr' layout"):
        with Graph(close_on_exit=True, debug=debug_graph):
            matrix = Array("Matrix", MATRIX, layout="sparse-csr")
            connect(matrix.outputs[0])


def test_layout_tools(test_name: str, output_path: str):
    with Graph(close_on_exit=True) as graph:
        edges = Array("Edges", linspace(0.0, 1.0, SIZE + 1))
        matrix = Array("Matrix", MATRIX, layout="sparse-csr", edges=(edges, edges))

    output = matrix.outputs[0]
    assert isinstance(output.data, csr_array) and array_equal(output.data.toarray(), MATRIX)
    assert output.dd.capacity == count_nonzero(MATRIX)
    plot_auto(output)
    plt.savefig(f"{output_path}/{test_name}.png")
    plt.close()
    savegraph(graph, f"{output_path}/{test_name}-graph.png", show="all")
    savegraph(graph, f"{output_path}/{test_name}-graph.dot", show="all")
    with open(f"{output_path}/{test_name}-graph.dot") as dot:
        text = dot.read()
    # the data is shown as a dense array
    assert "[sparse-csr]" in text and "[[1. 0. 0. 0. 0. 0.]" in text
    # loaded without pickle: the matrix is saved as a dense array, not as an object
    save_matrices({"matrix": output.data}, f"{output_path}/{test_name}.npz")
    with load(f"{output_path}/{test_name}.npz") as saved:
        assert array_equal(saved["matrix"], MATRIX)

    with Graph(close_on_exit=True):
        single = Array("Single", [[2.0]], layout="sparse-csr")
    assert single.outputs[0].to_dict()["value"] == 2.0


def test_layout_graph_failed_output(test_name: str, output_path: str):
    # a failed output is shown from its raw arrays, CSRArray must be expanded
    edges = linspace(0.0, 1.0, SIZE + 1)
    with Graph(close_on_exit=True) as graph:
        matrix = AxisDistortionMatrix("matrix")
        Array("Edges", edges) >> matrix.inputs["EdgesOriginal"]
        # the kernel fails on the target edges different from the original ones
        Array("Edges target", 2 * edges) >> matrix.inputs["EdgesTarget"]
        Array("Edges modified", edges) >> matrix.inputs["EdgesModified"]
        Array("Edges backward", edges) >> matrix.inputs["EdgesModifiedBackwards"]
    savegraph(graph, f"{output_path}/{test_name}.dot", show="all")
    with open(f"{output_path}/{test_name}.dot") as dot:
        assert "cought exception" in dot.read()


def test_data_dense_sparse():
    with Graph(close_on_exit=True):
        dense, sparse = Array("Dense", MATRIX), Array("Sparse", MATRIX, layout="sparse-csr")
        copy_dense, copy_sparse = Copy("Copy dense"), Copy("Copy sparse")
        dense >> copy_dense
        sparse >> copy_sparse

    output = dense.outputs[0]
    assert output.data_dense is output.data and output.data_sparse is None
    assert dense.get_data_dense() is output.data

    output = sparse.outputs[0]
    assert isinstance(output.data, csr_array) and output.data_sparse is output.data
    matrix = sparse.get_data_dense()
    assert array_equal(matrix, MATRIX) and not matrix.flags.writeable

    for node, source in ((copy_dense, dense), (copy_sparse, sparse)):
        input = node.inputs[0]
        assert node.get_input_data(0) is input.data is source.outputs[0].data
        assert input.data_sparse is source.outputs[0].data_sparse
        assert array_equal(input.data_dense, MATRIX)
