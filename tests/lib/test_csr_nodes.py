from logging import WARNING

from numpy import allclose, arange, array, array_equal, diag, full, linspace, ones, sin, zeros
from numpy.random import default_rng
from pytest import mark, raises
from scipy.sparse import csr_array

from dag_modelling.core.exception import InitializationError, TypeFunctionError
from dag_modelling.core.graph import Graph
from dag_modelling.lib.common import Array, Cache, Copy
from dag_modelling.lib.hist import (
    AxisDistortionMatrix,
    AxisDistortionMatrixLinear,
    AxisDistortionMatrixPointwise,
    HistSmearNormalMatrixBC,
    HistSmearNormalMatrixIntegral,
    RebinMatrix,
)
from dag_modelling.lib.hist.axis_distortion_matrix import _axisdistortion_numba
from dag_modelling.lib.hist.axis_distortion_matrix_linear import _axisdistortion_linear_numba
from dag_modelling.lib.hist.axis_distortion_matrix_pointwise import (
    _axisdistortion_pointwise_numba,
)
from dag_modelling.lib.linalg import VectorMatrixProduct
from dag_modelling.lib.normalization import NormalizeMatrix, RenormalizeDiag
from dag_modelling.lib.physics import EnergyResolution
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.tools.profiling import MemoryProfiler


def check_csr(output, dense):
    """Check the CSR `output` against a dense matrix"""
    matrix = output.data
    assert output.dd.layout == "sparse-csr" and isinstance(matrix, csr_array)
    # a forgotten update() leaves the view shorter than nnz
    assert matrix.data.size == matrix.indices.size == matrix.nnz
    assert not matrix.data.flags.writeable and not matrix.indptr.flags.writeable
    # the producers do not store zeros
    assert matrix.data.all()
    # scipy keeps the canonical flag of a long-lived object, check a new one
    fresh = csr_array((matrix.data, matrix.indices, matrix.indptr), shape=matrix.shape)
    fresh.check_format(full_check=True)
    assert fresh.has_canonical_format
    assert matrix.toarray().tobytes() == dense.tobytes()


def check_products(products: dict):
    (dense, dense_product), (csr, csr_product) = products["dense"], products["sparse-csr"]
    check_csr(csr.outputs["matrix"], dense.outputs["matrix"].data)
    assert allclose(csr_product.get_data(), dense_product.get_data(), atol=0, rtol=1e-14)


# scale 0.9 with 240 bins makes the traversal revisit cells due to rounding
@mark.parametrize("mode", ("column", "row"))
@mark.parametrize("scale", (0.9, 0.97, 1.0, 1.03))
@mark.parametrize("nbins", (1, 12, 240))
@mark.parametrize("cls", (AxisDistortionMatrix, AxisDistortionMatrixLinear))
def test_AxisDistortionMatrix_csr(
    cls: type, nbins: int, scale: float, mode: str, test_name: str, debug_graph, output_path: str
):
    edges = linspace(0.0, 10.0, nbins + 1)
    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        array_edges = Array("Edges", edges, mode="fill")
        array_modified = Array("Edges modified", edges * scale, mode="fill")
        array_backward = Array("Edges backward", edges / scale, mode="fill")
        array_vector = Array("Vector", arange(1.0, nbins + 1), mode="fill")
        products = {}
        for layout in ("dense", "sparse-csr"):
            matrix = cls(f"matrix {layout}", layout=layout)
            array_edges >> matrix.inputs["EdgesOriginal"]
            array_edges >> matrix.inputs["EdgesTarget"]
            array_modified >> matrix.inputs["EdgesModified"]
            if cls is AxisDistortionMatrix:
                array_backward >> matrix.inputs["EdgesModifiedBackwards"]
            product = VectorMatrixProduct(f"product {layout}", mode=mode)
            matrix.outputs["matrix"] >> product.inputs["matrix"]
            array_vector >> product
            products[layout] = matrix, product

    (dense, dense_product), (csr, csr_product) = products["dense"], products["sparse-csr"]
    output = csr.outputs["matrix"]
    assert output.dd.shape == dense.outputs["matrix"].dd.shape == (nbins, nbins)
    assert output.dd.axes_edges == dense.outputs["matrix"].dd.axes_edges
    assert csr_product.outputs[0].dd.axes_edges == dense_product.outputs[0].dd.axes_edges
    check_products(products)

    # the last shift moves all the edges out of range: the kernel returns early
    data = output._data.data
    for scale_new, shift in ((1.0, 0.0), (1.02, 0.0), (0.95, 0.0), (1.0, 20.0)):
        array_modified.outputs[0].set(edges * scale_new + shift, force_taint=True)
        array_backward.outputs[0].set((edges - shift) / scale_new, force_taint=True)
        check_products(products)
    assert output._data.data is data
    assert output.data.nnz == 0

    savegraph(graph, f"{output_path}/{test_name}.png")


def pointwise_curve(edges, kind: str):
    x = linspace(edges[0] - 0.5, edges[-1] + 0.5, 3 * edges.size)
    match kind:
        case "scaled":
            return x, 1.03 * x
        case "decreasing":
            return x, x[::-1].copy()
        case "wavy":
            # goes up and down within the columns: the rows go back
            return x, x + 0.6 * sin(3.0 * x)


def band(size: int):
    return (
        diag(arange(1.0, size + 1)) + diag(full(size - 1, 0.5), -1) + diag(full(size - 1, 0.25), 1)
    )


@mark.parametrize("kind", ("scaled", "decreasing", "wavy"))
@mark.parametrize("nbins", (1, 12, 240))
def test_AxisDistortionMatrixPointwise_csr(nbins: int, kind: str):
    edges = linspace(0.0, 10.0, nbins + 1)
    x, y = pointwise_curve(edges, kind)
    with Graph(close_on_exit=True):
        array_edges = Array("Edges", edges, mode="fill")
        array_x = Array("X", x, mode="fill")
        array_y = Array("Y", y, mode="fill")
        matrices = {}
        for layout in ("dense", "sparse-csr"):
            matrix = AxisDistortionMatrixPointwise(f"matrix {layout}", layout=layout)
            array_edges >> matrix.inputs["EdgesOriginal"]
            array_edges >> matrix.inputs["EdgesTarget"]
            array_x >> matrix.inputs["DistortionOriginal"]
            array_y >> matrix.inputs["DistortionTarget"]
            matrices[layout] = matrix.outputs["matrix"]

    dense, csr = matrices["dense"], matrices["sparse-csr"]
    capacity = csr.dd.capacity
    assert capacity == 2 * nbins - 1
    check_csr(csr, dense.data)
    # a monotonic curve fits, the wavy one makes the arrays grow
    if kind != "wavy":
        assert csr._data.capacity == capacity
    elif nbins == 240:
        assert csr._data.capacity > capacity
    assert MemoryProfiler.estimate_node(csr.node)[2]["coo"][1] >= capacity

    for scale in (0.97, 1.05):
        array_y.outputs[0].set(y * scale, force_taint=True)
        check_csr(csr, dense.data)


@mark.parametrize("cls", (HistSmearNormalMatrixBC, HistSmearNormalMatrixIntegral))
def test_HistSmearNormalMatrix_csr(cls: type):
    edges = linspace(1.0, 11.0, 41)
    with Graph(close_on_exit=True):
        array_edges = Array("Edges", edges, mode="fill")
        sigma = Array("RelSigma", full(edges.size - 1, 0.20), mode="fill")
        matrices = {}
        for layout in ("dense", "sparse-csr"):
            matrix = cls(f"matrix {layout}", layout=layout)
            sigma >> matrix.inputs["RelSigma"]
            array_edges >> matrix.inputs["Edges"]
            array_edges >> matrix.inputs["EdgesOut"]
            matrices[layout] = matrix.outputs[0]
    with Graph(close_on_exit=True):
        array_edges = Array("Edges", edges, mode="fill")
        sigma_narrow = Array("RelSigma", full(edges.size - 1, 0.02), mode="fill")
        fresh = cls("fresh")
        sigma_narrow >> fresh.inputs["RelSigma"]
        array_edges >> fresh.inputs["Edges"]
        array_edges >> fresh.inputs["EdgesOut"]

    dense, csr = matrices["dense"], matrices["sparse-csr"]
    assert dense.dd.layout == "dense" and csr.dd.capacity == edges.size - 1
    check_csr(csr, dense.data)
    internals = MemoryProfiler.estimate_node(csr.node)[2]
    assert internals["dense"] == (dense.data.nbytes, dense.data.size)

    # a narrower resolution leaves no old values in the cells below the threshold
    sigma.outputs[0].set(full(edges.size - 1, 0.02))
    assert dense.data.tobytes() == fresh.outputs[0].data.tobytes()
    check_csr(csr, dense.data)


def test_RebinMatrix_edges_change():
    edges_old = linspace(0.0, 2.0, 21)
    edges_new = edges_old[::2]
    edges_new_other = edges_old[[0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 20]]
    matrices = {}
    with Graph(close_on_exit=True):
        array_old = Array("Edges old", edges_old, mode="fill")
        array_new = Array("Edges new", edges_new, mode="fill")
        for layout in ("dense", "sparse-csr"):
            matrix = RebinMatrix(f"matrix {layout}", layout=layout)
            array_old >> matrix.inputs["edges_old"]
            array_new >> matrix.inputs["edges_new"]
            matrices[layout] = matrix.outputs["matrix"]
    dense, csr = matrices["dense"], matrices["sparse-csr"]
    check_csr(csr, dense.data)
    assert MemoryProfiler.estimate_node(csr.node)[2]["coo"][1] == edges_old.size - 1

    # the dense matrix is reset: no ones stay from the old edges
    array_new.outputs[0].set(edges_new_other)
    assert (dense.data.sum(axis=0) == 1).all()
    check_csr(csr, dense.data)


@mark.parametrize("mode", ("diag", "offdiag"))
def test_RenormalizeDiag_csr(mode: str):
    edges = linspace(0.0, 10.0, 25)
    x, y = pointwise_curve(edges, "scaled")
    renormalized = {}
    with Graph(close_on_exit=True):
        array_edges = Array("Edges", edges, mode="fill")
        array_x = Array("X", x, mode="fill")
        array_y = Array("Y", y, mode="fill")
        scale = Array("scale", [1.3], mode="fill")
        static = Array("Static", band(edges.size - 1), mode="fill")
        static_csr = Array("Static CSR", band(edges.size - 1), layout="sparse-csr")
        for layout in ("dense", "sparse-csr"):
            matrix = AxisDistortionMatrixPointwise(f"matrix {layout}", layout=layout)
            array_edges >> matrix.inputs["EdgesOriginal"]
            array_edges >> matrix.inputs["EdgesTarget"]
            array_x >> matrix.inputs["DistortionOriginal"]
            array_y >> matrix.inputs["DistortionTarget"]
            node = RenormalizeDiag(f"renormalize {layout}", mode=mode, ndiag=2)
            matrix >> node
            (static_csr if layout == "sparse-csr" else static) >> node
            scale >> node("scale")
            renormalized[layout] = node

    dense, csr = renormalized["dense"], renormalized["sparse-csr"]
    assert [input.name for input in csr.inputs] == ["matrix", "matrix_01"]
    for output, dense_output in zip(csr.outputs, dense.outputs):
        check_csr(output, dense_output.data)
    assert MemoryProfiler.estimate_node(csr)[2]["colsum"][1] == edges.size - 1

    # the input grows for a curve that goes back, the output follows
    array_y.outputs[0].set(x + 0.6 * sin(3.0 * x))
    for output, dense_output in zip(csr.outputs, dense.outputs):
        check_csr(output, dense_output.data)
    assert csr.outputs[0]._data.capacity > csr.outputs[0].dd.capacity


def test_RenormalizeDiag_csr_zero_scale():
    # with scale 0 the first column sums to zero and stays zero, as in dense
    matrix = diag(arange(1.0, 5.0)) + diag(ones(3), 1)
    renormalized = {}
    with Graph(close_on_exit=True):
        scale = Array("scale", [0.0])
        for layout in ("dense", "sparse-csr"):
            node = RenormalizeDiag(f"renormalize {layout}")
            Array(f"Matrix {layout}", matrix, layout=layout) >> node
            scale >> node("scale")
            renormalized[layout] = node.outputs[0]
    dense, csr = renormalized["dense"].data, renormalized["sparse-csr"].data
    assert csr.toarray().tobytes() == dense.tobytes()


def test_RenormalizeDiag_csr_wrong_inputs():
    # `AllPositionals` does not cover the keyword input
    with raises(TypeFunctionError, match="does not support the 'sparse-csr' layout"):
        with Graph(close_on_exit=True):
            node = RenormalizeDiag("renormalize")
            Array("Matrix", band(3)) >> node
            Array("Scale", [[2.0]], layout="sparse-csr") >> node("scale")
    with raises(TypeFunctionError, match="the same layout"):
        with Graph(close_on_exit=True):
            node = RenormalizeDiag("renormalize")
            Array("Dense", band(3)) >> node
            Array("CSR", band(3), layout="sparse-csr") >> node
            Array("Scale", [1.1]) >> node("scale")


@mark.parametrize("mode", ("column", "row"))
@mark.parametrize("scipy_input", (False, True))
def test_Array_csr(scipy_input: bool, mode: str):
    rng = default_rng(1)
    desired = rng.uniform(size=(7, 9))
    desired[desired < 0.7] = 0.0
    source = csr_array(desired) if scipy_input else desired
    vector = arange(1.0, desired.shape[1 if mode == "column" else 0] + 1)

    with Graph(close_on_exit=True):
        matrix = Array("Matrix", source, layout="sparse-csr")
        product = VectorMatrixProduct("product", mode=mode)
        matrix >> product.inputs["matrix"]
        Array("Vector", vector, mode="fill") >> product

    if scipy_input:
        # the node keeps its own arrays
        for array_source in (source.data, source.indices, source.indptr):
            array_source[:] = 0

    output = matrix.outputs[0]
    assert output.dd.layout == "sparse-csr" and output.dd.shape == desired.shape
    check_csr(output, desired)
    product_desired = desired @ vector if mode == "column" else vector @ desired
    assert allclose(product.get_data(), product_desired, atol=0, rtol=1e-14)


def test_Array_csr_repeated_cells():
    # a repeated cell (0, 2), unsorted columns and a stored zero in the second row
    data, indices = array([1.0, 1.0, 3.0, 4.0, 0.0]), array([2, 2, 1, 0, 2])
    indptr = array([0, 2, 5])
    source = csr_array((data, indices, indptr), shape=(2, 3))
    desired = array([[0.0, 0.0, 2.0], [4.0, 3.0, 0.0]])
    with Graph(close_on_exit=True):
        matrix = Array("Matrix", source, layout="sparse-csr")
        dense = Array("Dense", source)
    check_csr(matrix.outputs[0], desired)
    # the default layout expands a scipy matrix
    assert dense.outputs[0].dd.layout == "dense" and array_equal(dense.outputs[0].data, desired)


@mark.parametrize("mode", ("columns", "rows"))
def test_NormalizeMatrix_csr(mode: str):
    edges = linspace(0.0, 10.0, 25)
    x, y = pointwise_curve(edges, "scaled")
    normalized = {}
    with Graph(close_on_exit=True):
        array_edges = Array("Edges", edges, mode="fill")
        array_x = Array("X", x, mode="fill")
        array_y = Array("Y", y, mode="fill")
        for layout in ("dense", "sparse-csr"):
            matrix = AxisDistortionMatrixPointwise(f"matrix {layout}", layout=layout)
            array_edges >> matrix.inputs["EdgesOriginal"]
            array_edges >> matrix.inputs["EdgesTarget"]
            array_x >> matrix.inputs["DistortionOriginal"]
            array_y >> matrix.inputs["DistortionTarget"]
            node = NormalizeMatrix(f"normalize {layout}", mode=mode)
            matrix >> node
            normalized[layout] = node.outputs[0]
    dense, csr = normalized["dense"], normalized["sparse-csr"]
    check_csr(csr, dense.data)

    # the input changes and grows for a curve that goes back
    array_y.outputs[0].set(x + 0.6 * sin(3.0 * x))
    check_csr(csr, dense.data)
    assert csr._data.capacity > csr.dd.capacity


@mark.parametrize("mode", ("columns", "rows"))
def test_NormalizeMatrix_csr_zero_sum(mode: str):
    # the first row and column sum to zero: their cells become zeros, as in dense
    matrix = array([[1.0, -1.0], [-1.0, 2.0]])
    normalized = {}
    with Graph(close_on_exit=True):
        for layout in ("dense", "sparse-csr"):
            node = NormalizeMatrix(f"normalize {layout}", mode=mode)
            Array(f"Matrix {layout}", matrix, layout=layout) >> node
            normalized[layout] = node.outputs[0]
    dense, csr = normalized["dense"].data, normalized["sparse-csr"].data
    assert csr.toarray().tobytes() == dense.tobytes()


@mark.parametrize("cls", (Copy, Cache))
def test_Copy_layout(cls: type):
    matrix = band(4)
    with Graph(close_on_exit=True):
        source = Array("Matrix", matrix, mode="fill")
        to_csr = cls("to csr", layout="sparse-csr")
        keep = cls("keep")
        to_dense = cls("to dense", layout="dense")
        source >> to_csr
        to_csr >> keep
        keep >> to_dense

    assert [node.outputs[0].dd.layout for node in (to_csr, keep, to_dense)] == [
        "sparse-csr",
        "sparse-csr",
        "dense",
    ]
    for node in (to_csr, keep):
        check_csr(node.outputs[0], matrix)
    assert to_dense.outputs[0].data.tobytes() == matrix.tobytes()
    # the arrays start with room for an entry per row and grow
    assert to_csr.outputs[0]._data.capacity > to_csr.outputs[0].dd.capacity == matrix.shape[0]

    # a cache keeps the first values
    source.outputs[0].set(2.0 * matrix)
    desired = matrix if cls is Cache else 2.0 * matrix
    for node in (to_csr, keep):
        check_csr(node.outputs[0], desired)
    assert to_dense.outputs[0].data.tobytes() == desired.tobytes()


def test_Copy_layout_wrong():
    with raises(InitializationError, match="layout must be"):
        Copy("copy", layout="CSR")
    with raises(TypeFunctionError, match="supports only 2d inputs"):
        with Graph(close_on_exit=True):
            Array("Vector", ones(3)) >> Copy("copy", layout="sparse-csr")
    with raises(TypeFunctionError, match="the same layout"):
        with Graph(close_on_exit=True):
            copy = Copy("copy")
            Array("Dense", band(3)) >> copy
            Array("CSR", band(3), layout="sparse-csr") >> copy


def test_distortion_kernels_need_output():
    edges = linspace(0.0, 10.0, 5)
    with raises(ValueError, match="the CSR arrays must be given"):
        _axisdistortion_numba(edges, edges, edges, edges, None)
    with raises(ValueError, match="the CSR arrays must be given"):
        _axisdistortion_linear_numba(edges, edges, edges, None)
    with raises(ValueError, match="the COO arrays must be given"):
        _axisdistortion_pointwise_numba(edges, edges, edges, edges, None)


def test_VectorMatrixProduct_row_csr_warning(caplog):
    with caplog.at_level(WARNING, logger="dag_modelling"), Graph(close_on_exit=True):
        product = VectorMatrixProduct("product", mode="row")
        Array("Matrix", band(3), layout="sparse-csr") >> product.inputs["matrix"]
        Array("Vector", ones(3), mode="fill") >> product
    assert "CSC" in caplog.text


def test_EnergyResolution_layout():
    eres = EnergyResolution(layout="sparse-csr")
    assert eres._energy_resolution_matrix_bc_list[0]._layout == "sparse-csr"
    _, storage = EnergyResolution.replicate(path="eres", layout="sparse-csr")
    assert storage["nodes.eres.matrix"]._layout == "sparse-csr"


def test_csr_layouts():
    for cls in (RebinMatrix, AxisDistortionMatrix, AxisDistortionMatrixLinear):
        assert cls("matrix")._layout == "sparse-csr"
    assert AxisDistortionMatrixPointwise("matrix")._layout == "sparse-csr"
    for cls in (HistSmearNormalMatrixBC, HistSmearNormalMatrixIntegral):
        assert cls("matrix")._layout == "dense"
    for cls in (
        AxisDistortionMatrix,
        AxisDistortionMatrixLinear,
        AxisDistortionMatrixPointwise,
        HistSmearNormalMatrixBC,
        HistSmearNormalMatrixIntegral,
        RebinMatrix,
    ):
        with raises(InitializationError, match="layout must be"):
            cls("matrix", layout="CSR")
    with raises(InitializationError, match="layout must be"):
        Array("Matrix", zeros((2, 2)), layout="CSR")
    with raises(InitializationError, match="needs a matrix and mode 'store'"):
        Array("Matrix", zeros((2, 2)), layout="sparse-csr", mode="fill")
    with raises(InitializationError, match="needs a matrix and mode 'store'"):
        Array("Vector", zeros(2), layout="sparse-csr")
