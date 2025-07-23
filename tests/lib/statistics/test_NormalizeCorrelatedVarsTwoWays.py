from numpy import allclose, arange, array, finfo, full_like, ones_like, sqrt, zeros_like
from pytest import mark, raises
from scipy.linalg import cholesky, solve_triangular

from dag_modelling.core.exception import TypeFunctionError
from dag_modelling.core.graph import Graph
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.lib.common import Array
from dag_modelling.lib.linalg import Cholesky
from dag_modelling.lib.statistics import NormalizeCorrelatedVarsTwoWays

debug = False


@mark.parametrize("dtype", ("d", "f"))
def test_NormalizeCorrelatedVarsTwoWays_00(dtype):
    fp_tolerance = finfo(dtype).resolution * 2

    inCentral = arange(3.0, dtype=dtype) * 100.0
    inV = array([[10, 2, 1], [2, 12, 3], [1, 3, 13]], dtype=dtype)
    inD = inV.diagonal()
    inL = cholesky(inV, lower=True)
    inLd = sqrt(inD)
    inOffset = array((-10.0, 20.0, 30.0), dtype=dtype)
    inVec = inCentral + inOffset
    inNorm = full_like(inVec, -100)
    with Graph(close_on_exit=True) as graph:
        var_matrix = Array("var_matrix", inV)
        var_diag = Array("var_diag", inD)
        Lmatrix = Cholesky("cholesky 1d")
        Ldiag = Cholesky("cholesky 2d")
        central = Array("central", inCentral)
        value1d = Array("vec 1d", inVec, mode="store_weak")
        normvalue1d = Array("normvalue 1d", inNorm, mode="store_weak")
        value2d = Array("vec 2d", inVec, mode="store_weak")
        normvalue2d = Array("normvalue 2d", inNorm, mode="store_weak")
        norm1d = NormalizeCorrelatedVarsTwoWays("norm1d")
        norm2d = NormalizeCorrelatedVarsTwoWays("norm2d")

        central >> norm1d.inputs["central"]
        central >> norm2d.inputs["central"]

        var_matrix >> Lmatrix
        var_diag >> Ldiag

        Lmatrix >> norm2d.inputs["matrix"]
        Ldiag >> norm1d.inputs["matrix"]

        (value1d, normvalue1d) >> norm1d
        (value2d, normvalue2d) >> norm2d

    nodes = (
        var_matrix,
        var_diag,
        Lmatrix,
        Ldiag,
        central,
        value1d,
        normvalue1d,
        value1d,
        normvalue2d,
        norm1d,
        norm2d,
    )

    assert all(node.tainted == True for node in nodes)
    norm_matrix = norm2d.get_data(1)
    norm_diag = norm1d.get_data(1)
    back_matrix = norm2d.get_data(0)
    back_diag = norm1d.get_data(0)

    assert all(node.tainted == False for node in nodes)
    assert all(inNorm != back_matrix)
    assert all(inNorm != back_diag)

    norm1 = solve_triangular(inL, inOffset, lower=True)
    norm2 = inOffset / inLd

    if debug:
        print("V:", inV)
        print("Vdiag:", inD)
        print("L:", inL)
        print("Ldiag:", inLd)
        print("Central:", inCentral)
        print("In:", inVec)
        print("Offset:", inOffset)
        print("Norm 1:", norm1)
        print("Norm 2:", norm2)
        print("Rec 1:", back_matrix)
        print("Rec 2:", back_diag)
        print("Diff 1:", inVec - back_matrix)
        print("Diff 2:", inVec - back_diag)

    atol = finfo(dtype).resolution
    assert allclose(norm1, norm_matrix, atol=atol, rtol=0)
    assert allclose(norm2, norm_diag, atol=0, rtol=0)
    assert allclose(inVec, back_matrix, atol=0, rtol=0)
    assert allclose(inVec, back_diag, atol=0, rtol=0)

    inZeros = zeros_like(inVec)
    inOnes = ones_like(inVec)

    #
    # Set norm value
    #
    normvalue1d.set(inZeros)
    normvalue2d.set(inZeros)
    back_matrix = norm2d.get_data(0)
    back_diag = norm1d.get_data(0)
    norm_matrix = norm2d.get_data(1)
    norm_diag = norm1d.get_data(1)
    assert allclose(inZeros, norm_matrix, atol=0, rtol=0)
    assert allclose(inZeros, norm_diag, atol=0, rtol=0)
    assert allclose(inCentral, back_matrix, atol=0, rtol=0)
    assert allclose(inCentral, back_diag, atol=0, rtol=0)

    #
    # Set normvalue
    #
    normvalue1d.set(inOnes)
    normvalue2d.set(inOnes)
    back_matrix = norm2d.get_data(0)
    back_diag = norm1d.get_data(0)
    norm_matrix = norm2d.get_data(1)
    norm_diag = norm1d.get_data(1)
    checkDiagOnes = inCentral + inLd
    checkMatrixOnes = inCentral + inL @ inOnes
    assert allclose(inOnes, norm_matrix, atol=0, rtol=0)
    assert allclose(inOnes, norm_diag, atol=0, rtol=0)
    assert allclose(checkMatrixOnes, back_matrix, atol=0, rtol=0)
    assert allclose(checkDiagOnes, back_diag, atol=0, rtol=0)

    #
    # Set value and norm value
    #
    value1d.set(inCentral)
    value2d.set(inCentral)
    normvalue1d.set(inOnes)
    normvalue2d.set(inOnes)
    back_matrix = norm2d.get_data(0)
    back_diag = norm1d.get_data(0)
    norm_matrix = norm2d.get_data(1)
    norm_diag = norm1d.get_data(1)
    assert allclose(inOnes, norm_matrix, atol=0, rtol=0)
    assert allclose(inOnes, norm_diag, atol=0, rtol=0)
    assert not allclose(inCentral, back_matrix, atol=0, rtol=0)
    assert not allclose(inCentral, back_diag, atol=0, rtol=0)

    #
    # Set value and norm value
    #
    normvalue1d.set(inOnes)
    normvalue2d.set(inOnes)
    value1d.set(inCentral)
    value2d.set(inCentral)
    back_matrix = norm2d.get_data(0)
    back_diag = norm1d.get_data(0)
    norm_matrix = norm2d.get_data(1)
    norm_diag = norm1d.get_data(1)
    assert not allclose(inOnes, norm_matrix, atol=0, rtol=0)
    assert not allclose(inOnes, norm_diag, atol=0, rtol=0)
    assert allclose(inCentral, back_matrix, atol=0, rtol=0)
    assert allclose(inCentral, back_diag, atol=0, rtol=0)

    #
    # Set value (with immediate flag)
    #
    norm2d._immediate = True
    norm1d._immediate = True
    value1d.set(inCentral)
    value2d.set(inCentral)
    norm_matrix = norm2d.outputs[1]._data
    norm_diag = norm1d.outputs[1]._data
    back_matrix = norm2d.outputs[0]._data
    back_diag = norm1d.outputs[0]._data
    assert allclose(inZeros, norm_matrix, atol=0, rtol=0)
    assert allclose(inZeros, norm_diag, atol=0, rtol=0)
    assert allclose(inCentral, back_matrix, atol=0, rtol=0)
    assert allclose(inCentral, back_diag, atol=0, rtol=0)

    #
    # Set value (with immediate flag)
    #
    norm2d._immediate = True
    norm1d._immediate = True
    normvalue1d.set(inOnes)
    normvalue2d.set(inOnes)
    norm_matrix = norm2d.outputs[1]._data
    norm_diag = norm1d.outputs[1]._data
    back_matrix = norm2d.outputs[0]._data
    back_diag = norm1d.outputs[0]._data
    assert allclose(inOnes, norm_matrix, atol=0, rtol=0)
    assert allclose(inOnes, norm_diag, atol=0, rtol=0)
    assert allclose(checkMatrixOnes, back_matrix, atol=0, rtol=0)
    assert allclose(checkDiagOnes, back_diag, atol=0, rtol=0)

    #
    # Set central
    #
    norm2d._immediate = False
    norm1d._immediate = False
    central.set(-inOnes)
    back_matrix = norm2d.get_data(0)
    back_diag = norm1d.get_data(0)
    norm_matrix = norm2d.get_data(1)
    norm_diag = norm1d.get_data(1)
    assert all(norm_matrix != inOnes)
    assert all(norm_matrix != inOnes)
    assert allclose(checkMatrixOnes, back_matrix, atol=0, rtol=0)
    assert allclose(checkDiagOnes, back_diag, atol=0, rtol=0)

    #
    # Revert central
    #
    central.set(inCentral)
    back_matrix = norm2d.get_data(0)
    back_diag = norm1d.get_data(0)
    norm_matrix = norm2d.get_data(1)
    norm_diag = norm1d.get_data(1)
    assert allclose(inOnes, norm_matrix, atol=fp_tolerance, rtol=0)
    assert allclose(inOnes, norm_diag, atol=fp_tolerance, rtol=0)
    assert allclose(checkMatrixOnes, back_matrix, atol=0, rtol=0)
    assert allclose(checkDiagOnes, back_diag, atol=0, rtol=0)

    #
    # Set sigma
    #
    var_matrix.set(inV * 2)
    var_diag.set(inD * 2)
    back_matrix = norm2d.get_data(0)
    back_diag = norm1d.get_data(0)
    norm_matrix = norm2d.get_data(1)
    norm_diag = norm1d.get_data(1)
    assert all(norm_matrix != inOnes)
    assert all(norm_matrix != inOnes)
    assert allclose(checkMatrixOnes, back_matrix, atol=0, rtol=0)
    assert allclose(checkDiagOnes, back_diag, atol=0, rtol=0)

    #
    # Revert sigma
    #
    var_matrix.set(inV)
    var_diag.set(inD)
    back_matrix = norm2d.get_data(0)
    back_diag = norm1d.get_data(0)
    norm_matrix = norm2d.get_data(1)
    norm_diag = norm1d.get_data(1)
    assert allclose(inOnes, norm_matrix, atol=fp_tolerance, rtol=0)
    assert allclose(inOnes, norm_diag, atol=fp_tolerance, rtol=0)
    assert allclose(checkMatrixOnes, back_matrix, atol=0, rtol=0)
    assert allclose(checkDiagOnes, back_diag, atol=0, rtol=0)

    savegraph(graph, f"output/test_NormalizeCorrelatedVarsTwoWays_00_{dtype}.png", show=["all"])


def test_NormalizeCorrelatedVarsTwoWays_01(dtype="d"):
    inVec = arange(4.0, dtype=dtype) * 100.0
    inNorm = full_like(inVec, -100)
    inV = array([[10, 2, 1], [2, 12, 3], [1, 3, 13]], dtype=dtype)
    inD = inV.diagonal()
    with Graph() as graph1:
        var_diag = Array("var_diag", inD)
        vec = Array("vec", inVec, mode="store_weak")
        nvec = Array("vec", inNorm, mode="store_weak")
        norm1d = NormalizeCorrelatedVarsTwoWays("norm1d")

        vec >> norm1d.inputs["central"]
        var_diag >> norm1d.inputs["matrix"]

        (vec, nvec) >> norm1d

    with Graph() as graph2:
        var_matrix = Array("var_matrix", inV)
        vec = Array("vec", inVec, mode="store_weak")
        nvec = Array("vec", inNorm, mode="store_weak")
        norm2d = NormalizeCorrelatedVarsTwoWays("norm2d")

        vec >> norm2d.inputs["central"]
        var_matrix >> norm2d.inputs["matrix"]

        (vec, nvec) >> norm2d

    with raises(TypeFunctionError):
        graph1.close()

    with raises(TypeFunctionError):
        graph2.close()
