from numpy import allclose, arange, array, finfo, sqrt
from pytest import mark, raises
from scipy.linalg import cholesky, solve_triangular

from dag_modelling.core.exception import TypeFunctionError
from dag_modelling.core.graph import Graph
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.lib.common import Array
from dag_modelling.lib.linalg import Cholesky
from dag_modelling.lib.statistics import NormalizeCorrelatedVars

debug = False


@mark.parametrize("dtype", ("d", "f"))
def test_NormalizeCorrelatedVars_00(dtype, output_path: str):
    inCentral = arange(3.0, dtype=dtype) * 100.0
    inV = array([[10, 2, 1], [2, 12, 3], [1, 3, 13]], dtype=dtype)
    inD = inV.diagonal()
    inL = cholesky(inV, lower=True)
    inLd = sqrt(inD)
    inOffset = array((-10.0, 20.0, 30.0), dtype=dtype)
    inVec = inCentral + inOffset
    with Graph(close_on_exit=True) as graph:
        matrix = Array("matrix", inV, mode="fill")
        diag = Array("diag", inD, mode="fill")
        Lmatrix = Cholesky("cholesky 1d")
        Ldiag = Cholesky("cholesky 2d")
        central = Array("central", inCentral, mode="fill")
        vec = Array("vec", inVec, mode="fill")
        norm1d_fwd = NormalizeCorrelatedVars("norm1d fwd")
        norm2d_fwd = NormalizeCorrelatedVars("norm2d fwd")

        norm1d_bwd = NormalizeCorrelatedVars("norm1d bwd", mode="backward")
        norm2d_bwd = NormalizeCorrelatedVars("norm2d bwd", mode="backward")

        central >> norm1d_fwd.inputs["central"]
        central >> norm2d_fwd.inputs["central"]
        central >> norm1d_bwd.inputs["central"]
        central >> norm2d_bwd.inputs["central"]

        matrix >> Lmatrix
        diag >> Ldiag

        Lmatrix >> norm2d_fwd.inputs["matrix"]
        Ldiag >> norm1d_fwd.inputs["matrix"]
        Lmatrix >> norm2d_bwd.inputs["matrix"]
        Ldiag >> norm1d_bwd.inputs["matrix"]

        vec >> norm1d_fwd
        norm1d_fwd >> norm1d_bwd

        vec >> norm2d_fwd
        norm2d_fwd >> norm2d_bwd

    nodes = (
        matrix,
        diag,
        Lmatrix,
        Ldiag,
        central,
        vec,
        norm1d_fwd,
        norm2d_fwd,
        norm1d_bwd,
        norm2d_bwd,
    )

    assert all(node.tainted == True for node in nodes)
    back_matrix = norm2d_bwd.get_data(0)
    back_diag = norm1d_bwd.get_data(0)

    assert all(node.tainted == False for node in nodes)

    result_matrix = norm2d_fwd.get_data(0)
    result_diag = norm1d_fwd.get_data(0)

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
    assert allclose(norm1, result_matrix, atol=atol, rtol=0)
    assert allclose(norm2, result_diag, atol=0, rtol=0)
    assert allclose(inVec, back_matrix, atol=atol, rtol=0)
    assert allclose(inVec, back_diag, atol=0, rtol=0)

    savegraph(graph, f"{output_path}/test_NormalizeCorrelatedVars_00_{dtype}.png")


def test_NormalizeCorrelatedVars_01(dtype="d"):
    inVec = arange(4.0, dtype=dtype) * 100.0
    inV = array([[10, 2, 1], [2, 12, 3], [1, 3, 13]], dtype=dtype)
    inD = inV.diagonal()
    with Graph() as graph1:
        diag = Array("diag", inD, mode="fill")
        vec = Array("vec", inVec, mode="fill")
        norm1d_fwd = NormalizeCorrelatedVars("norm1d fwd")

        vec >> norm1d_fwd.inputs["central"]
        diag >> norm1d_fwd.inputs["matrix"]

    with Graph() as graph2:
        matrix = Array("matrix", inV, mode="fill")
        vec = Array("vec", inVec, mode="fill")
        norm2d_fwd = NormalizeCorrelatedVars("norm2d fwd")

        vec >> norm2d_fwd.inputs["central"]
        matrix >> norm2d_fwd.inputs["matrix"]

    with raises(TypeFunctionError):
        graph1.close()

    with raises(TypeFunctionError):
        graph2.close()
