from numpy import allclose, array, diag, finfo, log
from pytest import mark, raises

from dag_modelling.core.exception import TypeFunctionError
from dag_modelling.core.graph import Graph
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.lib.common import Array
from dag_modelling.lib.statistics import LogProdDiag


@mark.parametrize("dtype", ("d", "f"))
def test_LogProdDiag_00(test_name, debug_graph, dtype, output_path: str):
    inV1 = array([[10, 0, 0], [2, 12, 0], [1, 3, 13]], dtype=dtype)
    inV2 = inV1**2
    inD = diag(inV1)
    inL2d1 = 2 * log(diag(inV1)).sum()
    inL2d2 = 2 * log(diag(inV2)).sum()
    inL1d = 2 * log(inD).sum()

    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        V1 = Array("V1", inV1, mode="store")
        V2 = Array("V2", inV2, mode="store")
        D = Array("D", (inD), mode="store")
        log_prod_diag2d = LogProdDiag("LogProdDiag 2d")
        log_prod_diag1d = LogProdDiag("LogProdDiag 1d")
        (V1, V2) >> log_prod_diag2d
        D >> log_prod_diag1d

    assert V1.tainted == True
    assert V2.tainted == True
    assert D.tainted == True
    assert log_prod_diag2d.tainted == True
    assert log_prod_diag1d.tainted == True

    result2d1 = log_prod_diag2d.get_data(0)
    result2d2 = log_prod_diag2d.get_data(1)
    result1d = log_prod_diag1d.get_data(0)
    assert V1.tainted == False
    assert V2.tainted == False
    assert D.tainted == False
    assert log_prod_diag2d.tainted == False
    assert log_prod_diag1d.tainted == False

    atol = finfo(dtype).resolution
    assert allclose(inL2d1, result2d1, atol=atol, rtol=0)
    assert allclose(inL2d2, result2d2, atol=atol, rtol=0)
    assert allclose(inL1d, result1d, atol=atol, rtol=0)

    # Change to V1 and D
    inV1 += diag(range(1,4))
    inD = diag(inV1)

    assert V1.outputs[0].set(inV1)==True
    assert D.outputs[0].set(inD)==True

    inL2d1 = 2 * log(diag(inV1)).sum()
    inL2d2 = 2 * log(diag(inV2)).sum()
    inL1d = 2 * log(inD).sum()

    assert V1.tainted == False
    assert V2.tainted == False
    assert D.tainted == False
    assert log_prod_diag2d.tainted == True
    assert log_prod_diag1d.tainted == True

    result2d1 = log_prod_diag2d.get_data(0)
    result2d2 = log_prod_diag2d.get_data(1)
    result1d = log_prod_diag1d.get_data(0)
    assert V1.tainted == False
    assert V2.tainted == False
    assert D.tainted == False
    assert log_prod_diag2d.tainted == False
    assert log_prod_diag1d.tainted == False

    atol = finfo(dtype).resolution
    assert allclose(inL2d1, result2d1, atol=atol, rtol=0)
    assert allclose(inL2d2, result2d2, atol=atol, rtol=0)
    assert allclose(inL1d, result1d, atol=atol, rtol=0)

    savegraph(graph, f"{output_path}/{test_name}.png")


def test_LogProdDiag_01_type_functions():
    inV = array(
        [
            [10, 2, 1],
            [2, 12, 3],
        ],
        dtype="d",
    )

    with Graph() as g1:
        V1 = Array("V1", inV, mode="store")
        log_prod_diag1 = LogProdDiag("LogProdDiag")
        V1 >> log_prod_diag1

    with Graph() as g2:
        V2 = Array("V2", inV[0], mode="store")
        log_prod_diag2 = LogProdDiag("LogProdDiag")
        V2 >> log_prod_diag1

    with raises(TypeFunctionError):
        g1.close()

    with raises(TypeFunctionError):
        g2.close()
