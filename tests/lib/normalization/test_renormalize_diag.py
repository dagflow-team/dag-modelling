from numpy import allclose, arange, diag, finfo, zeros_like
from pytest import mark

from dag_modelling.core.graph import Graph
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.lib.common import Array
from dag_modelling.lib.normalization import RenormalizeDiag


@mark.parametrize("dtype", ("d", "f"))
@mark.parametrize("mode", ("diag", "offdiag"))
@mark.parametrize("ndiag", (1, 2, 3, 4))
@mark.parametrize("scale", (2.0, 0.33))
def test_RenormalizeDiag(testname, debug_graph, ndiag, dtype, mode, scale, debug=True):
    size = 4
    matrix = arange(size**2, dtype=dtype).reshape(size, size)
    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        array_matrix = Array("matrix", matrix, mode="fill")
        scale_diag = Array("scale", [scale], mode="fill")

        prod = RenormalizeDiag("RenormalizeDiag", mode=mode, ndiag=ndiag)
        array_matrix >> prod
        scale_diag >> prod("scale")

    actual = prod.get_data()
    # NOTE: uncomment to print input and result
    if debug:
        print(f"matrix:\n{matrix}")
        print(f"result:\n{actual}")

    atol = finfo(dtype).resolution
    # check that sum of every column equals 1
    assert allclose(
        tuple(actual[:, icol].sum() for icol in range(size)), [1] * size, atol=atol, rtol=0
    )

    # check that we scaled correct diagonals
    desired = zeros_like(matrix)
    if mode == "diag":
        for idiag in range(-size + 1, size):
            desired += (
                diag(matrix.diagonal(idiag), idiag) * scale
                if abs(idiag) < ndiag
                else diag(matrix.diagonal(idiag), idiag)
            )
    else:
        for idiag in range(-size + 1, size):
            desired += (
                diag(matrix.diagonal(idiag), idiag) * scale
                if abs(idiag) >= ndiag
                else diag(matrix.diagonal(idiag), idiag)
            )
    for icol in range(size):
        desired[:, icol] /= ssum if (ssum := desired[:, icol].sum()) != 0.0 else 1

    # NOTE: uncomment to print desired
    if debug:
        print(f"desired:\n{desired}")
    assert allclose(desired, actual, atol=atol, rtol=0)

    ograph = f"output/{testname}.png"
    print(f"Write graph: {ograph}")
    savegraph(graph, ograph, show="all")
