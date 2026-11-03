from time import process_time

from numpy import allclose, arange, diag, finfo, zeros_like
from pytest import mark

from dag_modelling.core.graph import Graph
from dag_modelling.lib.common import Array
from dag_modelling.lib.normalization import RenormalizeDiag
from dag_modelling.plot.graphviz import savegraph


@mark.parametrize("dtype", ("d", "f"))
@mark.parametrize("mode", ("diag", "offdiag"))
@mark.parametrize("ndiag", (1, 2, 3, 4))
@mark.parametrize("scale", (2.0, 0.33))
@mark.parametrize("layout", ("dense", "sparse-csr"))
def test_RenormalizeDiag(
    test_name, debug_graph, ndiag, dtype, mode, scale, layout, output_path: str, debug=False
):
    size = 4
    matrix = arange(size**2, dtype=dtype).reshape(size, size)
    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        if layout == "sparse-csr":
            array_matrix = Array("matrix", matrix, layout="sparse-csr")
        else:
            array_matrix = Array("matrix", matrix, mode="fill")
        scale_diag = Array("scale", [scale], mode="fill")

        prod = RenormalizeDiag("RenormalizeDiag", mode=mode, ndiag=ndiag)
        array_matrix >> prod
        scale_diag >> prod("scale")

    actual = prod.get_data()
    actual = actual.toarray() if layout == "sparse-csr" else actual
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

    ograph = f"{output_path}/{test_name}.png"
    print(f"Write graph: {ograph}")
    savegraph(graph, ograph, show="all")


@mark.skip(reason="for manual tests")
@mark.parametrize("mode", ("offdiag",))
def test_RenormalizeDiag_benchmark(
    test_name, debug_graph, mode, output_path: str, ndiag=1, scale=1.1, dtype="d"
):
    size = 2400
    n_times = 1000
    matrix = arange(size**2, dtype=dtype).reshape(size, size)
    with Graph(close_on_exit=True, debug=debug_graph) as graph:
        array_matrix = Array("matrix", matrix, mode="fill")
        scale_diag = Array("scale", [scale], mode="fill")

        prod = RenormalizeDiag("RenormalizeDiag", mode=mode, ndiag=ndiag)
        array_matrix >> prod
        scale_diag >> prod("scale")

    actual = prod.get_data()

    atol = finfo(dtype).resolution
    # check that sum of every column equals 1

    prod.function()

    time1 = process_time()
    for _ in range(n_times):
        prod.function()
    time2 = process_time()
    print(f"time for mode={mode}: {(time2-time1)/n_times*1e6:.3f} μs")

    ograph = f"{output_path}/{test_name}.png"
    print(f"Write graph: {ograph}")
    savegraph(graph, ograph, show="all")
