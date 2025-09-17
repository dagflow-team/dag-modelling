from numpy import allclose, arange, diag
from pytest import mark, raises

from dag_modelling.core.exception import TypeFunctionError
from dag_modelling.core.graph import Graph
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.lib.common import Array
from dag_modelling.lib.summation import SumMatOrDiag


@mark.parametrize("dtype", ("d", "f"))
def test_SumMatOrDiag_01(dtype, debug_graph, output_path: str):
    for size in (5, 4, 100):
        array1 = arange(size, dtype=dtype) + 1.0
        array2 = arange(size, dtype=dtype) * 3
        matrix1 = arange(size * size, dtype=dtype).reshape(size, size) + 1.0
        matrix2 = arange(size * size, dtype=dtype).reshape(size, size) * 2.5
        arrays_in = (array1, array2, matrix1, matrix2)

        combinations = (
            (0,),
            (2,),
            (0, 1),
            (0, 2),
            (2, 0),
            (0, 1, 2),
            (2, 3),
            (0, 1, 2, 3),
        )

        sms = []

        with Graph(close_on_exit=True, debug=debug_graph) as graph:
            arrays = tuple(Array(f"test {i}", array_in, mode="fill") for i, array_in in enumerate(arrays_in))

            for cmb in combinations:
                sm = SumMatOrDiag(f"sum {cmb}")
                tuple(arrays[i] for i in cmb) >> sm
                sms.append(sm)

        for cmb, sm in zip(combinations, sms):
            res = 0.0
            all1d = True
            for i in cmb:
                array_in = arrays_in[i]
                if len(array_in.shape) == 1:
                    array_in = diag(array_in)
                else:
                    all1d = False
                res += array_in

            if all1d:
                res = diag(res)

            assert sm.tainted
            output_data = sm.outputs[0].data
            assert allclose(output_data, res, rtol=0, atol=0)
            assert not sm.tainted
            sm.taint()
            sm.touch()
            assert allclose(output_data, res, rtol=0, atol=0)

        savegraph(
            graph,
            f"{output_path}/test_SumMatOrDiag_00_{dtype}_{size}.png",
            show="all",
        )


@mark.parametrize("dtype", ("d", "f"))
def test_SumMatOrDiag_02(dtype, debug_graph):
    size = 5
    in_array1 = arange(size, dtype=dtype)  # 0
    in_array2 = arange(size + 1, dtype=dtype)  # 1
    in_matrix1 = arange(size**2, dtype=dtype).reshape(size, size)
    in_matrix2 = arange(size * (size + 1), dtype=dtype).reshape(size, size + 1)  # 3
    in_matrix3 = arange((size + 1) * (size + 1), dtype=dtype).reshape(size + 1, size + 1)  # 4
    arrays_in = (in_array1, in_array2, in_matrix1, in_matrix2, in_matrix3)

    combinations = ((0, 1), (0, 3), (0, 4), (3, 0), (4, 0), (2, 3), (2, 4))
    with Graph(close_on_exit=False, debug=debug_graph):
        arrays = tuple(Array(f"test {i}", array_in, mode="fill") for i, array_in in enumerate(arrays_in))

        for i1, i2 in combinations:
            sm = SumMatOrDiag("sum")
            (arrays[i1], arrays[i2]) >> sm

            with raises(TypeFunctionError):
                sm.close()
