from typing import Literal

from numpy import allclose, array, finfo
from pytest import mark

from dag_modelling.core.graph import Graph
from dag_modelling.lib.common import Array
from dag_modelling.lib.hist import (
    AxisDistortionMatrix,
    AxisDistortionMatrixLinear,
    AxisDistortionMatrixPointwise,
)
from dag_modelling.plot.graphviz import savegraph


@mark.parametrize(
    "dtype",
    (
        "d",
        "f",
    ),
)
@mark.parametrize(
    "setname",
    (
        "test1",
        "test2_undershoot_left",
        "test2_undershoot_right",
        "test3_minimal1",
        "test3_minimal2",
        "test3_minimal3",
        "test3_minimal4",
        "test3_minimal5",
        "test3_minimal6",
        "test4_variable",
    ),
)
@mark.parametrize(
    "mode",
    # ("exact", "linear", "pointwise"),
    ("pointwise",),
)
def test_AxisDistortionMatrix(
    setname: str, dtype: str, mode: Literal["exact", "linear", "pointwise"], output_path: str
):
    test_sets_current = test_sets[mode]
    edgesset = test_sets_current[setname]
    edges = array(edgesset["edges"], dtype=dtype)
    edges_modified = array(edgesset["edges_modified"], dtype=dtype)
    edges_backward = array(edgesset["edges_backward"], dtype=dtype)
    desired = array(edgesset["matrix"], dtype=dtype)
    nbins = len(edges) - 1

    print()
    print("Edges before:\n", edges)
    print("Edges after:\n", edges_modified)
    print("Edges back:\n", edges_backward)
    print("Desired matrix:\n", desired)
    print("Desired matrix sum:\n", desired.sum(axis=0))

    with Graph(close_on_exit=True) as graph:
        Edges = Array("Edges", edges, mode="fill")
        EdgesModified = Array("Edges modified", edges_modified, mode="fill")
        if mode == "exact":
            EdgesBackward = Array("Edges, projected backward", edges_backward, mode="fill")

        match mode:
            case "linear":
                mat = AxisDistortionMatrixLinear("LSNL matrix (linear)")

                Edges >> mat.inputs["EdgesOriginal"]
                Edges >> mat.inputs["EdgesTarget"]
                EdgesModified >> mat.inputs["EdgesModified"]
            case "exact":
                mat = AxisDistortionMatrix("LSNL matrix")

                Edges >> mat.inputs["EdgesOriginal"]
                Edges >> mat.inputs["EdgesTarget"]
                EdgesModified >> mat.inputs["EdgesModified"]
                EdgesBackward >> mat.inputs["EdgesModifiedBackwards"]
            case "pointwise":
                mat = AxisDistortionMatrixPointwise("LSNL matrix (pointwise)")

                Edges >> mat.inputs["EdgesOriginal"]
                Edges >> mat.inputs["EdgesTarget"]

                Edges >> mat.inputs["DistortionOriginal"]
                EdgesModified >> mat.inputs["DistortionTarget"]
            case _:
                assert False

    res = mat.get_data()

    ressum = res.sum(axis=0)
    print("Obtained matrix:\n", res)
    print("Obtained matrix sum:\n", ressum)

    if mode == "pointwise":
        atol = finfo(dtype).resolution * 0.5
    else:
        atol = 0 if dtype == "d" else finfo(dtype).resolution * 0.5
    assert allclose(res, desired, atol=atol, rtol=0)

    idxstart, idxend = 0, nbins
    while idxstart < nbins and ressum[idxstart] < 1.0:
        idxstart += 1
    while idxend > 0 and ressum[idxend - 1] < 1.0:
        idxend -= 1
    assert allclose(ressum[idxstart:idxend], 1, rtol=0, atol=0)

    out_edges = mat.outputs[0].dd.axes_edges
    assert out_edges[0] is out_edges[1]
    assert out_edges[0] is Edges.outputs[0]

    savegraph(
        graph,
        f"{output_path}/test_AxisDistortionMatrix{mode.capitalize()}_{dtype}.png",
    )


# fmt: off
test_sets = {
        "linear": {
            'test1': dict(
                # from:           0         1    2              3              4
                edges          = [1.0,      2.0, 3.0,           4.0,           5.0, ],
                edges_backward = [     1.4,           3.4, 3.8,      4.2, 4.6       ],
                # to                   0              1    2         3    4
                # edges        = [     1.0,           2.0, 3.0,      4.0, 5.0       ]
                edges_modified = [0.8,      1.4, 1.8,           3.5,           5.5, ],
                # from:           0         1    2              3              4
                matrix =       [                                                       # To:
                # From:                 0                    1                    2                    3       #
                    [ (1.4-1.0)/(1.4-0.8), (1.8-1.4)/(1.8-1.4), (2.0-1.8)/(3.5-1.8),           (0.0-0.0) ],    # 0
                    [           (0.0-0.0),           (0.0-0.0), (3.0-2.0)/(3.5-1.8),           (0.0-0.0) ],    # 1
                    [           (0.0-0.0),           (0.0-0.0), (3.5-3.0)/(3.5-1.8), (4.0-3.5)/(5.5-3.5) ],    # 2
                    [           (0.0-0.0),           (0.0-0.0),           (0.0-0.0), (5.0-4.0)/(5.5-3.5) ],    # 3
                    ]
                ),
            'test2_undershoot_left': dict(
                # from:                0    1    2              3              4
                edges          = [     1.0, 2.0, 3.0,           4.0,           5.0, ],
                edges_backward = [0.9,                3.4, 3.8,      4.2, 4.6       ],
                # to              0                   1    2         3    4
                # edges        = [1.0,                2.0, 3.0,      4.0, 5.0       ]
                edges_modified = [     1.1, 1.4, 1.8,           3.5,           5.5, ],
                # from:                0    1    2              3              4
                matrix =       [                                                       # To:
                # From:                 0                    1                    2                    3       #
                    [ (1.4-1.1)/(1.4-1.1), (1.8-1.4)/(1.8-1.4), (2.0-1.8)/(3.5-1.8),           (0.0-0.0) ],    # 0
                    [           (0.0-0.0),           (0.0-0.0), (3.0-2.0)/(3.5-1.8),           (0.0-0.0) ],    # 1
                    [           (0.0-0.0),           (0.0-0.0), (3.5-3.0)/(3.5-1.8), (4.0-3.5)/(5.5-3.5) ],    # 2
                    [           (0.0-0.0),           (0.0-0.0),           (0.0-0.0), (5.0-4.0)/(5.5-3.5) ],    # 3
                    ]
                ),
            'test2_undershoot_right': dict(
                # from:                0    1    2              3              4
                edges          = [     1.0, 2.0, 3.0,           4.0,           5.0, ],
                edges_backward = [0.9,                3.4, 3.8,      4.2, 4.6       ],
                # to              0                   1    2         3              4
                # edges        = [1.0,                2.0, 3.0,      4.0,           5.0       ]
                edges_modified = [     1.1, 1.4, 1.9,           3.5,           4.9, ],
                # from:                0    1    2              3              4
                matrix =       [                                                       # To:
                # From:                 0                    1                    2                    3       #
                    [ (1.4-1.1)/(1.4-1.1), (1.9-1.4)/(1.9-1.4), (2.0-1.9)/(3.5-1.9),           (0.0-0.0) ],    # 0
                    [           (0.0-0.0),           (0.0-0.0), (3.0-2.0)/(3.5-1.9),           (0.0-0.0) ],    # 1
                    [           (0.0-0.0),           (0.0-0.0), (3.5-3.0)/(3.5-1.9), (4.0-3.5)/(4.9-3.5) ],    # 2
                    [           (0.0-0.0),           (0.0-0.0),           (0.0-0.0), (4.9-4.0)/(4.9-3.5) ],    # 3
                    ]
                ),
            'test3_minimal1': dict(
                # from:           0         1
                edges          = [1.0,      2.0            ],
                edges_backward = [     1.4,           3.4  ],
                # to                   0              1
                # edges        = [     1.0,           2.0  ]
                edges_modified = [0.9,      1.4            ],
                # from:                0    1
                matrix =       [       # To:
                # From:           0    #
                    [ (1.4-1.0)/(1.4-0.9) ], # 0
                    ]
                ),
            'test3_minimal2': dict(
                # from:                0    1
                edges          = [     1.0, 2.0            ],
                edges_backward = [0.9,                3.4  ],
                # to              0                   1
                # edges        = [1.0,                2.0  ]
                edges_modified = [     1.1, 1.4            ],
                # from:                0    1
                matrix =       [       # To:
                # From:           0    #
                    [ (1.4-1.1)/(1.4-1.1) ], # 0
                    ]
                ),
            'test3_minimal3': dict(
                # from:           0              1
                edges          = [1.0,           2.0 ],
                edges_backward = [     1.4,  1.6     ],
                # to                   0     1
                # edges        = [     1.0,  2.0     ]
                edges_modified = [0.9,           2.4 ],
                # from:                0         1
                matrix =       [       # To:
                # From:           0    #
                    [ (2.0-1.0)/(2.4-0.9) ], # 0
                    ]
                ),
            'test3_minimal4': dict(
                # from:                0         1
                edges          = [     1.0,      2.0 ],
                edges_backward = [0.9,       1.6     ],
                # to              0          1
                # edges        = [1.0,       2.0     ]
                edges_modified = [     1.1,      2.4 ],
                # from:                0         1
                matrix =       [       # To:
                # From:           0    #
                    [ (2.0-1.1)/(2.4-1.1) ], # 0
                    ]
                ),
            'test3_minimal5': dict(
                # from:                     0    1
                edges          = [          1.0, 2.0 ],
                edges_backward = [0.8,  0.9          ],
                # to              0     1
                # edges        = [1.0,  2.0          ]
                edges_modified = [          2.1, 2.4 ],
                # from:                0         1
                matrix =       [       # To:
                # From:           0    #
                    [           0.0 ], # 0
                    ]
                ),
            'test3_minimal6': dict(
                # from:           0    1
                edges          = [1.0, 2.0           ],
                edges_backward = [         2.8,  2.9 ],
                # to                       0     1
                # edges        = [         1.0,  2.0 ]
                edges_modified = [0.8, 0.9           ],
                # from:                0         1
                matrix =       [       # To:
                # From:           0    #
                    [           0.0 ], # 0
                    ]
                ),
            'test4_variable': dict(
                # from:           0         1    2              3              4
                edges          = [1.0,      2.0, 4.0,           7.0,           11.0 ],
                edges_backward = [     1.4,           4.4, 4.8,      7.2, 8.6       ],
                # to                   0              1    2         3    4
                # edges        = [     1.0,           2.0, 4.0,      7.0, 11.0       ]
                edges_modified = [0.8,      1.4, 1.8,           6.5,           11.5 ],
                # from:           0         1    2              3              4
                matrix =       [                                                       # To:
                # From:                 0                    1                    2                    3        #
                    [ (1.4-1.0)/(1.4-0.8), (1.8-1.4)/(1.8-1.4), (2.0-1.8)/(6.5-1.8),           (0.0-0.0) ],     # 0
                    [           (0.0-0.0),           (0.0-0.0), (4.0-2.0)/(6.5-1.8),           (0.0-0.0) ],     # 1
                    [           (0.0-0.0),           (0.0-0.0), (6.5-4.0)/(6.5-1.8), (7.0-6.5)/(11.5-6.5) ],    # 2
                    [           (0.0-0.0),           (0.0-0.0),           (0.0-0.0), (11.0-7.0)/(11.5-6.5) ],    # 3
                    ]
                ),
        },
        "exact": {
            'test1': dict(
                # from:           0         1    2              3              4
                edges          = [1.0,      2.0, 3.0,           4.0,           5.0, ],
                edges_backward = [     1.4,           3.4, 3.8,      4.2, 4.6       ],
                # to                   0              1    2         3    4
                # edges        = [     1.0,           2.0, 3.0,      4.0, 5.0       ]
                edges_modified = [0.8,      1.4, 1.8,           3.5,           5.5, ],
                # from:           0         1    2              3              4
                matrix =       [                                                       # To:
                # From:           0              1              2              3       #
                    [ (2.0-1.4)/1.0, (3.0-2.0)/1.0, (3.4-3.0)/1.0,           0.0 ],    # 0
                    [           0.0,           0.0, (3.8-3.4)/1.0,           0.0 ],    # 1
                    [           0.0,           0.0, (4.0-3.8)/1.0, (4.2-4.0)/1.0 ],    # 2
                    [           0.0,           0.0,           0.0, (4.6-4.2)/1.0 ],    # 3
                    ]
                ),
            'test2_undershoot_left': dict(
                # from:                0    1    2              3              4
                edges          = [     1.0, 2.0, 3.0,           4.0,           5.0, ],
                edges_backward = [0.9,                3.4, 3.8,      4.2, 4.6       ],
                # to              0                   1    2         3    4
                # edges        = [1.0,                2.0, 3.0,      4.0, 5.0       ]
                edges_modified = [     1.1, 1.4, 1.8,           3.5,           5.5, ],
                # from:                0    1    2              3              4
                matrix =       [                                                       # To:
                # From:           0              1              2              3       #
                    [ (2.0-1.0)/1.0, (3.0-2.0)/1.0, (3.4-3.0)/1.0,           0.0 ],    # 0
                    [           0.0,           0.0, (3.8-3.4)/1.0,           0.0 ],    # 1
                    [           0.0,           0.0, (4.0-3.8)/1.0, (4.2-4.0)/1.0 ],    # 2
                    [           0.0,           0.0,           0.0, (4.6-4.2)/1.0 ],    # 3
                    ]
                ),
            'test2_undershoot_right': dict(
                # from:                0    1    2              3              4
                edges          = [     1.0, 2.0, 3.0,           4.0,           5.0, ],
                edges_backward = [0.9,                3.4, 3.8,      4.2, 4.6       ],
                # to              0                   1    2         3             4
                # edges        = [1.0,                2.0, 3.0,      4.0,          5.0 ]
                edges_modified = [     1.1, 1.4, 1.8,           3.5,           4.9, ],
                # from:                0    1    2              3              4
                matrix =       [                                                       # To:
                # From:           0              1              2              3       #
                    [ (2.0-1.0)/1.0, (3.0-2.0)/1.0, (3.4-3.0)/1.0,           0.0 ],    # 0
                    [           0.0,           0.0, (3.8-3.4)/1.0,           0.0 ],    # 1
                    [           0.0,           0.0, (4.0-3.8)/1.0, (4.2-4.0)/1.0 ],    # 2
                    [           0.0,           0.0,           0.0, (4.6-4.2)/1.0 ],    # 3
                    ]
                ),
            'test3_minimal1': dict(
                # from:           0         1
                edges          = [1.0,      2.0            ],
                edges_backward = [     1.4,           3.4  ],
                # to                   0              1
                # edges        = [     1.0,           2.0  ]
                edges_modified = [0.9,      1.4            ],
                # from:                0    1
                matrix =       [       # To:
                # From:           0    #
                    [ (2.0-1.4)/1.0 ], # 0
                    ]
                ),
            'test3_minimal2': dict(
                # from:                0    1
                edges          = [     1.0, 2.0            ],
                edges_backward = [0.9,                3.4  ],
                # to              0                   1
                # edges        = [1.0,                2.0  ]
                edges_modified = [     1.1, 1.4            ],
                # from:                0    1
                matrix =       [       # To:
                # From:           0    #
                    [ (2.0-1.0)/1.0 ], # 0
                    ]
                ),
            'test3_minimal3': dict(
                # from:           0              1
                edges          = [1.0,           2.0 ],
                edges_backward = [     1.4,  1.6     ],
                # to                   0     1
                # edges        = [     1.0,  2.0     ]
                edges_modified = [0.9,           2.4 ],
                # from:                0         1
                matrix =       [       # To:
                # From:           0    #
                    [ (1.6-1.4)/1.0 ], # 0
                    ]
                ),
            'test3_minimal4': dict(
                # from:                0         1
                edges          = [     1.0,      2.0 ],
                edges_backward = [0.9,       1.6     ],
                # to              0          1
                # edges        = [1.0,       2.0     ]
                edges_modified = [     1.1,      2.4 ],
                # from:                0         1
                matrix =       [       # To:
                # From:           0    #
                    [ (1.6-1.0)/1.0 ], # 0
                    ]
                ),
            'test3_minimal5': dict(
                # from:                     0    1
                edges          = [          1.0, 2.0 ],
                edges_backward = [0.8,  0.9          ],
                # to              0     1
                # edges        = [1.0,  2.0          ]
                edges_modified = [          2.1, 2.4 ],
                # from:                0         1
                matrix =       [       # To:
                # From:           0    #
                    [           0.0 ], # 0
                    ]
                ),
            'test3_minimal6': dict(
                # from:           0    1
                edges          = [1.0, 2.0           ],
                edges_backward = [         2.8,  2.9 ],
                # to                       0     1
                # edges        = [         1.0,  2.0 ]
                edges_modified = [0.8, 0.9           ],
                # from:                0         1
                matrix =       [       # To:
                # From:           0    #
                    [           0.0 ], # 0
                    ]
                ),
            'test4_variable': dict(
                # from:           0         1    2              3              4
                edges          = [1.0,      2.0, 4.0,           7.0,           11.0 ],
                edges_backward = [     1.4,           4.4, 4.8,      7.2, 8.6       ],
                # to                   0              1    2         3    4
                # edges        = [     1.0,           2.0, 4.0,      7.0, 11.0       ]
                edges_modified = [0.8,      1.4, 1.8,           6.5,           11.5 ],
                # from:           0         1    2              3              4
                matrix =       [                                                       # To:
                # From:           0              1              2              3       #
                    [ (2.0-1.4)/1.0, (4.0-2.0)/2.0, (4.4-4.0)/3.0,           0.0 ],    # 0
                    [           0.0,           0.0, (4.8-4.4)/3.0,           0.0 ],    # 1
                    [           0.0,           0.0, (7.0-4.8)/3.0, (7.2-7.0)/4.0 ],    # 2
                    [           0.0,           0.0,           0.0, (8.6-7.2)/4.0 ],    # 3
                    ]
                ),
        }
    }
test_sets["pointwise"] = test_sets["linear"]
