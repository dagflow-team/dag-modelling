from numpy import arange, meshgrid
from pytest import mark

from dag_modelling.core.graph import Graph
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.lib.common import Array
from dag_modelling.core.storage import NodeStorage

@mark.skip(reason="no way of currently testing this")
@mark.parametrize('dtype', ('d', 'f',))
def test_to_root(test_name, debug_graph, dtype, output_path: str):
    sizex = 12
    sizey = 10
    data = (arange(sizex, dtype=dtype)-6)**2
    datay = (0.5*(arange(sizey, dtype=dtype)-5))**2
    data2 = data[:,None]*datay[None,:]

    meshx = arange(sizex, dtype=dtype)+0.5
    edgesx = arange(sizex+1, dtype=dtype)

    meshy = arange(sizey, dtype=dtype)+0.5
    edgesy = arange(sizey+1, dtype=dtype)

    mesh2x, mesh2y = meshgrid(meshx, meshy, indexing='ij')

    labels = {
            'edgesx': {
                'text': 'Edges X',
                'plot_title': 'Edges X $\\theta^{2}$',
                'axis': 'X $\\theta^{2}$',
                },
            'edgesy': {
                'text': 'Edges Y',
                'plot_title': 'Edges Y $\\theta^{2}$',
                'axis': 'Y $\\theta^{2}$',
                'rootaxis': 'Y #Theta^{3}',
                },
            'meshx': {
                'text': 'Mesh X 1d',
                'plot_title': 'Mesh X $\\theta^{2}$ 1d',
                },
            'meshy': {
                'text': 'Mesh Y 1d',
                'plot_title': 'Mesh Y $\\theta^{2}$ 1d',
                'axis': 'Y $\\theta^{2}$ 1d',
                'rootaxis': 'Y #Theta^{3}  1d',
                },
            'mesh2x': {
                'text': 'Mesh X 2d',
                'plot_title': 'Mesh X $\\theta^{2}$ 2d',
                },
            'mesh2y': {
                'text': 'Mesh Y 2d',
                'plot_title': 'Mesh Y $\\theta^{2}$ 2d',
                'axis': 'Y $\\theta^{2}$ 2d',
                'rootaxis': 'Y #Theta^{3}  2d',
                },
            'array_edges': {
                'text': 'Histogram 1d',
                'plot_title': 'Histogram 1d $\\theta^{2}$',
                },
            'array_mesh': {
                'text': 'Graph 1d',
                'plot_title': 'Graph 1d $\\theta^{2}$',
                },
            'case_2d': {
                'both': {
                    'array_2d_both': {
                        'text': 'Graph 2d',
                        'plot_title': 'Graph 2d $\\theta^{2}$',
                        'root_title': 'LaTeX Graph 2d #Theta^{3}',
                        },
                    },
                'array_2d_edges': {
                    'text': 'Histogram 2d',
                    'plot_title': 'Histogram 2d $\\theta^{2}$',
                    },
                'array_2d_mesh': {
                    'text': 'Graph 2d',
                    'plot_title': 'Graph 2d $\\theta^{2}$',
                    },
                }
            }

    with Graph(close_on_exit=True, debug=debug_graph) as graph, NodeStorage({}) as storage:
        EdgesX, _ = Array.replicate(name='edgesx', array=edgesx)
        EdgesY, _ = Array.replicate(name='edgesy', array=edgesy)

        MeshX, _ = Array.replicate(name='meshx', array=meshx)

        Mesh2X, _ = Array.replicate(name='mesh2x', array=mesh2x)
        Mesh2Y, _ = Array.replicate(name='mesh2y', array=mesh2y)

        arr1e, _ = Array.replicate(name='array_edges', array=data, edges=EdgesX.outputs[0])
        arr1n, _ = Array.replicate(name='array_mesh', array=data)
        arr1n.outputs[0].dd.axes_meshes = (MeshX.outputs[0],)
        arr1n.outputs[0].dd.meshes_inherited = False

        arr2e, _ = Array.replicate(name='case_2d.array_2d_edges', array=data2, edges=(EdgesX, EdgesY))
        arr2n, _ = Array.replicate(name='case_2d.array_2d_mesh', array=data2)
        arr2n.outputs[0].dd.axes_meshes = (Mesh2X.outputs[0], Mesh2Y.outputs[0])
        arr2n.outputs[0].dd.meshes_inherited = False

        arr2b, _ = Array.replicate(name='case_2d.both.array_2d_both', array=data2, edges=(EdgesX, EdgesY))
        arr2b.outputs[0].dd.axes_meshes = (Mesh2X.outputs[0], Mesh2Y.outputs[0])
        arr2b.outputs[0].dd.meshes_inherited = False

    storage('outputs').read_labels(labels)
    tbl = storage.to_table()
    print(tbl)

    storage('outputs').plot(show_all=False)

    try:
        import ROOT
    except ImportError:
        pass
    else:
        storage('outputs').to_root(f'output/{test_name}.root')

    savegraph(graph, f"{output_path}/{test_name}.pdf")
