from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import allclose, array, dtype, frombuffer
from numpy.typing import NDArray

from nested_mapping.visitor import NestedMappingVisitor

from ..core.output import Output
from ..tools.logger import INFO1, INFO2, logger

if TYPE_CHECKING:
    from collections.abc import Mapping
    from contextlib import suppress
    from pathlib import Path
    from typing import Any

    with suppress(ImportError):
        from ROOT import TH1, TH1D, TH2, TH2D, TDirectory, TFile, TObject


def to_root(output: Output | Any, **kwargs) -> dict[str, TObject]:
    rets = {}
    if not isinstance(output, Output):
        return rets

    dd = output.dd
    dim = dd.dim
    save_hist = bool(dd.axes_edges)
    save_graph = bool(dd.axes_meshes)
    save_hist = save_hist or not (save_hist or save_graph)
    if dim == 1:
        if save_hist:
            rets["hist"] = to_TH1(output, **kwargs)
        if save_graph:
            rets["graph"] = to_TGraph(output, **kwargs)
    elif dim == 2:
        if save_hist:
            rets["hist"] = to_TH2(output, **kwargs)
        if save_graph:
            rets["graph"] = to_TGraph2(output, **kwargs)
    else:
        raise RuntimeError(f"Unsupported output dimension: {dim}")

    return rets


def _buffer_clean(buf: NDArray) -> NDArray:
    # Should use ascontiguousarray, but it does not work properly for 'f'
    return array(buf, dtype="d")


def edges_to_args(edges: NDArray, *, rtol: float = 1.0e-9) -> tuple:
    edges = _buffer_clean(edges)
    widths = edges[1:] - edges[:-1]
    if allclose(widths, widths.max(), rtol=rtol, atol=0):
        return edges.size - 1, edges[0], edges[-1]

    return edges.size - 1, edges


def get_buffer_hist1(hist: TH1) -> NDArray:
    buf = hist.GetArray()
    buf = frombuffer(buf, dtype(buf.typecode), hist.GetNbinsX() + 2)
    return buf[1:-1]


def get_buffer_hist2(hist: TH2) -> NDArray:
    nx, ny = hist.GetNbinsX(), hist.GetNbinsY()
    buf = hist.GetArray()
    res = frombuffer(buf, dtype(buf.typecode), (nx + 2) * (ny + 2)).reshape((ny + 2, nx + 2))
    # TODO: check order
    return res[1 : ny + 1, 1 : nx + 1].T


def to_TH1(output: Output, *, substitutions: dict[str, str] = {}) -> TH1D:
    from ROOT import TH1D

    data = _buffer_clean(output.data)
    labels = output.labels

    try:
        edges = edges_to_args(output.dd.axes_edges[0].data)
    except IndexError:
        edges = data.size, 0, float(data.size)
    hist = TH1D("", labels.get_root_title(substitutions=substitutions), *edges)
    buffer = get_buffer_hist1(hist)

    buffer[:] = data
    hist.SetEntries(data.sum())
    hist.SetXTitle(output.dd.axis_label(0, axistype="edges", root=True) or "Index [#]")
    hist.SetYTitle(labels.rootaxis)

    return hist


def to_TH2(output: Output, *, substitutions: dict[str, str] = {}) -> TH2D:
    from ROOT import TH2D

    data = _buffer_clean(output.data)
    labels = output.labels

    try:
        edgesX = edges_to_args(output.dd.axes_edges[0].data)
    except IndexError:
        edgesX = data.shape[0], 0, float(data.shape[0])
    try:
        edgesY = edges_to_args(output.dd.axes_edges[1].data)
    except IndexError:
        edgesY = data.shape[1], 0, float(data.shape[1])

    hist = TH2D("", labels.get_root_title(substitutions=substitutions), *(edgesX + edgesY))
    buffer = get_buffer_hist2(hist)

    buffer[:] = data
    hist.SetEntries(data.sum())
    hist.SetXTitle(output.dd.axis_label(0, axistype="edges", root=True) or "Index [#]")
    hist.SetYTitle(output.dd.axis_label(1, axistype="edges", root=True) or "Index [#]")
    hist.SetZTitle(labels.rootaxis)

    return hist


def to_TGraph(output, *, substitutions: dict[str, str] = {}):
    from ROOT import TGraph

    labels = output.labels

    x = _buffer_clean(output.dd.axes_meshes[0].data)
    y = _buffer_clean(output.data)

    title = labels.get_root_title(substitutions=substitutions)
    xtitle = output.dd.axis_label(0, axistype="mesh", root=True) or "Index [#]"
    ytitle = labels.rootaxis

    graph = TGraph(x.size, x, y)
    graph.SetTitle(f"{title};{xtitle};{ytitle}")

    return graph


def to_TGraph2(output, *, substitutions: dict[str, str] = {}):
    from ROOT import TGraph2D

    labels = output.labels

    x = _buffer_clean(output.dd.axes_meshes[0].data)
    y = _buffer_clean(output.dd.axes_meshes[1].data)
    z = _buffer_clean(output.data)

    title = labels.get_root_title(substitutions=substitutions)
    xtitle = output.dd.axis_label(0, axistype="mesh", root=True) or "Index [#]"
    ytitle = output.dd.axis_label(1, axistype="mesh", root=True) or "Index [#]"
    ztitle = labels.rootaxis

    graph = TGraph2D(x.size, x.ravel(), y.ravel(), z.ravel())
    graph.SetTitle(f"{title};{xtitle};{ytitle};{ztitle}")

    return graph


class ExportToRootVisitor(NestedMappingVisitor):
    __slots__ = (
        "_file",
        "_cwd",
        "_prevd",
        "_level",
        "_i_element",
        "_n_elements",
        "_latex_substitutions",
        "_kwargs",
    )
    _file: TFile
    _prevd: list["TDirectory"]
    _cwd: TDirectory
    _level: int
    _i_element: int
    _n_elements: int
    _latex_substitutions: Mapping[str, str]
    _kwargs: dict[str, Any]

    def __init__(
        self,
        filename: Path | str,
        *,
        latex_substitutions: Mapping[str, str] = {},
        **kwargs,
    ):
        from ROOT import TFile

        filename = str(filename)
        logger.log(INFO1, f"Create {filename}")
        self._file = TFile(filename, "RECREATE")
        self._file.AddDirectory(False)
        if self._file.IsZombie():
            raise RuntimeError(f"File {filename!s} is zombie")
        self._prevd = []
        self._cwd = self._file
        self._level = 0

        self._latex_substitutions = dict(latex_substitutions)
        self._kwargs = kwargs

        self._i_element = 0
        self._n_elements = 0

    def start(self, dct):
        self._n_elements = dct.len_recursive()
        self._i_element = 0

    def enterdict(self, key, v):
        if not key:
            return

        self._level += 1

        cwd = self._file
        for subkey in key:
            cwd = cwd.mkdir(subkey, "", True)

        self._prevd.append(self._cwd)
        self._cwd = cwd

    def visit(self, key, value):
        self._i_element += 1
        path = "/".join(key[self._level:])

        objects = to_root(value, substitutions=self._latex_substitutions)
        if not objects:
            logger.log(INFO2, f"empty {'/'.join(key)} [{self._i_element}/{self._n_elements}]")
            return

        name = path
        hist = objects.pop("hist", None)
        if hist is not None:
            logger.log(INFO2, f"write {'/'.join(key)} [{self._i_element}/{self._n_elements}]")
            self._cwd.WriteTObject(hist, name, "overwrite")
            name = f"{name}_graph"

        graph = objects.pop("graph", None)
        if graph is not None:
            logger.log(INFO2, f"write {'/'.join(key)} [{self._i_element}/{self._n_elements}]")
            self._cwd.WriteTObject(graph, name, "overwrite")

        if objects:
            raise RuntimeError(f"Unsupported ROOT objects: {objects}")

    def exitdict(self, k, v):
        if not self._prevd:
            return

        self._level -= 1

        if self._cwd.GetNkeys() == 0:
            prevd = self._prevd[-1]
            prevd.rmdir(self._cwd.GetName())
        self._cwd = self._prevd.pop()

    def stop(self, dct):
        logger.log(INFO1, f"Close {self._file.GetName()}")
        self._file.Close()
