from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from matplotlib import colormaps
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from numpy import asanyarray, meshgrid, zeros_like
from numpy.ma import array as masked_array

from ..core.node_base import NodeBase
from ..core.output import Output
from ..tools.logger import INFO1, logger

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


class plot_auto:
    __slots__ = (
        "_object",
        "_output",
        "_array",
        "_edges",
        "_meshes",
        "_plotoptions",
        "_ret",
        "_title",
        "_xlabel",
        "_ylabel",
        "_zlabel",
        "_latex_substitutions",
    )
    _object: NodeBase | Output | ArrayLike
    _output: Output | None
    _array: NDArray
    _edges: tuple[NDArray, ...]
    _meshes: tuple[NDArray, ...]
    _plotoptions: Mapping[str, Any]
    _ret: tuple

    _title: str | None
    _xlabel: str | None
    _ylabel: str | None
    _zlabel: str | None

    _latex_substitutions: Mapping[str, str]

    def __init__(
        self,
        object: NodeBase | Output | ArrayLike,
        *args,
        filter_kw: dict = {},
        show_path: bool = True,
        save: str | None = None,
        close: bool = False,
        show: bool = False,
        savefig_kw: dict = {},
        latex_substitutions: Mapping[str, str] = {},
        plotoptions: Mapping[str, Any] | str = {},
        **kwargs,
    ):
        match plotoptions:
            case str():
                self._plotoptions = {"method": plotoptions}
            case Mapping():
                self._plotoptions = deepcopy(plotoptions)
            case _:
                raise ValueError(plotoptions)

        self._object = object
        self._output = None
        self._latex_substitutions = latex_substitutions
        self._ret = ()
        self._get_data(**filter_kw)
        self._get_labels()

        if (kwargs_extra := self._plotoptions.get("kwargs")) is not None:
            kwargs = dict(kwargs, **kwargs_extra)

        ndim = len(self._array.shape)
        if ndim == 1:
            _edges = self._edges[0] if self._edges else None
            _meshes = self._meshes[0] if self._meshes else None
            self._ret = plot_array_1d(
                self._array,
                _edges,
                _meshes,
                *args,
                plotter=self,
                **kwargs,
            )
        elif ndim == 2:
            colorbar = kwargs.pop("colorbar", {})
            if colorbar is True:
                colorbar = {}
            kwargs.setdefault("rasterized", self._plotoptions.get("rasterized", True))
            if self._output and isinstance(colorbar, Mapping):
                colorbar.setdefault("label", self._output.labels.axis_unit)
            self._ret = plot_array_2d(
                self._array,
                self._edges,
                self._meshes,
                *args,
                plotter=self,
                colorbar=colorbar,
                plotoptions=self._plotoptions,
                **kwargs,
            )
        else:
            raise RuntimeError(f"Do not know how to plot {ndim}d")

        if self._output is not None:
            has_legend = "label" in kwargs
            self.annotate_axes(show_path=show_path, legend=has_legend)

        if save:
            logger.log(INFO1, f"Write: {save}")
            plt.savefig(save, **savefig_kw)
        if show:
            plt.show()
        if close:
            plt.close()

    def _get_output_data(self, *args, **kwargs):
        if plotoptions := self._output.labels.plotoptions:
            self._plotoptions = dict(plotoptions, **self._plotoptions)

        if (masked_value := self._plotoptions.get("mask_value", None)) is not None:
            kwargs = dict(kwargs, masked_value=masked_value)

        data = _mask_if_needed(self._output.data, *args, **kwargs)
        self._array = data
        self._edges = self._output.dd.edges_arrays
        self._meshes = self._output.dd.meshes_arrays

    def _get_array_data(self, *args, **kwargs):
        self._array = _mask_if_needed(self._object, *args, **kwargs)
        self._edges = ()
        self._meshes = ()

    def _get_data(self, *args, **kwargs):
        if isinstance(self._object, Output):
            self._output = self._object
        elif isinstance(self._object, NodeBase):
            self._output = self._object.outputs[0]
        else:
            if masked_value := self._plotoptions.get("mask_value", None):
                kwargs = dict(kwargs, masked_value=masked_value)
            self._get_array_data(*args, **kwargs)
            return

        self._get_output_data(*args, **kwargs)

    def _get_labels(self):
        self._zlabel = None
        if not self._output:
            self._title = None
            self._xlabel = None
            self._ylabel = None
            return

        labels = self._output.labels

        self._title = labels.get_plot_title(substitutions=self._latex_substitutions)
        self._xlabel = self._output.dd.axis_label(0) or labels.xaxis_unit or "Index [#]"
        self._ylabel = labels.axis_unit

        if self._output.dd.dim == 2:
            method = self._plotoptions.get("method")

            self._zlabel = self._ylabel
            if method == "slicesx":
                self._xlabel = self._output.dd.axis_label(1) or labels.yaxis_unit or "Index [#]"
                self._zlabel = self._output.dd.axis_label(0) or labels.xaxis_unit or "Index [#]"
            elif method == "slicesy":
                self._zlabel = self._output.dd.axis_label(1) or "Index [#]"
            else:
                self._zlabel = self._ylabel
                self._ylabel = self._output.dd.axis_label(1) or labels.yaxis_unit or "Index [#]"

    def annotate_axes(
        self,
        /,
        ax: Axes | None = None,
        *,
        legend: bool = False,
        show_path: bool = True,
    ) -> None:
        ax = ax or plt.gca()
        labels = self._output.labels

        if self._title and not ax.get_title():
            if self._array.ndim > 1:
                nlines = self._title.count("\n") + 1
                fontsize = nlines > 1 and "x-small" or "medium"
            else:
                fontsize = "medium"

            ax.set_title(self._title, size=fontsize)

        if self._xlabel and not ax.get_xlabel():
            ax.set_xlabel(self._xlabel)
        if self._ylabel and not ax.get_ylabel():
            ax.set_ylabel(self._ylabel)
        try:
            prev_zlabel = ax.get_zlabel()
        except Exception:
            prev_zlabel = ""
        if self._zlabel and prev_zlabel:
            with suppress(AttributeError):
                ax.set_zlabel(self._zlabel)

        if aspect := self._plotoptions.get("aspect"):
            ax.set_aspect(aspect)

        if xscale := self._plotoptions.get("xscale"):
            ax.set_xscale(xscale)

        if yscale := self._plotoptions.get("yscale"):
            ax.set_yscale(yscale)

        if legend:
            ax.legend()

        if (plot_diagonal := self._plotoptions.get("plot_diagonal")) is not None:
            xlim = ax.get_xlim()
            ax.plot(xlim, xlim, **plot_diagonal)

        if subplots_adjust_kw := self._plotoptions.get("subplots_adjust"):
            plt.subplots_adjust(**subplots_adjust_kw)

        if show_path:
            path = labels.paths
            if not path:
                return

            fig = plt.gcf()
            try:
                ax.text2D(0.02, 0.02, path[0], transform=fig.dpi_scale_trans, fontsize="small")
            except AttributeError:
                ax.text(
                    0.02,
                    0.02,
                    path[0],
                    transform=fig.dpi_scale_trans,
                    fontsize="x-small",
                )

        if self._plotoptions.get("show"):
            plt.show()

    @property
    def zlabel(self) -> str | None:
        return self._zlabel


def plot_array_1d(
    array: NDArray,
    edges: NDArray | None = None,
    meshes: NDArray | None = None,
    yerr: float | NDArray | None = None,
    xerr: float | NDArray | None = None,
    *args,
    **kwargs,
) -> tuple[tuple, ...]:
    if edges is not None:
        if yerr is not None or xerr is not None:
            if xerr is None:
                xerr = 0.5 * (edges[1:] - edges[:-1])
            return plot_array_1d_errors(
                0.5 * (edges[1:] + edges[:-1]),
                array,
                yerr=yerr,
                xerr=xerr,
                *args,
                **kwargs,
            )
        return plot_array_1d_hist(array, edges, *args, **kwargs)
    elif meshes is not None:
        if yerr is not None or xerr is not None:
            return plot_array_1d_errors(meshes, array, yerr=yerr, xerr=xerr, *args, **kwargs)
        return plot_array_1d_vs(array, meshes, *args, **kwargs)
    else:
        return plot_array_1d_array(array, *args, **kwargs)


def plot_array_1d_hist(
    array: NDArray,
    edges: NDArray | None = None,
    *args,
    plotter: plot_auto | None = None,
    **kwargs,
) -> tuple:
    return plt.stairs(array, edges, *args, **kwargs)


def plot_array_1d_errors(
    x: NDArray,
    y: NDArray,
    yerr: float | NDArray | None = None,
    xerr: float | NDArray | None = None,
    *args,
    plotter: plot_auto | None = None,
    **kwargs,
) -> tuple:
    return plt.errorbar(x, y, yerr=yerr, xerr=xerr, *args, **kwargs)


def plot_array_1d_vs(
    array: NDArray,
    meshes: NDArray | None = None,
    *args,
    plotter: plot_auto | None = None,
    **kwargs,
) -> tuple:
    return plt.plot(meshes, array, *args, **kwargs)


def plot_array_1d_array(array: NDArray, *args, plotter: plot_auto | None = None, **kwargs) -> tuple:
    return plt.plot(array, *args, **kwargs)


def plot_array_2d(
    array: NDArray,
    edges: tuple[NDArray, ...],
    meshes: tuple[NDArray, ...],
    *args,
    plotoptions: Mapping[str, Any] = {},
    **kwargs,
) -> tuple[tuple, ...]:
    if edges:
        return plot_array_2d_hist(array, edges, *args, plotoptions=plotoptions, **kwargs)
    elif meshes:
        return plot_array_2d_vs(array, meshes, *args, plotoptions=plotoptions, **kwargs)
    else:
        return plot_array_2d_array(array, *args, **kwargs)


def plot_array_2d_array(array: NDArray, *args, plotter: plot_auto | None = None, **kwargs) -> tuple:
    kwargs.setdefault("aspect", "auto")
    return plot_array_2d_hist_matshow(array, None, *args, **kwargs)


def plot_array_2d_hist_bar3d(
    dZ: NDArray,
    edges: tuple[NDArray, ...],
    *args,
    plotter: plot_auto | None = None,
    cmap: str | None = None,
    **kwargs,
) -> tuple:
    xedges, yedges = edges
    xw = xedges[1:] - xedges[:-1]
    yw = yedges[1:] - yedges[:-1]

    X, Y = meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    X, Y = X.ravel(), Y.ravel()

    dX, dY = meshgrid(xw, yw, indexing="ij")
    dX, dY, dZ = dX.ravel(), dY.ravel(), dZ.ravel()
    Z = zeros_like(dZ)

    apply_colors(dZ, cmap, kwargs, "color")
    colorbar = kwargs.pop("colorbar", False)

    ax = plt.gca()
    res = ax.bar3d(X, Y, Z, dX, dY, dZ, *args, **kwargs)

    return _colorbar_or_not_3d(res, colorbar, dZ)


def plot_array_2d_hist_pcolorfast(
    Z: NDArray,
    edges: tuple[NDArray, ...],
    *args,
    plotter: plot_auto | None = None,
    **kwargs,
) -> tuple:
    xedges, yedges = edges
    return pcolorfast(xedges, yedges, Z.T, *args, **kwargs)


def plot_array_2d_hist_pcolormesh(
    Z: NDArray,
    edges: tuple[NDArray, ...],
    *args,
    plotter: plot_auto | None = None,
    **kwargs,
) -> tuple:
    x, y = meshgrid(edges[0], edges[1], indexing="ij")
    return pcolormesh(x, y, Z, *args, **kwargs)


def plot_array_2d_hist_pcolor(
    Z: NDArray,
    edges: tuple[NDArray, ...],
    *args,
    plotter: plot_auto | None = None,
    **kwargs,
) -> tuple:
    x, y = meshgrid(edges[0], edges[1], indexing="ij")
    return pcolor(x, y, Z, *args, **kwargs)


def plot_array_2d_hist_imshow(
    Z: NDArray,
    edges: tuple[NDArray, ...] | None = None,
    *args,
    plotter: plot_auto | None = None,
    **kwargs,
):
    if edges:
        xedges, yedges = edges
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        kwargs.setdefault("extent", extent)
    kwargs.setdefault("origin", "lower")
    return imshow(Z.T, *args, **kwargs)


def plot_array_2d_hist_matshow(
    Z: NDArray,
    edges: tuple[NDArray, ...] | None = None,
    *args,
    plotter: plot_auto | None = None,
    **kwargs,
):
    kwargs.setdefault("fignum", False)
    if edges:
        xedges, yedges = edges
        extent = [xedges[0], xedges[-1], yedges[-1], yedges[0]]
        kwargs.setdefault("extent", extent)
    return matshow(Z.T, *args, **kwargs)


plot_array_2d_hist_methods = {
    "pcolor": plot_array_2d_hist_pcolor,
    "pcolorfast": plot_array_2d_hist_pcolorfast,
    "pcolormesh": plot_array_2d_hist_pcolormesh,
    "imshow": plot_array_2d_hist_imshow,
    "matshow": plot_array_2d_hist_matshow,
    "bar3d": plot_array_2d_hist_bar3d,
}


def plot_array_2d_hist(
    dZ: NDArray,
    edges: tuple[NDArray, ...],
    *args,
    plotoptions: Mapping[str, Any] = {},
    **kwargs,
) -> tuple:
    smethod: str = (
        "pcolormesh" if (method := plotoptions.get("method", "auto")) == "auto" else method
    )
    try:
        function = plot_array_2d_hist_methods[smethod]
    except KeyError as e:
        raise RuntimeError(f"Invlid 2d hist plotoptions: {plotoptions}") from e

    return function(dZ, edges, *args, **kwargs)


def plot_array_2d_vs_pcolormesh(
    Z: NDArray,
    meshes: tuple[NDArray, ...],
    *args,
    plotter: plot_auto | None = None,
    **kwargs,
) -> tuple:
    x, y = meshes
    kwargs.setdefault("shading", "nearest")
    return pcolormesh(x, y, Z, *args, **kwargs)


def plot_array_2d_vs_pcolor(
    Z: NDArray,
    meshes: tuple[NDArray, ...],
    *args,
    plotter: plot_auto | None = None,
    **kwargs,
) -> tuple:
    x, y = meshes
    kwargs.setdefault("shading", "nearest")
    return pcolormesh(x, y, Z, *args, **kwargs)


def plot_array_2d_vs_slicesx(
    Z: NDArray,
    meshes: tuple[NDArray, ...],
    *args,
    plotter: plot_auto | None = None,
    **kwargs,
):
    x, y = meshes
    haslabels = False
    zlabel = plotter and plotter.zlabel or "value"

    legtitle = f"Slice {zlabel}:"
    for data, slicey, slicex in zip(Z, x, y):
        if (slicey[0] == slicey).all():
            label = f"${slicey[0]:.2g}$"
            haslabels = True
        else:
            label = None
        plt.plot(slicex, data, label=label)

    if haslabels is not None:
        ax = plt.gca()
        ax.legend(title=legtitle)


def plot_array_2d_vs_slicesy(Z: NDArray, meshes: tuple[NDArray, ...], *args, **kwargs):
    return plot_array_2d_vs_slicesx(Z.T, tuple(mesh.T for mesh in reversed(meshes)), *args, **kwargs)


def plot_array_2d_vs_surface(
    Z: NDArray,
    meshes: tuple[NDArray, ...],
    *args,
    plotter: plot_auto | None = None,
    **kwargs,
) -> tuple:
    return plot_surface(meshes[0], meshes[1], Z, *args, **kwargs)


def plot_array_2d_vs_wireframe(
    Z: NDArray,
    meshes: tuple[NDArray, ...],
    *args,
    # facecolors: str | None = None,
    plotter: plot_auto | None = None,
    cmap: str | bool | None = None,
    colorbar: dict | bool = False,
    **kwargs,
) -> tuple:
    X, Y = meshes

    ax = plt.gca()
    if cmap is not None:
        colors, _ = apply_colors(Z, cmap, kwargs, "facecolors")
        if colors is not None:
            kwargs["rcount"] = Z.shape[0]
            kwargs["ccount"] = Z.shape[1]
            kwargs["shade"] = False
            res = ax.plot_surface(X, Y, Z, **kwargs)
            res.set_facecolor((0, 0, 0, 0))

            return _colorbar_or_not_3d(res, colorbar, Z)

    return ax.plot_wireframe(X, Y, Z, *args, **kwargs)


plot_array_2d_vs_methods = {
    "surface": plot_array_2d_vs_surface,
    "wireframe": plot_array_2d_vs_wireframe,
    "pcolormesh": plot_array_2d_vs_pcolormesh,
    "pcolor": plot_array_2d_vs_pcolor,
    "slicesx": plot_array_2d_vs_slicesx,
    "slicesy": plot_array_2d_vs_slicesy,
}


def plot_array_2d_vs(
    array: NDArray,
    meshes: tuple[NDArray, ...],
    *args,
    plotoptions: Mapping[str, Any] = {},
    **kwargs,
) -> tuple:
    smethod: str = (
        "pcolormesh" if (method := plotoptions.get("method", "auto")) == "auto" else method
    )  # pyright: ignore [reportGeneralTypeIssues]
    try:
        function = plot_array_2d_vs_methods[smethod]
    except KeyError as e:
        raise RuntimeError("unimplemented") from e

    return function(array, meshes, *args, **kwargs)


def _mask_if_needed(datain: ArrayLike, /, *, masked_value: float | None = None) -> NDArray:
    data = asanyarray(datain)
    if masked_value is None:
        return data

    mask = data == masked_value
    return masked_array(data, mask=mask)


def _patch_with_colorbar(function, mode3d=False):
    """Patch pyplot.function or ax.plotoptions by adding a "colorbar"
    option."""
    returner = mode3d and _colorbar_or_not_3d or _colorbar_or_not
    if isinstance(function, str):

        def newfcn(
            *args,
            cmap: bool | str | None = None,
            colorbar: bool | None = None,
            **kwargs,
        ):
            ax = plt.gca()
            actual_fcn = getattr(ax, function)
            kwargs["cmap"] = cmap is True and "viridis" or cmap
            res = actual_fcn(*args, **kwargs)
            return returner(res, colorbar)

    else:

        def newfcn(
            *args,
            cmap: bool | str | None = None,
            colorbar: bool | None = None,
            **kwargs,
        ):
            kwargs["cmap"] = cmap is True and "viridis" or cmap
            res = function(*args, **kwargs)
            return returner(res, colorbar)

    return newfcn


def apply_colors(buf: NDArray, cmap: str | bool | None, kwargs: dict, colorsname: str) -> tuple:
    if cmap is True:
        cmap = "viridis"
    elif not cmap:
        return None, None

    bmin, bmax = buf.min(), buf.max()
    norm = (buf - bmin) / (bmax - bmin)

    cmap = colormaps.get_cmap(cmap)
    res = cmap(norm)
    kwargs[colorsname] = res
    return res, cmap


def add_colorbar(
    colormapable,
    rasterized: bool = True,
    minorticks: bool = False,
    minorticks_values: NDArray | None = None,
    label: str | None = None,
    **kwargs,
):
    """Add a colorbar to the axis with height aligned to the axis."""
    ax = plt.gca()
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.gcf().colorbar(colormapable, cax=cax, **kwargs)

    if minorticks:
        if isinstance(minorticks, str):
            if minorticks == "log":
                minorticks_values = colormapable.norm(minorticks_values)
            # elif minorticks=='linear':
            #     pass

            l1, l2 = cax.get_ylim()
            minorticks_values = minorticks_values[
                (minorticks_values >= l1) * (minorticks_values <= l2)
            ]
            cax.yaxis.set_ticks(minorticks_values, minor=True)
        else:
            cax.minorticks_on()

    if rasterized:
        cbar.solids.set_rasterized(True)

    if label is not None:
        cbar.set_label(label, rotation=270, labelpad=15)
    plt.sca(ax)
    return cbar


def add_colorbar_3d(res, cbaropt: dict = {}, mappable=None):
    """Add a colorbar to the 3d axis with height aligned to the axis."""
    cbaropt.setdefault("aspect", 4)
    cbaropt.setdefault("shrink", 0.5)

    if mappable is None:
        cbar = plt.colorbar(res, **cbaropt)
    else:
        colourMap = plt.cm.ScalarMappable()
        colourMap.set_array(mappable)
        cbar = plt.colorbar(colourMap, **cbaropt)

    return res, cbar


def _colorbar_or_not(res, cbaropt: Mapping | bool | None):
    if not cbaropt:
        return res

    if not isinstance(cbaropt, Mapping):
        cbaropt = {}

    cbar = add_colorbar(res, **cbaropt)

    return res, cbar


def _colorbar_or_not_3d(res, cbaropt: Mapping | bool | None, mappable=None):
    if not cbaropt:
        return res

    if not isinstance(cbaropt, Mapping):
        cbaropt = {}

    cbaropt.setdefault("aspect", 4)
    cbaropt.setdefault("shrink", 0.5)

    if mappable is None:
        cbar = plt.colorbar(res, ax=plt.gca(), **cbaropt)
    else:
        colourMap = plt.cm.ScalarMappable()
        colourMap.set_array(mappable)
        cbar = plt.colorbar(colourMap, ax=plt.gca(), **cbaropt)

    return res, cbar


pcolorfast = _patch_with_colorbar("pcolorfast")
pcolor = _patch_with_colorbar(plt.pcolor)
pcolormesh = _patch_with_colorbar(plt.pcolormesh)
imshow = _patch_with_colorbar(plt.imshow)
matshow = _patch_with_colorbar(plt.matshow)
plot_surface = _patch_with_colorbar("plot_surface", mode3d=True)
