from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from numpy import empty, errstate, integer, linspace, matmul, meshgrid, newaxis
from numpy.polynomial.legendre import leggauss

from ...core.exception import (
    CalculationError,
    CriticalError,
    InitializationError,
    UnclosedGraphError,
)
from ...core.node import Node
from ...core.type_functions import (
    check_dimension_of_inputs,
    check_edges_dimension_of_inputs,
    check_number_of_inputs,
    check_subtype_of_inputs,
)

if TYPE_CHECKING:
    from numpy.typing import DTypeLike, NDArray

    from ...core.input import Input
    from ...core.output import Output

ModeType = Literal["rect", "trap", "gl", "gl2d"]


# TODO: without njit?
def _gl_sampler(orders: NDArray, sample: NDArray, weights: NDArray, edges: NDArray):
    """Uses `numpy.polynomial.legendre.leggauss` to sample points with weights
    on the range [-1,1] and transforms to any range [a, b]"""
    offset = 0
    for i, n in enumerate(orders):
        if n < 1:
            continue
        (
            sample[offset : offset + n],
            weights[offset : offset + n],
        ) = leggauss(n)
        # transforms to the original range [a, b]
        sample[offset : offset + n] = 0.5 * (
            sample[offset : offset + n] * (edges[i + 1] - edges[i]) + (edges[i + 1] + edges[i])
        )
        weights[offset : offset + n] *= 0.5 * (edges[i + 1] - edges[i])
        # NOTE: the operations above may allocate additional memory in runtime!
        offset += n


class IntegratorSampler(Node):
    """The `IntegratorSampler` node creates a sample for the `IntegratorCore`
    node.

    There are several samplers for `1d` (`rect`, `trap`, `gl`) and only `gl2d`
    for `2d` integrator, where `rect` is the rectangular, `trap` is the trapezoidal,
    `gl` is the 1d Gauss-Legendre, and `gl2d` is the 2d Gauss-Legendre.

    There is optional argument `offset` for the `rect` sampler,
    taking the following values: `left`, `center`, or `right`.

    There is no positional self.inputs. It is supposed that `orders` already have `edges`.
    There are two self.outputs: 0 - `sample`, 1 - `weights`
    """

    __slots__ = (
        "__bufferX",
        "__bufferY",
        "_dtype",
        "_mode",
        "_ndim",
        "_align",
        "_orders_x",
        "_orders_y",
        "_weights",
        "_x",
        "_y",
    )

    _dtype: DTypeLike
    _mode: ModeType
    _ndim: int
    _align: Literal["left", "center", "right"] | None
    __bufferX: NDArray
    __bufferY: NDArray
    _orders_x: Input
    _orders_y: Input
    _weights: Output
    _x: Output
    _y: Output

    def __init__(
        self,
        *args,
        mode: ModeType,
        dtype: DTypeLike = "d",
        align: Literal["left", "center", "right"] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs, allowed_kw_inputs=("orders_x", "orders_y"))
        if mode not in {"rect", "trap", "gl", "gl2d"}:
            raise InitializationError(
                f"Argument `mode` must be 'rect', 'trap', 'gl', or '2d', but given '{mode}'!",
                node=self,
            )
        if align is not None and mode != "rect":
            raise InitializationError(
                "Argument 'align' is used only within 'rect' mode!", node=self
            )
        self._dtype = dtype
        self._mode = mode
        self._align = align if align is not None else "center"
        self._orders_x = self._add_input("orders_x", positional=False)
        self._x = self._add_output("x")
        if mode == "gl2d":
            self._orders_y = self._add_input("orders_y", positional=False)
            self._y = self._add_output("y")
            self._ndim = 2
        else:
            self._ndim = 1
        self._weights = self._add_output("weights", positional=False)
        self._functions_dict.update(
            {
                "rect": self._fcn_rect,
                "trap": self._fcn_trap,
                "gl": self._fcn_gl1d,
                "gl2d": self._fcn_gl2d,
            }
        )

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def dtype(self) -> DTypeLike:
        return self._dtype

    @property
    def align(self) -> str | None:
        return self._align

    def taint(self, *, caller: Input | None = None, **kwargs):
        if caller is not None and (caller is self._orders_x or caller is self._orders_y):
            raise CriticalError(
                "IntegratorSampler: can not change integration orders without reopening graph",
                node=self,
                input=caller,
            )
        super().taint(caller=caller, **kwargs)

    def _type_function(self) -> None:
        """The function to determine the dtype and shape.

        Checks self.inputs dimension and, selects an integration
        algorithm, determines dtype and shape for self.outputs
        """
        check_number_of_inputs(self, 0)
        lenX = self.__check_orders("orders_x")
        if self.mode == "gl2d":
            lenY = self.__check_orders("orders_y")
            shape = (lenX, lenY)
        else:
            shape = (lenX,)
        for output in (*self.outputs, self._weights):
            output.dd.dtype = self.dtype
            output.dd.shape = shape
        self.function = self._functions_dict[self.mode]

    def __check_orders(self, name: str) -> int:
        """The method checks dimension (==1) of the input `name`, type
        (==`integer`), and returns the `dd.shape[0]`"""
        check_dimension_of_inputs(self, name, 1)
        check_subtype_of_inputs(self, name, dtype=integer)
        check_edges_dimension_of_inputs(self, name, 1)
        orders = self.inputs[name]
        try:
            npoints = sum(orders.data)
        except UnclosedGraphError:
            raise CalculationError(
                "Orders for IntegratorSampler should be available (closed) before the graph is closed",
                node=self,
                input=orders,
            )

        return npoints

    def _post_allocate(self) -> None:
        """Allocates the `buffer`"""
        orders_x = self._orders_x
        edgeshapeX = orders_x.dd.axes_edges[0].dd.shape[0] - 1
        if self.mode == "rect":
            shapeX = (4, edgeshapeX)
        elif self.mode in {"trap", "gl"}:
            shapeX = (edgeshapeX,)
        else:
            lenY = sum(self._orders_y.data)
            shapeY = (2, lenY)
            self.__bufferY = empty(shape=shapeY, dtype=self.dtype)
            lenX = sum(orders_x.data)
            shapeX = (2, lenX)
        self.__bufferX = empty(shape=shapeX, dtype=self.dtype)

    def _fcn_rect(self):
        """The rectangular sampling."""
        orders_x = self._orders_x
        edges = orders_x.dd.axes_edges[0].data  # n+1
        orders = orders_x.data  # n
        sample = self.outputs[0]._data  # m = sum(orders)
        weights = self._weights._data
        binwidths = self.__bufferX[0]  # n
        samplewidths = self.__bufferX[1]  # n
        low = self.__bufferX[2]  # n
        high = self.__bufferX[3]  # n

        binwidths[:] = edges[1:] - edges[:-1]
        with errstate(invalid="ignore"):  # to ignore division by zero
            samplewidths[:] = binwidths / orders
        if self.align == "left":
            low[:] = edges[:-1]
            high[:] = edges[1:] - samplewidths
        elif self.align == "center":
            low[:] = edges[:-1] + samplewidths * 0.5
            high[:] = edges[1:] - samplewidths * 0.5
        else:
            low[:] = edges[:-1] + samplewidths
            high[:] = edges[1:]

        offset = 0
        for i, n in enumerate(orders):
            if n > 1:
                sample[offset : offset + n] = linspace(low[i], high[i], n)
                weights[offset : offset + n] = samplewidths[i]
            else:
                sample[offset : offset + n] = low[i]
                weights[offset : offset + n] = binwidths[i]
            offset += n

    def _fcn_trap(self):
        """The trapezoidal sampling."""
        orders_x = self._orders_x
        edges = orders_x.dd.axes_edges[0].data  # n+1
        orders = orders_x.data  # n
        sample = self.outputs[0]._data  # m = sum(orders)
        weights = self._weights._data
        samplewidths = self.__bufferX  # n

        samplewidths[:] = edges[1:] - edges[:-1]
        with errstate(invalid="ignore"):  # to ignore division by zero
            samplewidths[:] = samplewidths[:] / (orders - 2.0)

        offset = 0
        for i, n in enumerate(orders):
            sample[offset : offset + n] = linspace(edges[i], edges[i + 1], n)
            weights[offset] = samplewidths[i] * 0.5
            if n > 2:
                weights[offset + 1 : offset + n - 2] = samplewidths[i]
            offset += n - 1
        weights[-1] = samplewidths[-1] * 0.5

    def _fcn_gl1d(self):
        """The 1d Gauss-Legendre sampling."""
        orders_x = self._orders_x
        edges = orders_x.dd.axes_edges[0].data
        orders = orders_x.data
        sample = self.outputs[0]._data
        weights = self._weights._data

        _gl_sampler(orders, sample, weights, edges)

    def _fcn_gl2d(self):
        """The 2d Gauss-Legendre sampling."""
        orders_x = self._orders_x
        orders_y = self._orders_y
        edgesX = orders_x.dd.axes_edges[0].data  # p + 1
        edgesY = orders_y.dd.axes_edges[0].data  # q + 1
        orders_x = orders_x.data
        orders_y = orders_y.data
        weightsX = self.__bufferX[0]  # (n, )
        weightsY = self.__bufferY[0]  # (m, )
        sampleX = self.__bufferX[1]  # (n, )
        sampleY = self.__bufferY[1]  # (m, )
        X = self.outputs[0]._data  # (n, m)
        Y = self.outputs[1]._data  # (n, m)
        weights = self._weights._data  # (n, m)

        _gl_sampler(orders_x, sampleX, weightsX, edgesX)
        _gl_sampler(orders_y, sampleY, weightsY, edgesY)

        X[:], Y[:] = meshgrid(sampleX, sampleY, indexing="ij")
        matmul(weightsX[newaxis].T, weightsY[newaxis], out=weights)
