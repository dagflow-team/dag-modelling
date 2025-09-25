from numpy import abs, add, copyto, divide, multiply, sqrt, square, subtract

from .abstract import ManyToOneNode, OneToOneNode


class Sum(ManyToOneNode):
    """Sum of all the inputs together."""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("broadcastable", True)
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "Σᵢ")

    def _function(self):
        for callback in self._input_nodes_callbacks:
            callback()

        output_data = self._output_data
        copyto(output_data, self._input_data0)
        for input_data in self._input_data_other:
            add(output_data, input_data, out=output_data)


class Difference(ManyToOneNode):
    """Difference of inputs: A-B-C-..."""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("broadcastable", True)
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "a₀-Σᵢ")

    def _function(self):
        for callback in self._input_nodes_callbacks:
            callback()

        output_data = self._output_data
        copyto(output_data, self._input_data0)
        for input_data in self._input_data_other:
            subtract(output_data, input_data, out=output_data)


class Product(ManyToOneNode):
    """Product of all the inputs together."""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("broadcastable", True)
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "Πᵢ")

    def _function(self):
        for callback in self._input_nodes_callbacks:
            callback()

        output_data = self._output_data
        copyto(output_data, self._input_data0)
        for _input_data in self._input_data_other:
            multiply(output_data, _input_data, out=output_data)


class ProductShifted(ManyToOneNode):
    """Product of all the inputs together, shifted by a constant."""

    __slots__ = ("_shift",)

    _shift: float

    def __init__(self, *args, shift: float | int, **kwargs):
        kwargs.setdefault("broadcastable", True)
        super().__init__(*args, **kwargs)

        self._shift = float(shift)
        if shift % 1 == 0:
            self._labels.setdefault("mark", f"{int(self._shift)}+Πᵢ")
        else:
            self._labels.setdefault("mark", f"{self._shift}+Πᵢ")

    def _function(self):
        for callback in self._input_nodes_callbacks:
            callback()

        output_data = self._output_data
        copyto(output_data, self._input_data0)
        for _input_data in self._input_data_other:
            multiply(output_data, _input_data, out=output_data)
        add(output_data, self._shift, out=output_data)


class ProductShiftedScaled(ManyToOneNode):
    """Product of all the inputs together, shifted by a constant."""

    __slots__ = ("_shift",)

    _shift: float

    def __init__(self, *args, shift: float | int, **kwargs):
        kwargs.setdefault("broadcastable", True)
        super().__init__(*args, **kwargs)

        self._shift = float(shift)
        if shift % 1 == 0:
            self._labels.setdefault("mark", f"a·({int(self._shift)}+Πᵢ)")
        else:
            self._labels.setdefault("mark", f"a·({self._shift}+Πᵢ)")

    def _function(self):
        for callback in self._input_nodes_callbacks:
            callback()

        output_data = self._output_data
        copyto(output_data, self._input_data_other[0])
        for _input_data in self._input_data_other[1:]:
            multiply(output_data, _input_data, out=output_data)
        add(output_data, self._shift, out=output_data)

        multiply(output_data, self._input_data0, out=output_data)


class Division(ManyToOneNode):
    """Division of the first input to other.

    .. note:: a division by zero returns `nan`
    """

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("broadcastable", True)
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "÷")

    def _function(self):
        for callback in self._input_nodes_callbacks:
            callback()

        output_data = self._output_data
        copyto(output_data, self._input_data0)
        for _input_data in self._input_data_other:
            divide(self._output_data, _input_data, out=self._output_data)


class Square(OneToOneNode):
    """Square function."""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "x²")

    def _function(self):
        for callback in self._input_nodes_callbacks:
            callback()

        for input_data, output_data in self._input_output_data:
            square(input_data, out=output_data)


class Sqrt(OneToOneNode):
    """Square function."""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "√x")

    def _function(self):
        for callback in self._input_nodes_callbacks:
            callback()

        for input_data, output_data in self._input_output_data:
            sqrt(input_data, out=output_data)


class Abs(OneToOneNode):
    """Modulus function."""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "|x|")

    def _function(self):
        for callback in self._input_nodes_callbacks:
            callback()

        for input_data, output_data in self._input_output_data:
            abs(input_data, out=output_data)
