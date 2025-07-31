from numpy import arccos, arcsin, arctan, cos, sin, tan

from .abstract import OneToOneNode


class Cos(OneToOneNode):
    """Cos function"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "cos")

    def _function(self):
        for callback in self._input_nodes_callbacks:
            callback()

        for input_data, output_data in self._input_output_data:
            cos(input_data, out=output_data)


class Sin(OneToOneNode):
    """Sin function"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "sin")

    def _function(self):
        for callback in self._input_nodes_callbacks:
            callback()

        for input_data, output_data in self._input_output_data:
            sin(input_data, out=output_data)


class ArcCos(OneToOneNode):
    """ArcCos function"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "acos")

    def _function(self):
        for callback in self._input_nodes_callbacks:
            callback()

        for input_data, output_data in self._input_output_data:
            arccos(input_data, out=output_data)


class ArcSin(OneToOneNode):
    """ArcSin function"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "asin")

    def _function(self):
        for callback in self._input_nodes_callbacks:
            callback()

        for input_data, output_data in self._input_output_data:
            arcsin(input_data, out=output_data)


class Tan(OneToOneNode):
    """Tan function"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "tan")

    def _function(self):
        for callback in self._input_nodes_callbacks:
            callback()

        for input_data, output_data in self._input_output_data:
            tan(input_data, out=output_data)


class ArcTan(OneToOneNode):
    """Arctan function"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "atan")

    def _function(self):
        for callback in self._input_nodes_callbacks:
            callback()

        for input_data, output_data in self._input_output_data:
            arctan(input_data, out=output_data)
