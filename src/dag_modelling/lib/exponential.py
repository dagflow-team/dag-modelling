from numpy import exp, expm1, log, log1p, log10

from .abstract import OneToOneNode


class Exp(OneToOneNode):
    """exp(x) function"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "exp")

    def _function(self):
        for indata, outdata in zip(self.inputs.iter_data(), self.outputs.iter_data_unsafe()):
            exp(indata, out=outdata)


class Expm1(OneToOneNode):
    """exp(x)-1 function"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "exp-1")

    def _function(self):
        for indata, outdata in zip(self.inputs.iter_data(), self.outputs.iter_data_unsafe()):
            expm1(indata, out=outdata)


class Log(OneToOneNode):
    """log(x) function"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "log")

    def _function(self):
        for indata, outdata in zip(self.inputs.iter_data(), self.outputs.iter_data_unsafe()):
            log(indata, out=outdata)


class Log1p(OneToOneNode):
    """log(x+1) function"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "log(x+1)")

    def _function(self):
        for indata, outdata in zip(self.inputs.iter_data(), self.outputs.iter_data_unsafe()):
            log1p(indata, out=outdata)


class Log10(OneToOneNode):
    """log10(x) function"""

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "log₁₀")

    def _function(self):
        for indata, outdata in zip(self.inputs.iter_data(), self.outputs.iter_data_unsafe()):
            log10(indata, out=outdata)
