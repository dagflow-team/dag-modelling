from __future__ import annotations

from collections.abc import Sequence

from ..core.exception import InitializationError


class LimbNameFormatter:
    __slots__ = ()

    def format(self, num: int) -> str:
        raise RuntimeError("Virtual method called")

    @staticmethod
    def from_string(string: str):
        return string if "{" in string else SimpleLimbNameFormatter(string)

    @staticmethod
    def from_sequence(seq: Sequence[str]):
        return SequentialLimbNameFormatter(seq)

    @staticmethod
    def from_value(value: str | Sequence[str] | LimbNameFormatter):
        if isinstance(value, LimbNameFormatter):
            return value
        elif isinstance(value, str):
            return LimbNameFormatter.from_string(value)
        elif isinstance(value, Sequence):
            return LimbNameFormatter.from_sequence(value)

        raise InitializationError(
            f"Expect str, Tuple[str] or LimbNameFormatter, got {type(value).__name__}"
        )


Formattable = LimbNameFormatter | str


class SimpleLimbNameFormatter(LimbNameFormatter):
    __slots__ = ("_base", "_numfmt")
    _base: str
    _numfmt: str

    def __init__(self, base: str, numfmt: str = "_{:02d}"):
        self._base = base
        self._numfmt = numfmt

    def format(self, num: int) -> str:
        return self._base + self._numfmt.format(num) if num > 0 else self._base


class SequentialLimbNameFormatter(LimbNameFormatter):
    __slots__ = ("_base", "_numfmt", "_startidx")

    _base: tuple[str, ...]
    _numfmt: str
    _startidx: int

    def __init__(self, base: Sequence[str], numfmt: str = "_{:02d}", startidx: int = 0):
        self._base = tuple(base)
        self._numfmt = numfmt
        self._startidx = startidx

    def format(self, num: int) -> str:
        num -= self._startidx
        idx = num % len(self._base)
        groupnum = num // len(self._base)
        base = self._base[idx]
        if groupnum > 0:
            return base + self._numfmt.format(groupnum)
        elif num < 0:
            raise ValueError(f"SequentialLimbNameFormatter got num={num}<0")

        return base
