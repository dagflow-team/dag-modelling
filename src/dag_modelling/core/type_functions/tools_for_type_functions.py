from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

AllPositionals = slice(None)
LimbKey = str | int | slice | Sequence

try:
    zip((), (), strict=True)
    zip_dag = zip
except TypeError:
    # provide a replacement of strict zip from Python 3.1
    # to be deprecated at some point
    from itertools import zip_longest

    def _zip(*iterables, strict: bool = False):
        sentinel = object()
        for combo in zip_longest(*iterables, fillvalue=sentinel):
            if strict and sentinel in combo:
                raise ValueError("Iterables have different lengths")
            yield combo
    zip_dag = _zip


class MethodSequenceCaller:
    """Class to call a sequence of methods"""

    __slots__ = ("methods",)

    def __init__(self) -> None:
        self.methods = []

    def __call__(self, inputs, outputs) -> None:
        for method in self.methods:
            method(inputs, outputs)

    def add(self, method: Callable) -> None:
        self.methods.append(method)


def copy_dtype(input, output):
    output.dd.dtype = input.dd.dtype


def copy_shape(input, output):
    output.dd.shape = input.dd.shape


def copy_edges(input, output):
    output.dd.axes_edges = input.dd.axes_edges


def copy_meshes(input, output):
    output.dd.axes_meshes = input.dd.axes_meshes


def shapes_are_broadcastable(shape1: Sequence[int], shape2: Sequence[int]) -> bool:
    return all(a == 1 or b == 1 or a == b for a, b in zip_dag(shape1[::-1], shape2[::-1]))
