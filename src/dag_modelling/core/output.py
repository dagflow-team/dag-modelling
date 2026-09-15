from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.labels import repr_pretty
from .edges import EdgeContainer
from .output_methods import _OutputMethods

if TYPE_CHECKING:
    from .input import Input


class Output(_OutputMethods):
    __slots__ = (
        "_data",
        "_data_ro",
        "_dd",
        "_node",
        "_name",
        "_child_inputs",
        "_parent_input",
        "_allocating_input",
        "_allocatable",
        "_owns_buffer",
        "_forbid_reallocation",
        "_labels",
        "_debug",
    )

    def touch(self, force_computation: bool = False):
        return self._node.touch(force_computation=force_computation)

    def touch_above_threshold(self, taint_threshold: int, force_computation: bool = False) -> None:
        self._node.touch_above_threshold(taint_threshold, force_computation=force_computation)

    def taint_children(
        self,
        *,
        force_taint: bool = False,
        force_computation: bool = False,
        caller: "Input | None" = None,
    ) -> None:
        for input in self._child_inputs:
            input.taint(
                force_taint=force_taint,
                force_computation=force_computation,
                caller=caller,
            )

    def taint_children_below_threshold(
        self,
        taint_threshold: int = 0,
        *,
        force_taint: bool = False,
        force_computation: bool = False,
        caller: Input | None = None,
    ) -> None:
        for child_input in self._child_inputs:
            child_input.taint_below_threshold(
                taint_threshold,
                force_taint=force_taint,
                force_computation=force_computation,
                caller=caller,
            )

class Outputs(EdgeContainer):
    __slots__ = ()

    def __init__(self, iterable=None) -> None:
        super().__init__(iterable)
        self._dtype = Output

    def __str__(self) -> str:
        return f"○[{tuple(obj.name for obj in self)}]→"

    _repr_pretty_ = repr_pretty
