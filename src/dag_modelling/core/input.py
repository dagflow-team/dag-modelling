from __future__ import annotations

from ..core.labels import repr_pretty
from .edges import EdgeContainer
from .input_methods import _InputMethods


class Input(_InputMethods):
    __slots__ = (
        "_own_data",
        "_own_dd",
        "_node",
        "_name",
        "_parent_output",
        "_child_output",
        "_allocatable",
        "_owns_buffer",
        "_debug",
    )

    def touch_above_threshold(self, taint_threshold: int):
        self._parent_output.touch_above_threshold(taint_threshold)

    def taint(
        self,
        *,
        force_taint: bool = False,
        force_computation: bool = False,
        caller: "Input | None" = None,
    ):
        self._node.taint(
            force_taint=force_taint,
            force_computation=force_computation,
            caller=self if caller is None else caller,
        )

    def taint_below_threshold(
        self,
        taint_threshold: int = 0,
        *,
        force_taint: bool = False,
        force_computation: bool = False,
        caller: Input | None = None,
    ) -> None:
        self._node.taint_below_threshold(
            taint_threshold,
            force_taint=force_taint,
            force_computation=force_computation,
            caller=self if caller is None else caller,
        )


class Inputs(EdgeContainer):
    __slots__ = ()

    def __init__(self, iterable=None):
        super().__init__(iterable)
        self._dtype = Input

    def __str__(self):
        return f"→[{tuple(obj.name for obj in self)}]○"

    _repr_pretty_ = repr_pretty

    def touch(self) -> None:
        for input in self:
            input.touch()
