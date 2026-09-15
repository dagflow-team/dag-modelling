from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .input import Inputs
    from .output import Outputs

_SLOT_NAMES = (
    "tainted",
    "tainted_value",
    "frozen",
    "frozen_tainted",
    "invalid",
    "closed",
    "allocated",
    "needs_reallocation",
    "needs_post_allocate",
    "types_tainted",
    "_children",
    "_parents",
    "function",
    "_functions_dict",
    "_n_calls",
    "_input_nodes_callbacks",
    "_immediate",
)


class _NodeEvalMethods:
    __slots__ = ()

    def __init__(
        self,
        *,
        children: Outputs,
        parents: Inputs,
        tainted: bool = True,
        frozen: bool = False,
        immediate: bool = False,
    ) -> None:
        self._children = children
        self._parents = parents
        self.tainted = tainted
        self.tainted_value = 1
        self.frozen = frozen
        self.frozen_tainted = False
        self.invalid = False
        self.closed = False
        self.allocated = False
        self.types_tainted = True
        self.needs_reallocation = False
        self.needs_post_allocate = False
        # execution
        self.function = None
        self._functions_dict = {}
        self._n_calls = 0
        self._input_nodes_callbacks = []
        self._immediate = immediate

    def __str__(self):
        return ", ".join(f"{slot}={getattr(self, slot)}" for slot in _SLOT_NAMES)

    @property
    def children(self):
        return self._children

    @property
    def parents(self):
        return self._parents

    def taint_type(self, force_taint: bool = False) -> None:
        if self.types_tainted and not force_taint:
            return
        self.types_tainted = True
        self.tainted = True
        self.frozen = False
        for child in self._children:
            child.taint_children_type(force_taint=force_taint)

    def reset_n_calls(self):
        self._n_calls = 0
