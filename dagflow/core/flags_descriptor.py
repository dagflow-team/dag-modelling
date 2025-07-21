from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .input import Input, Inputs
    from .output import Outputs


class FlagsDescriptor:
    """
    The class stores the node flags
    """

    __slots__ = (
        "tainted",
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
    )

    # flags
    tainted: bool
    frozen: bool
    frozen_tainted: bool
    invalid: bool
    closed: bool
    allocated: bool
    needs_reallocation: bool
    needs_post_allocate: bool
    types_tainted: bool
    # observers and observed
    # _node: Node
    _children: Outputs
    _parents: Inputs

    def __init__(
        self,
        *,
        children: Outputs,
        parents: Inputs,
        tainted: bool = True,
        frozen: bool = False,
    ) -> None:
        self._children = children
        self._parents = parents
        self.tainted = tainted
        self.frozen = frozen
        self.frozen_tainted = False
        self.invalid = False
        self.closed = False
        self.allocated = False
        self.types_tainted = True
        self.needs_reallocation = False
        self.needs_post_allocate = False

    def __str__(self) -> str:
        return ", ".join(f"{slot}={getattr(self, slot)}" for slot in self.__slots__)

    @property
    def children(self) -> Outputs:
        return self._children

    @property
    def parents(self) -> Inputs:
        return self._parents

    def _invalidate(self, invalid: bool) -> None:
        if invalid:
            self.invalidate()
        elif any(parent.invalid for parent in self.parents.iter_all()):
            return
        else:
            self.invalidate(False)
        self.invalidate_children(invalid)

    def invalidate(self, invalid: bool = True) -> None:
        self.invalid = invalid
        self.frozen_tainted = False
        self.frozen = False
        self.tainted = True

    def invalidate_children(self, invalid: bool = True) -> None:
        for child in self.children:
            child.invalid = invalid

    def invalidate_parents(self, invalid: bool = True) -> None:
        for parent in self.parents.iter_all():
            node = parent.parent_node
            node.invalidate(invalid)
            node.invalidate_parents(invalid)

    def freeze(self) -> None:
        self.frozen = True
        self.frozen_tainted = False

    def taint_children(
        self,
        *,
        force_taint: bool = False,
        force_computation: bool = False,
        caller: Input | None = None,
    ) -> None:
        for child in self.children:
            child.taint_children(
                force_taint=force_taint, force_computation=force_computation, caller=caller
            )

    def taint_type(self, force_taint: bool = False):
        if self.types_tainted and not force_taint:
            return
        self.types_tainted = True
        self.tainted = True
        self.frozen = False
        for child in self.children:
            child.taint_children_type(force_taint=force_taint)
