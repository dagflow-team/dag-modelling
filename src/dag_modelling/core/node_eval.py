from __future__ import annotations

from typing import TYPE_CHECKING

from .node_eval_methods import _NodeEvalMethods

if TYPE_CHECKING:
    from .input import Input


class NodeEval(_NodeEvalMethods):
    __slots__ = (
        # Flags
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
        # Graph references
        "_children",
        "_parents",
        # Execution
        "function",
        "_functions_dict",
        "_n_calls",
        "_input_nodes_callbacks",
        "_immediate",
    )

    def _invalidate(self, invalid: bool) -> None:
        if invalid:
            self.invalidate()
        elif any(parent.invalid for parent in self._parents.iter_all()):
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
        for child in self._children:
            child.invalid = invalid

    def invalidate_parents(self, invalid: bool = True) -> None:
        for parent in self._parents.iter_all():
            node = parent.parent_node
            node.invalidate(invalid)
            node.invalidate_parents(invalid)

    def freeze(self) -> None:
        self.frozen = True
        self.frozen_tainted = False

    def unfreeze(self) -> bool:
        if not self.frozen:
            return False
        self.frozen = False
        if self.frozen_tainted:
            self.frozen_tainted = False
            return True
        return False

    def tainted_above_threshold(
        self,
        taint_threshold: int = 0,
    ) -> bool:
        return self.tainted_value > taint_threshold

    def taint_children(
        self,
        *,
        force_taint: bool = False,
        force_computation: bool = False,
        caller: "Input | None" = None,
    ) -> None:
        for child in self._children:
            child.taint_children(
                force_taint=force_taint, force_computation=force_computation, caller=caller
            )

    def taint_children_below_threshold(
        self,
        taint_threshold: int = 0,
        *,
        force_taint: bool = False,
        force_computation: bool = False,
        caller: Input | None = None,
    ) -> None:
        for child in self.children:
            child.taint_children_below_threshold(
                taint_threshold,
                force_taint=force_taint,
                force_computation=force_computation,
                caller=caller,
            )

    def _eval_implementation(self) -> None:
        self._n_calls += 1
        self.function()

    def _touch_implementation(self) -> None:
        self._n_calls += 1
        self.function()
        self.tainted = False

    def _touch_above_threshold_implementation(self, taint_threshold: int):
        # for debug purposes
        # replaced function is expected to process taint_threshold
        self._n_calls += 1
        self._function_stub_with_threshold(taint_threshold)
        self.tainted_value = taint_threshold

    def _function_stub_with_threshold(self, taint_threshold: int):
        """An empty function stub of the Node that touches parent nodes to
        start a recursive execution of a graph (without computations)"""
        for input in self.parents.iter_all():
            input.touch_above_threshold(taint_threshold)
