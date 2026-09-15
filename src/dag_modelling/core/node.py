from __future__ import annotations

from typing import TYPE_CHECKING

from .exception import UnclosedGraphError
from .node_base import NodeBase
from .node_methods import _NodeMethods

if TYPE_CHECKING:
    from .input import Input


class Node(NodeBase, _NodeMethods):
    __slots__ = (
        "_name",
        "_labels",
        "_graph",
        "_logger",
        "_exception",
        "_internal_buffers",
        "_meta_node",
        "_debug",
        "_evaluator",
        "_structure",
    )

    def __init__(self, *args, **kwargs):
        # NodeBase has __init__ in its dict (inherited from _NodeBaseMethods);
        # MRO would dispatch there before _NodeMethods. Bypass explicitly.
        _NodeMethods.__init__(self, *args, **kwargs)

    def touch(self, force_computation=False):
        if not force_computation:
            if not self._evaluator.tainted:
                return
            if not self._evaluator.closed:
                raise UnclosedGraphError("Cannot evaluate not closed node!", node=self)
        self._evaluator._touch_implementation()

    def touch_above_threshold(self, taint_threshold: int = 0, force_computation=False):
        if not force_computation:
            if not self._evaluator.tainted_above_threshold(taint_threshold):
                return
            if not self.closed:
                raise UnclosedGraphError("Cannot evaluate not closed node!", node=self)
        self._evaluator._touch_above_threshold_implementation(taint_threshold)

    def unfreeze(self):
        if self._evaluator.unfreeze():
            self.taint()

    def taint(
        self,
        *,
        force_taint: bool = False,
        force_computation: bool = False,
        caller: "Input | None" = None,
    ):
        if self._evaluator.tainted and not force_taint:
            return
        if self._evaluator.frozen:
            self._evaluator.frozen_tainted = True
            return

        self._evaluator.tainted = True
        ret = self._evaluator._touch_implementation() if (self._evaluator._immediate or force_computation) else None
        self._evaluator.taint_children(
            force_taint=force_taint, force_computation=force_computation, caller=caller
        )
        return ret

    def taint_below_threshold(
        self,
        taint_threshold: int = 0,
        *,
        force_taint: bool = False,
        force_computation: bool = False,
        caller: "Input | None" = None,
    ):
        if self._evaluator.tainted_value > taint_threshold and not force_taint:
            return
        if self.frozen:
            self._evaluator.frozen_tainted = True
            return

        self._evaluator.tainted_value = taint_threshold + 1
        ret = self._evaluator._touch_above_threshold_implementation(taint_threshold) if (self._evaluator._immediate or force_computation) else None
        self._evaluator.taint_children_below_threshold(
            taint_threshold, force_taint=force_taint, force_computation=force_computation, caller=caller
        )

        return ret

    def taint_children(
        self,
        *,
        force_taint: bool = False,
        force_computation: bool = False,
        caller: "Input | None" = None,
    ):
        self._evaluator.taint_children(
            force_taint=force_taint, force_computation=force_computation, caller=caller
        )
