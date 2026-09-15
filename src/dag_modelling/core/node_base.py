from __future__ import annotations

from .node_base_methods import _NodeBaseMethods


class NodeBase(_NodeBaseMethods):
    __slots__ = ("inputs", "outputs", "_input_strategy")
