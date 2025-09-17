from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .node import Node


class GraphBase:
    __slots__ = ("_nodes", "_nodes_set", "_new_nodes")
    _nodes: list[Node]
    _nodes_set: set[Node]
    _new_nodes: list[Node]

    def __init__(self, *args):
        self._nodes = list(args)
        self._nodes_set = set(args)
        self._new_nodes = list(args)

        if len(self._nodes_set) != len(self._nodes):
            raise RuntimeError("There are duplicated nodes")

    def _clear_new_nodes_list(self):
        self._new_nodes.clear()

    def register_node(self, node):
        if node in self._nodes_set:
            return
        self._nodes.append(node)
        self._new_nodes.append(node)
        self._nodes_set.add(node)

    def touch(self):
        """Touch all the nodes"""
        for node in self._nodes:
            node.touch()

    def print(self):
        print(f"Graph consists of {len(self._nodes)} nodes:")
        for node in self._nodes:
            node.print()

    def __iter__(self):
        """Iterate nodes"""
        return iter(self._nodes)
