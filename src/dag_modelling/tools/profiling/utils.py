from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from ...core.node import Node


def __child_nodes_gen(node: Node) -> Generator[Node, None, None]:
    """Access to the child nodes of the given node via the generator."""
    for output in node.outputs.iter_all():
        for child_input in output.child_inputs:
            yield child_input.node


def __parent_nodes_gen(node: Node) -> Generator[Node, None, None]:
    """Access to the parent nodes of the given node via the generator."""
    for input in node.inputs.iter_all():
        yield input.parent_node


def __check_reachable(nodes_gathered, sinks):
    for sink in sinks:
        if sink not in nodes_gathered:
            raise ValueError(
                f"One of the `sinks` nodes is unreachable: {sink} " "(no paths from sources)"
            )


def gather_related_nodes(sources: Sequence[Node], sinks: Sequence[Node]) -> set[Node]:
    """Find all nodes that lie on all possible paths between `sources` and
    `sinks`

    Modified Depth-first search (DFS) algorithm for multiple sources and
    sinks
    """
    related_nodes = set(sources)
    # Deque works well as Stack
    stack = deque()
    visited = set()
    for start_node in sources:
        cur_node = start_node
        while True:
            last_in_path = True
            for ch in __child_nodes_gen(cur_node):
                if ch in sinks:
                    related_nodes.add(ch)
                # If `_sinks` contains child node it would be already in `related_nodes`
                if ch in related_nodes:
                    related_nodes.update(stack)
                    related_nodes.add(cur_node)
                elif ch not in visited:
                    stack.append(cur_node)
                    cur_node = ch
                    last_in_path = False
                    break
            # No unvisited childs found (`for` loop did not encounter a `break`)
            else:
                visited.add(cur_node)
            if len(stack) == 0:
                break
            if last_in_path:
                cur_node = stack.pop()
    __check_reachable(related_nodes, sinks)
    return related_nodes


def reveal_source_sink(nodes: Sequence[Node]) -> tuple[list[Node], list[Node]]:
    """Find sources and sinks for given list of nodes."""
    sources = []
    sinks = []
    for node in nodes:
        have_parents = any(n in nodes for n in __parent_nodes_gen(node))
        have_childs = any(n in nodes for n in __child_nodes_gen(node))
        if have_parents and have_childs:
            continue
        elif have_parents:
            sinks.append(node)
        else:
            sources.append(node)
    return sources, sinks
