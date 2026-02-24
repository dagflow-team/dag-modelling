from __future__ import annotations

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


def reveal_source_sink(nodes: Sequence[Node]) -> tuple[list[Node], list[Node]]:
    """Find sources and sinks for given list of nodes."""
    sources = []
    sinks = []
    for node in nodes:
        has_parents = any(n in nodes for n in __parent_nodes_gen(node))
        has_children = any(n in nodes for n in __child_nodes_gen(node))
        if has_parents and has_children:
            continue
        elif has_parents:
            sinks.append(node)
        else:
            sources.append(node)

    return sources, sinks
