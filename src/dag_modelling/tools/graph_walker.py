from __future__ import annotations

from abc import abstractmethod
from dataclasses import InitVar, dataclass, field
from typing import Callable, Generator, Generic, Sequence, TypeVar, override

from ..core.input import Input
from ..core.node import Node
from ..core.output import Output
from ..tools.logger import DEBUG, INFO2, logger

NodeT = TypeVar("NodeT")
OutputT = TypeVar("OutputT")
InputT = TypeVar("InputT")


@dataclass(slots=True, kw_only=True, eq=False)
class GraphWalker:
    """Meta class for walking graph.

    Parameters
    ----------
    min_depth : int | None, default=None
        Depth of graph walking in backward direction, parameters is less or equal than 0.
    max_depth : int | None, default=None
        Depth of graph walking in forward direction, parameters is larger or equal than 0.
    enable_process_forward : bool, default=True
        Enable to process nodes in forward direction.
    enable_process_backward : bool, default=True
        Enable to process nodes in backward direction.
    enable_process_meshes_edges : bool, default=False
        Enable to process meshes and edges in process of walking.
    enable_process_full_graph : bool, default=False
        Enable to process full graph.
    node_skip_fcn : Callable[[Node], bool], default=lambda _: False
        Skip function.
    node_handler : NodeHandlerBase | None, defalut=NodeHandlerDGM
        Handler for nodes.
    """

    min_depth: int | None = None
    max_depth: int | None = None

    _nodes: dict[Node, int] = field(init=False, default_factory=dict)
    _queue_nodes: dict[Node, int] = field(init=False, default_factory=dict)
    _queue_meshes_edges: dict[Node, int] = field(init=False, default_factory=dict)
    edges: dict[Input, tuple[Node, Node]] = field(init=False, default_factory=dict)
    open_outputs: list[Output] = field(init=False, default_factory=list)
    open_inputs: list[Input] = field(init=False, default_factory=list)

    enable_process_forward: bool = True
    enable_process_backward: bool = True
    enable_process_meshes_edges: bool = False
    enable_process_full_graph: bool = False

    node_skip_fcn: Callable[[Node], bool] = lambda _: False

    _node_handler: NodeHandlerBase = field(init=False)
    node_handler: InitVar[NodeHandlerBase | None] = None

    def __post_init__(self, node_handler):
        self._node_handler = node_handler or NodeHandlerDGM()

        growth_disabled = not self.enable_process_backward and not self.enable_process_forward
        if growth_disabled and self.enable_process_full_graph:
            raise RuntimeError(
                "GraphWalker got conflicting arguments: "
                f"{self.enable_process_full_graph=}, {self.enable_process_backward=} and {self.enable_process_forward=}"
            )
        if not self.enable_process_backward and self.enable_process_meshes_edges:
            raise RuntimeError(
                "GraphWalker got conflicting arguments: "
                f"{self.enable_process_backward=} and {self.enable_process_meshes_edges=}"
            )

    @property
    def nodes(self) -> dict[Node, int]:
        return self._nodes

    def process_from_nodes(self, nodes: Sequence[Node], **kwargs):
        for node in nodes:
            self.process_from_node(node, **kwargs)

    def process_from_node(
        self,
        node: Node,
        *,
        depth: int = 0,
        process_full_graph: bool | None = None,
        process_initial_node: bool = True,
    ):
        """Main function

        1. Add the node to the queue.
        2. Go strictly backward from the node. Add nodes to the queue.
        3. Go strictly forward from the node. Add nodes to the queue.
        4. For each node in the queue:
            a. Optionally: process nodes forward/backward.
            b. Push queue to the list of nodes. The queue is copied to the queue of meshes/edges.
            c. Optionally: process meshes/edges.
            d. Repeat as long as new nodes are added to the queue.
        """
        if process_full_graph is None:
            process_full_graph = self.enable_process_full_graph

        if process_initial_node and not self._add_node_to_queue(node, depth=depth):
            return

        if self.enable_process_backward:
            self._build_queue_nodes_backward_from(node, depth=depth)

        if self.enable_process_forward:
            self._build_queue_nodes_forward_from(node, depth=depth)

        iteration = 0
        while self.has_queue:
            iteration += 1

            if self.enable_process_full_graph:
                self._process_nodes_from_queue()
            else:
                self._push_queue_to_storage()

            if self.enable_process_meshes_edges:
                self._process_queue_meshes_edges()
            else:
                self._queue_meshes_edges = {}

        logger.log(
            INFO2,
            f"Subgraph iteration done: "
            f"nodes={len(self.nodes)} edges={len(self.edges)} "
            f"inputs={len(self.open_inputs)} outputs={len(self.open_outputs)}",
        )

    @property
    def has_queue(self) -> bool:
        return bool(self._queue_nodes or self._queue_meshes_edges)

    def _push_queue_to_storage(self):
        self.nodes.update(self._queue_nodes)
        self._queue_meshes_edges.update(self._queue_nodes)
        self._queue_nodes = {}

    def _process_nodes_from_queue(self):
        while self._queue_nodes:
            self._process_nodes_from_queue_once()

    def _process_nodes_from_queue_once(self):
        """Goes forward/backward from each node from the queue."""
        if not self._queue_nodes:
            return
        queue = self._queue_nodes.copy()
        self._push_queue_to_storage()

        for node, depth in queue.items():
            self.process_from_node(
                node,
                depth=depth,
                process_full_graph=False,
                process_initial_node=False,
            )
        return queue

    def depth_outside_limits(self, depth: int) -> bool:
        if self.min_depth is not None and depth < self.min_depth:
            logger.log(
                DEBUG, f"  depth {depth} outside limits [{self.min_depth}, {self.max_depth}]"
            )
            return True

        if self.max_depth is not None and depth > self.max_depth:
            logger.log(
                DEBUG, f"  depth {depth} outside limits [{self.min_depth}, {self.max_depth}]"
            )
            return True

        return False

    def _node_already_added(self, node: Node) -> bool:
        if node in self._queue_nodes or node in self.nodes:
            logger.log(DEBUG, "  node already added")
            return True

        return False

    def _may_not_add_node(self, node: Node, depth: int) -> bool:
        logger.log(INFO2, f"? d: {depth: 3d} {self._node_handler.get_string(node)}")
        return (
            self.depth_outside_limits(depth)
            or self._node_already_added(node)
            or self.node_skip_fcn(node)
        )

    def _add_node_to_queue(self, node: Node, *, depth: int) -> bool:
        if self.depth_outside_limits(depth) or self._node_already_added(node):
            return False

        self._queue_nodes[node] = depth
        logger.log(INFO2, f"+ d: {depth: 3d} {self._node_handler.get_string(node)}")

        return True

    def _build_queue_nodes_backward_from(self, node: Node, *, depth: int):
        """Go strictly backward from the node.
        Add nodes to queue.
        Stop if depth is too low.
        """
        depth -= 1
        if self.depth_outside_limits(depth):
            return
        for input in self._node_handler.iter_inputs(node):
            try:
                parent_node = input.parent_node
            except AttributeError:
                self.open_inputs.append(input)
                continue

            if not input in self.edges:
                self.edges[input] = (parent_node, node)
            logger.log(INFO2, f"b d: {depth: 3d} {self._node_handler.get_string(parent_node)}")
            if not self._add_node_to_queue(parent_node, depth=depth):
                logger.log(DEBUG, "  skip")
                continue
            self._build_queue_nodes_backward_from(parent_node, depth=depth)

    def _build_queue_nodes_forward_from(self, node: Node, *, depth: int):
        """Go strictly forward from the node.
        Add nodes to queue.
        Stop if depth is too high.
        """
        depth += 1
        if self.depth_outside_limits(depth):
            return
        for output in self._node_handler.iter_outputs(node):
            if output.child_inputs:
                for child_input in output.child_inputs:
                    child_node = child_input.node
                    logger.log(
                        INFO2, f"f d: {depth: 3d} {self._node_handler.get_string(child_node)}"
                    )
                    if not self._add_node_to_queue(child_node, depth=depth):
                        logger.log(DEBUG, "  skip")
                        continue
                    self._build_queue_nodes_forward_from(child_node, depth=depth)
            else:
                self.open_outputs.append(output)

    def _process_queue_meshes_edges(self):
        """Add nodes of meshes and edges to the queue, if they are not already present."""
        for node, depth in self._queue_meshes_edges.items():
            depth -= 1
            if self.depth_outside_limits(depth):
                continue
            for output in self._node_handler.iter_meshes_edges(node):
                self._add_node_to_queue(output.node, depth=depth)
                logger.log(INFO2, f"e d: {depth: 3d} {self._node_handler.get_string(node)}")

        self._queue_meshes_edges = {}


class NodeHandlerBase(Generic[NodeT, OutputT, InputT]):
    __slots__ = ()

    @abstractmethod
    def iter_inputs(self, node: NodeT) -> Generator[InputT, None, None]:
        pass

    @abstractmethod
    def iter_outputs(self, node: NodeT) -> Generator[OutputT, None, None]:
        pass

    @abstractmethod
    def iter_meshes_edges(self, node: NodeT) -> Generator[OutputT, None, None]:
        pass

    @abstractmethod
    def get_string(self, node: NodeT) -> str:
        pass


class NodeHandlerDGM(NodeHandlerBase[Node, Output, Input]):

    @override
    def iter_inputs(self, node: Node) -> Generator[Input, None, None]:
        yield from node.inputs.iter_all()

    @override
    def iter_outputs(self, node: Node) -> Generator[Output, None, None]:
        yield from node.outputs.iter_all()

    @override
    def iter_meshes_edges(self, node: Node) -> Generator[Output, None, None]:
        for output in node.outputs.iter_all():
            yield from output.dd.axes_edges
            yield from output.dd.axes_edges

    @override
    def get_string(self, node: Node) -> str:
        if (path := node.labels.path) is not None:
            return f"path:  {path}"
        elif (text := node.labels.text) is not None:
            return f"text: {text}"
        elif (name := node.labels.name) is not None:
            return f"label: {name}"
        return f"node:  {node!r}"


def get_subgraph_nodes(
    sources: Sequence[Node], sinks: Sequence[Node], *, enable_process_meshes_edges: bool = False
) -> list[Node]:
    """
    Collect a nodes of a subgraph between sets of source nodes and sink nodes.
    ---
    Nodes are obtained by finding intersection between all the nodes backward from sinks and nodes
    forward from sources.

    Parameters
    ----------
    sources : Sequence[Node]
        Sequence of source nodes.
    sinks : Sequence[Node]
        Sequence of sink nodes.
    enable_process_meshes_edges : bool
        If enabled, collect also mesh/edge nodes.
    Returns
    -------
    list[Node]
        List of nodes.
    """
    walker_fwd = GraphWalker(enable_process_backward=False)
    walker_bwd = GraphWalker(
        enable_process_forward=False, enable_process_meshes_edges=enable_process_meshes_edges
    )

    walker_fwd.process_from_nodes(sources)
    walker_bwd.process_from_nodes(sinks)

    intersection = set(walker_fwd.nodes) & set(walker_bwd.nodes)

    return [node for node in walker_fwd.nodes if node in intersection]
