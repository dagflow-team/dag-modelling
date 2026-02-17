from __future__ import annotations

from abc import abstractmethod
from typing import Callable, Generator, Generic, TypeVar, override

from ..core.input import Input
from ..core.node import Node
from ..core.output import Output

NodeT = TypeVar("NodeT")
OutputT = TypeVar("OutputT")
InputT = TypeVar("InputT")


class GraphWalker:
    __slots__ = (
        "nodes",
        "_queue_nodes",
        "_queue_meshes_edges",
        "edges",
        "open_outputs",
        "open_inputs",
        "_current_depth",
        "min_depth",
        "max_depth",
        "_enable_process_backward",
        "_enable_process_forward",
        "_enable_process_meshes_edges",
        "_enable_process_full_graph",
        "_node_skip_fcn",
        "_node_handler",
    )

    nodes: dict[Node, int]
    _queue_nodes: dict[Node, int]
    _queue_meshes_edges: dict[Node, int]
    edges: dict[Input, tuple[Node, Node]]
    open_outputs: list[Output]
    open_inputs: list[Input]

    _current_depth: int

    min_depth: int | None
    max_depth: int | None

    _enable_process_backward: bool
    _enable_process_forward: bool
    _enable_process_backward: bool
    _enable_process_meshes_edges: bool
    _enable_process_full_graph: bool

    _node_skip_fcn: Callable[[Node], bool]

    _node_handler: NodeHandlerBase

    def __init__(
        self,
        *,
        min_depth: int | None,
        max_depth: int | None,
        process_backward: bool = True,
        process_forward: bool = True,
        process_meshes_edges: bool = False,
        process_full_graph: bool = False,
        node_skip_fcn: Callable[[Node], bool] = lambda _: False,
        node_handler: NodeHandlerBase | None = None,
    ):
        self.nodes = {}
        self._queue_nodes = {}
        self._queue_meshes_edges = {}
        self.edges = {}
        self.open_outputs = []
        self.open_inputs = []
        self._current_depth = 0
        self.min_depth = min_depth
        self.max_depth = max_depth
        self._enable_process_backward = process_backward
        self._enable_process_forward = process_forward
        self._enable_process_meshes_edges = process_meshes_edges
        self._enable_process_full_graph = process_full_graph

        self._node_skip_fcn = node_skip_fcn

        self._node_handler = node_handler or NodeHandlerDGM()

        growth_disabled = not process_backward and not process_forward
        if growth_disabled and process_full_graph:
            raise RuntimeError(
                "GraphWalker got conflicting arguments: "
                f"{process_full_graph=}, {process_backward=} and {process_forward=}"
            )
        if not process_backward and process_meshes_edges:
            raise RuntimeError(
                "GraphWalker got conflicting arguments: "
                f"{process_backward=} and {process_meshes_edges=}"
            )

    def process_from_node(
        self,
        node: Node,
        *,
        depth: int | None = 0,
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
            process_full_graph = self._enable_process_full_graph

        self.current_depth = depth
        depth = self.current_depth
        if process_initial_node and not self._add_node_to_queue(node):
            return

        if self._enable_process_backward:
            self._build_queue_nodes_backward_from(node, depth=depth)

        if self._enable_process_forward:
            self._build_queue_nodes_forward_from(node, depth=depth)

        iteration = 0
        while self.has_queue:
            iteration += 1

            if self._enable_process_full_graph:
                self._process_nodes_from_queue()
            else:
                self._push_queue_to_storage()

            if self._enable_process_meshes_edges:
                self._process_queue_meshes_edges()
            else:
                self._queue_meshes_edges = {}

        # print(f"Nodes: {len(self.nodes)}")
        # print(f"Edges: {len(self.edges)}")
        # print(f"Inputs: {len(self.open_inputs)}")
        # print(f"Outputs: {len(self.open_outputs)}")

    @property
    def has_queue(self) -> bool:
        return bool(self._queue_nodes or self._queue_meshes_edges)

    @property
    def current_depth(self) -> int:
        return self._current_depth

    @current_depth.setter
    def current_depth(self, value: int | None):
        if value is not None:
            self._current_depth = value

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

        for node, self.current_depth in queue.items():
            self.process_from_node(
                node,
                depth=self.current_depth,
                process_full_graph=False,
                process_initial_node=False,
            )
        return queue

    def _depth_outside_limits(self) -> bool:
        if self.min_depth is not None and self.current_depth < self.min_depth:
            return True

        if self.max_depth is not None and self.current_depth > self.max_depth:
            return True

        return False

    def _node_already_added(self, node: Node) -> bool:
        return node in self._queue_nodes or node in self.nodes

    def _may_not_add_node(self, node: Node) -> bool:
        # print(f"? depth: {self._current_depth: 3d} {self._node_handler.get_string(node)}")
        return (
            self._depth_outside_limits()
            or self._node_already_added(node)
            or self._node_skip_fcn(node)
        )

    def _add_node_to_queue(self, node: Node) -> bool:
        if self._depth_outside_limits() or self._node_already_added(node):
            return False

        self._queue_nodes[node] = self._current_depth

        return True

    def _build_queue_nodes_backward_from(self, node: Node, depth: int | None = None):
        """Go strictly backward from the node.
        Add nodes to queue.
        Stop if depth is too low.
        """
        self.current_depth = depth
        self.current_depth -= 1
        if self._depth_outside_limits():
            return
        for input in self._node_handler.iter_inputs(node):
            try:
                parent_node = input.parent_node
            except AttributeError:
                self.open_inputs.append(input)
                continue

            if not input in self.edges:
                self.edges[input] = (parent_node, node)
            if not self._add_node_to_queue(parent_node):
                continue
            # print(f"b depth: {self._current_depth: 3d} {self._node_handler.get_string(node)}")
            self._build_queue_nodes_backward_from(parent_node)

    def _build_queue_nodes_forward_from(self, node: Node, *, depth: int | None = None):
        """Go strictly forward from the node.
        Add nodes to queue.
        Stop if depth is too high.
        """
        self.current_depth = depth
        self.current_depth += 1
        if self._depth_outside_limits():
            return
        for output in self._node_handler.iter_outputs(node):
            if output.child_inputs:
                for child_input in output.child_inputs:
                    child_node = child_input.node
                    if not self._add_node_to_queue(child_node):
                        continue
                    # print(
                    #     f"f depth: {self._current_depth: 3d} {self._node_handler.get_string(node)}"
                    # )
                    self._build_queue_nodes_forward_from(child_node)
            else:
                self.open_outputs.append(output)

    def _process_queue_meshes_edges(self):
        """Add nodes of meshes and edges to the queue, if they are not already present."""
        if self._depth_outside_limits():
            return
        for node, self._current_depth in self._queue_meshes_edges.items():
            self.current_depth -= 1
            if self._depth_outside_limits():
                continue
            for output in self._node_handler.iter_meshes_edges(node):
                self._add_node_to_queue(output.node)
                # print(f"depth: {self._current_depth: 3d} {self._node_handler.get_string(node)}")

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
    __slots__ = ()

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
        elif (name := node.labels) is not None:
            return f"label: {name}"
        return f"node:  {node!r}"
