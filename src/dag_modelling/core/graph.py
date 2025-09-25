from __future__ import annotations

from ..tools.logger import Logger, get_logger
from .exception import ClosedGraphError, ClosingError, InitializationError, UnclosedGraphError
from .graph_base import GraphBase


class Graph(GraphBase):
    """
    The graph class:
    holds nodes as a list, has name, label, logger and uses context
    """

    __slots__ = (
        "_label",
        "_name",
        "_close_on_exit",
        "_strict",
        "_closed",
        "_nodes_closed",
        "_debug",
        "_logger",
    )

    _label: str | None
    _name: str
    _close_on_exit: bool
    _closed: bool
    _nodes_closed: bool
    _debug: bool
    _logger: Logger

    def __init__(self, *args, close_on_exit: bool = False, strict: bool = True, **kwargs):
        super().__init__(*args)
        self._label = kwargs.pop("label", None)
        self._name = kwargs.pop("name", "graph")
        self._debug = kwargs.pop("debug", False)
        self._close_on_exit = close_on_exit
        self._strict = strict
        self._closed = False
        self._nodes_closed = False
        # init or get default logger
        self._logger = get_logger(
            filename=kwargs.pop("logfile", None),
            debug=self.debug,
            console=kwargs.pop("console", True),
            formatstr=kwargs.pop("logformat", None),
            name=kwargs.pop("loggername", None),
        )
        if kwargs:
            raise InitializationError(f"Unparsed arguments: {kwargs}!")

    @property
    def debug(self) -> bool:
        return self._debug

    @property
    def logger(self) -> Logger:
        return self._logger

    @property
    def name(self) -> str:
        return self._name

    @property
    def closed(self) -> bool:
        return self._closed

    def label(self):
        """Returns formatted label."""
        if self._label:
            return self._label.format(self._label, nodes=len(self._nodes))

    def add_node(self, name, **kwargs):
        """Adds a node, if the graph is opened.

        It is possible to pass the node class via the `nodeclass` arg
        (default: `Node`)
        """
        if self.closed:
            raise ClosedGraphError(node=name)

        from .node import Node

        return kwargs.pop("nodeclass", Node)(name, graph=self, **kwargs)

    def add_nodes(self, nodes, **kwargs):
        """Adds nodes."""
        if self.closed:
            raise ClosedGraphError(node=nodes)

        return tuple(self.add_node(node, **kwargs) for node in nodes)

    def print(self):
        print(f"Graph with {len(self._nodes)} nodes")
        for node in self._nodes:
            node.print()

    def close(self, *, strict: bool = True, force: bool = False, **kwargs) -> bool:
        """Closes the graph."""
        if force:
            self._nodes_closed = False
        elif self._closed:
            return True
        self.logger.debug(f"Graph '{self.name}': Closing...")

        nodes_to_process = self._new_nodes if self._nodes_closed else self._nodes

        self.logger.debug(f"Graph '{self.name}': Update types...")
        for node in nodes_to_process:
            if not node.closed:
                try:
                    node.update_types()
                except ClosingError:
                    if strict:
                        raise
        self.logger.debug(f"Graph '{self.name}': Allocate memory...")
        for node in nodes_to_process:
            if not node.closed:
                try:
                    node.allocate(**kwargs)
                except ClosingError:
                    if strict:
                        raise
        self.logger.debug(f"Graph '{self.name}': Closing nodes...")
        for node in nodes_to_process:
            try:
                self._closed = node.close(close_children=True, **kwargs)
            except ClosingError:
                if strict:
                    raise
            if not self._closed:
                break
        else:
            self._closed = True

        self._clear_new_nodes_list()
        self._nodes_closed = True

        if strict and not self._closed:
            raise UnclosedGraphError("The graph is still open!")
        self.logger.debug(
            f"Graph '{self.name}': The graph {self._closed and 'is closed' or 'failed to close'}!"
        )
        return self._closed

    def open(
        self, force: bool = False, *, close_on_exit: bool = True, open_nodes: bool = False
    ) -> Graph:
        """Opens the graph recursively."""
        self._close_on_exit = close_on_exit

        if not self._closed and not force:
            return self

        self.logger.debug(f"Graph '{self.name}': Opening...")

        if open_nodes:
            self._closed = not all(
                node.open(force_taint=force, open_children=True) for node in self._nodes
            )
            self._nodes_closed = False
        else:
            self._closed = False

        if self._closed:
            raise ClosedGraphError("The graph is still closed!")

        return self

    def build_index_dict(self, index):
        for node in self:
            node.labels.build_index_dict(index)

            for output in node.outputs.iter_all():
                output.labels.build_index_dict(index)

    @classmethod
    def current(cls) -> Graph | None:
        return _context_graph[-1] if _context_graph else None

    def __enter__(self):
        _context_graph.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if _context_graph.pop() != self:
            raise RuntimeError("Graph: invalid context exit")

        if exc_val is not None:
            raise exc_val

        if self._close_on_exit:
            self.close(strict=self._strict)


_context_graph: list[Graph] = []
