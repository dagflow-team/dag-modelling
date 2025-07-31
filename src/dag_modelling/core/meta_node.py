from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from .exception import CriticalError, InitializationError
from .input_strategy import InheritInputStrategy, InputStrategyBase
from .node_base import NodeBase

if TYPE_CHECKING:
    from collections.abc import Callable

    from .input import Input

TStrOrPair = str | tuple[str, str]
TPairsOrDict = Sequence[TStrOrPair] | dict

MetaNodeStrategies = {"LeadingNode", "NewNode", "Disable"}
MetaNodeStrategiesType = Literal[MetaNodeStrategies]


class MetaNode(NodeBase):
    """A node containing multiple nodes and exposing part of their inputs and
    outputs."""

    __slots__ = (
        "_nodes",
        "_strategy",
        "_leading_node",
        "_new_node_cls",
        "_call_functions",
        "_node_inputs_pos",
        "_node_outputs_pos",
        "_input_strategy",
        "_call_positional_input",
        "__weakref__",  # needed for weakref
    )

    _nodes: list[NodeBase]
    _strategy: MetaNodeStrategiesType
    _leading_node: NodeBase | None
    _new_node_cls: type[NodeBase]
    _call_functions: dict[str, Callable]
    _node_inputs_pos: NodeBase | None
    _node_outputs_pos: NodeBase | None
    _input_strategy: InputStrategyBase
    _call_positional_input: Callable

    def __init__(
        self,
        strategy: MetaNodeStrategiesType = "LeadingNode",
        new_node_cls: type[NodeBase] = NodeBase,
    ):
        super().__init__()
        if strategy not in MetaNodeStrategies:
            raise InitializationError(
                f"strategy must be in {MetaNodeStrategies}, but given {strategy}",
                node=self,
            )
        self._strategy = strategy
        self._nodes = []
        self._leading_node = None
        self._node_inputs_pos = None
        self._node_outputs_pos = None
        self._input_strategy = InputStrategyBase(node=self)
        self._call_functions = {
            "LeadingNode": self._call_leading_node,
            "NewNode": self._call_new_node,
            "Disable": self._call_disabled,
        }
        self._call_positional_input = self._call_functions[strategy]
        self._new_node_cls = new_node_cls

    @property
    def nodes(self) -> dict[str, NodeBase]:
        return {node.name: node for node in self._nodes}

    @property
    def leading_node(self) -> NodeBase | None:
        return self._leading_node

    @property
    def new_node_cls(self) -> type[NodeBase]:
        return self._new_node_cls

    def _add_input_to_node(
        self, node: NodeBase, name: str | None = None, *args, **kwargs
    ) -> Input | None:
        inp = node(name, *args, **kwargs)
        if inp and inp.name not in self.inputs:
            self.inputs.add(inp)  # adding to meta_node.inputs
            if inp.child_output is not None:  # adding to meta_node.outputs
                self.outputs.add(inp.child_output)
        return inp

    def _call_new_node(
        self,
        node_args: dict | None = None,
        input_args: dict | None = None,
        meta_node_args: dict | None = None,
        new_node_cls: type[NodeBase] | None = None,
    ) -> Input | None:
        """Creates new node with positional input."""
        if node_args is None:
            node_args = {}
        if input_args is None:
            input_args = {}
        if meta_node_args is None:
            meta_node_args = {}
        if new_node_cls is None:  # use default cls
            new_node_cls = self.new_node_cls

        node = new_node_cls(**node_args)
        self._add_node(node, **meta_node_args)

        # set input and output indices as nodes count to avoid same naming
        ln = len(self._nodes)
        return self._add_input_to_node(node, idx_input=ln, idx_output=ln, **input_args)

    def _call_leading_node(self, *args, **kwargs) -> Input | None:
        """Creates new node with positional input."""
        if self.leading_node is None:
            raise CriticalError(
                "Cannot create a new input: the leading node is unknown!", node=self
            )
        return self._add_input_to_node(self.leading_node, *args, **kwargs)

    def _call_disabled(self, *args, **kwargs) -> Input | None:
        """Prevents creation of new nodes."""
        raise CriticalError(
            "Cannot create a new input: the node is not scalable!", node=self
        )

    def __call__(
        self,
        name: str | Sequence[str] | None = None,
        nodename: str | None = None,
        *args,
        **kwargs,
    ) -> Input | None | tuple[Input | None, ...]:
        """
        For positional inputs there are two strategies:
            * append a new positional input into the leading node,
            * append a new node with a positional input to self.
        Else returns a tuple of calls of all the nodes.
        """
        if name is None:
            return self._call_positional_input(*args, **kwargs)
        if nodename is not None:
            node = self.nodes.get(nodename)
            if node is None:
                raise CriticalError(
                    f"Cannot create an input due to the meta_node has no node with name={nodename}",
                    node=node,
                )
            if isinstance(name, str):
                return self._add_input_to_node(node, name=name, *args, **kwargs)
            return tuple(
                self._add_input_to_node(node, name=_name, *args, **kwargs)
                for _name in name
            )
        names = [name] * len(self._nodes) if isinstance(name, str) else name
        return tuple(
            self._add_input_to_node(node, name=_name, *args, **kwargs)
            for _name, node in zip(names, self._nodes)
        )

    def _add_node(
        self,
        node: NodeBase,
        *,
        inputs_pos: bool = False,
        outputs_pos: bool = False,
        outputs_pos_fmt: str | None = None,
        kw_inputs: TPairsOrDict = [],
        kw_inputs_optional: TPairsOrDict = [],
        kw_outputs: TPairsOrDict = [],
        kw_outputs_optional: TPairsOrDict = [],
        merge_inputs: Sequence[str] = [],
        missing_inputs: bool = False,
        also_missing_outputs: bool = False,
    ) -> None:
        if node in self._nodes:
            raise RuntimeError("NodeBase already added")

        self._nodes.append(node)
        node.meta_node = self
        if self._strategy == "LeadingNode" and self.leading_node is None:
            self._leading_node = node

        if inputs_pos:
            self._import_pos_inputs(node)
        if outputs_pos:
            self._import_pos_outputs(node, namefmt=outputs_pos_fmt)
        self._import_kw_inputs(node, kw_inputs, merge=merge_inputs)
        if kw_inputs_optional:
            self._import_kw_inputs(
                node, kw_inputs_optional, merge=merge_inputs, optional=True
            )
        self._import_kw_outputs(node, kw_outputs)
        if kw_outputs_optional:
            self._import_kw_outputs(node, kw_outputs_optional, optional=True)

        if missing_inputs:
            self._input_strategy = InheritInputStrategy(
                node, self, inherit_outputs=also_missing_outputs
            )
        if not missing_inputs and also_missing_outputs:
            raise RuntimeError("also_missiong_outputs=True option makes no sense")

    def _import_pos_inputs(self, node: NodeBase, *, keyword: bool = True) -> None:
        if self._strategy == "LeadingNode" and self.leading_node is not None:
            keyword = False
        elif self._node_inputs_pos is not None:
            raise RuntimeError("Positional inputs already inherited")
        else:
            self._node_inputs_pos = node
        for input in node.inputs:
            self.inputs.add(input, positional=True, keyword=keyword)

    def _import_pos_outputs(
        self, node: NodeBase, *, namefmt: str | None = None, keyword: bool = True
    ) -> None:
        if self._strategy == "LeadingNode" and self.leading_node is not None:
            keyword = False
        elif self._node_outputs_pos is not None:
            raise RuntimeError("Positional outputs already inherited")
        else:
            self._node_outputs_pos = node
        for output in node.outputs:
            self.outputs.add(
                output,
                positional=True,
                name=namefmt and namefmt.format(output.name),
                keyword=keyword,
            )

    def _import_kw_inputs(
        self,
        node: NodeBase,
        kw_inputs: TPairsOrDict = [],
        merge: Sequence[str] = [],
        optional: bool = False,
    ) -> None:
        iterable = kw_inputs.items() if isinstance(kw_inputs, dict) else kw_inputs
        for iname in iterable:
            tname = iname
            if not isinstance(iname, str):
                iname, tname = iname
            try:
                newinput = node.inputs.get_kw(iname)
            except KeyError as e:
                if optional:
                    continue
                raise RuntimeError(f"Input {iname} not found") from e
            mergethis = tname in merge
            self.inputs.add(newinput, name=tname, merge=mergethis, positional=False)

    def _import_kw_outputs(
        self, node: NodeBase, kw_outputs: TPairsOrDict = [], *, optional: bool = True
    ) -> None:
        iterable = kw_outputs.items() if isinstance(kw_outputs, dict) else kw_outputs
        for oname in iterable:
            tname = None
            if not isinstance(oname, str):
                oname, tname = oname
            try:
                output = node.outputs.get_kw(oname)
            except KeyError as e:
                if optional:
                    continue
                raise RuntimeError(f"Output {oname} not found") from e
            self.outputs.add(output, name=tname, positional=False)

    def print(self, recursive: bool = False):
        print(f"MetaNode: →[{self.inputs.len_all()}],[{self.outputs.len_all()}]→")

        def getstr(prefix_disconnected, prefix_connected, name, obj):
            if isinstance(obj, tuple):
                Nconnected = sum(bool(node.connected()) for node in obj)
                Nnode = len(obj)
                Ndisconnected = Nnode - Nconnected
                if Nconnected == 0:
                    return f"{name}: {prefix_disconnected} #{Ndisconnected}"
                elif Ndisconnected > 0:
                    return (
                        f"{name}: {prefix_disconnected} #{Ndisconnected} +"
                        f" {prefix_connected} #{Nconnected}"
                    )
                else:
                    return f"{name}: {prefix_connected} #{Nnode}"
            return f"{name}: {obj!s}"

        for i, (name, input) in enumerate(self.inputs.pos_edges.items()):
            print(f"     {i} {getstr('→○', '→●', name, input)}")
        for name, input in self.inputs.nonpos_edges.items():
            print(f"     {getstr('→○', '→●', name, input)}")
        for i, (name, output) in enumerate(self.outputs.pos_edges.items()):
            print(f"     {i} {getstr('○→', '●→', name, output)}")
        for name, output in self.outputs.nonpos_edges.items():
            print(f"     {getstr('○→', '●→', name, output)}")

        if recursive:
            for i, node in enumerate(self._nodes):
                print(f"subnode {i}: ", end="")
                node.print()
