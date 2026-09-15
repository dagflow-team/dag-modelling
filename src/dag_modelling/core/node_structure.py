from __future__ import annotations

from typing import TYPE_CHECKING

from .exception import (
    ClosedGraphError,
    ClosingError,
    CriticalError,
    OpeningError,
    ReconnectionError,
)
from .input import Input
from .iter import IsIterable
from .output import Output

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .input import Inputs
    from .node import Node
    from .node_eval import NodeEval
    from .output import Outputs


class NodeStructure:
    __slots__ = (
        "_node",
        "_inputs",
        "_outputs",
        "_evaluator",
        "_allowed_kw_inputs",
    )

    _node: Node
    _inputs: Inputs
    _outputs: Outputs
    _evaluator: NodeEval
    _allowed_kw_inputs: tuple

    def __init__(
        self,
        *,
        node: Node,
        inputs: Inputs,
        outputs: Outputs,
        evaluator: NodeEval,
        allowed_kw_inputs: tuple = (),
    ) -> None:
        self._node = node
        self._inputs = inputs
        self._outputs = outputs
        self._evaluator = evaluator
        self._allowed_kw_inputs = allowed_kw_inputs

    #
    # I/O management
    #
    def make_or_get_input(self, name=None, *args, **kwargs) -> Input | None:
        """Returns an existing input by `name`, else try to create new one. If
        `name` is given, creates an input by the default way, otherwise tries
        to use `input_strategy`. If `name` is not given simply uses the input
        handler.

        .. note:: creation of a new input is restricted for a *closed* graph
        """
        node = self._node
        if name is None:
            node._logger.debug(
                f"Node '{node._name}': Try to create an input with `input_strategy`"
            )
            if not self._evaluator.closed:
                return self.make_input(*args, **kwargs)
            raise ClosedGraphError(node=node)

        node._logger.debug(f"Node '{node._name}': Try to get/create the input '{name}'")
        inp = self._inputs.get(name, None)
        kwargs.setdefault("positional", False)
        if inp is None:
            inp = self.add_input(name, **kwargs)
        elif isinstance(inp, Input) and inp.connected and (output := inp.parent_output):
            raise ReconnectionError(input=inp, node=node, output=output)
        return inp

    def make_input(self, *args, exception=True, **kwargs) -> Input | None:
        """Creates a single input via an input handler."""
        try:
            return self._node.input_strategy(*args, **kwargs)
        except Exception as exc:
            if exception:
                raise RuntimeError(
                    "Unable to make an input automatically as no handler is set"
                ) from exc
            return None

    def add_input(self, name: str, **kwargs) -> Input:
        """Creates a single input with name, if the graph is not closed.

        If the node has `allowed_kw_inputs` checks whether the name in the `allowed_kw_inputs`
        and then directly creates a new input.
        Else the node has no `allowed_kw_inputs` firstly tries to use an input handler.
        If there is no input handler the method creates a new input directly.
        """
        node = self._node
        if self._evaluator.closed:
            raise ClosedGraphError(node=node)
        if self._allowed_kw_inputs:
            if name not in self._allowed_kw_inputs:
                raise CriticalError(
                    (
                        f"Cannot create an input with {name=} due to the name is not in the "
                        f"allowed_kw_inputs={self._allowed_kw_inputs}"
                    ),
                    node=node,
                )
            return self._add_input(name, **kwargs)
        if (inp := self.make_input(exception=False)) is None:
            inp = self._add_input(name, **kwargs)
        return inp

    def _add_inputs(self, name: Sequence[str], **kwargs) -> tuple[Input, ...]:
        """Creates a sequence of inputs.

        .. note:: there is no check whether the graph is closed or not.
        """
        if IsIterable(name):
            return tuple(self._add_input(n, **kwargs) for n in name)
        raise CriticalError(
            f"'name' of the input must be `Sequence[str]`, but given {name}",
            node=self._node,
        )

    def _add_input(
        self,
        name: str,
        *,
        positional: bool = True,
        keyword: bool = True,
        **kwargs,
    ) -> Input:
        """Creates a new single input if there is no input with `name`.

        .. note:: there is no check whether the graph is closed or not.
        """
        node = self._node
        node._logger.debug(f"Node '{node._name}': Add input '{name}'")
        if name in self._inputs:
            raise ReconnectionError(input=name, node=node)
        inp = Input(name, node, **kwargs)
        self._inputs.add(inp, positional=positional, keyword=keyword)
        self._evaluator.allocated = False
        return inp

    def add_output(
        self,
        name: str,
        *,
        keyword: bool = True,
        positional: bool = True,
        **kwargs,
    ) -> Output:
        """Creates a new single output if there is no output with `name` and
        the graph is not closed."""
        if self._evaluator.closed:
            raise ClosedGraphError(node=self._node)
        return self._add_output(name, keyword=keyword, positional=positional, **kwargs)

    def _add_output(
        self,
        name: str,
        *,
        keyword: bool = True,
        positional: bool = True,
        **kwargs,
    ) -> Output:
        """Creates a new single output if there is no output with `name`.

        .. note:: there is no check whether the graph is closed or not.
        """
        node = self._node
        node._logger.debug(f"Node '{node._name}': Add output '{name}'")
        if name in self._outputs:
            raise ReconnectionError(output=name, node=node)
        out = Output(name, node, **kwargs)
        self._outputs.add(out, positional=positional, keyword=keyword)
        self._evaluator.allocated = False
        return out

    def _add_outputs(self, name: Sequence[str], **kwargs) -> tuple[Output, ...]:
        """Creates a sequence of outputs.

        .. note:: there is no check whether the graph is closed or not.
        """
        if IsIterable(name):
            return tuple(self._add_output(n, **kwargs) for n in name)
        raise CriticalError(
            f"'name' of the output must be `Sequence[str]`, but given {name=}",
            node=self._node,
        )

    def add_pair(
        self, iname: str, oname: str, **kwargs
    ) -> tuple[Input | tuple[Input, ...], Output | tuple[Output, ...]]:
        """Creates a pair of input and output."""
        if self._evaluator.closed:
            raise ClosedGraphError(node=self._node)
        return self._add_pair(iname, oname, **kwargs)

    def _add_pairs(
        self,
        inames: Sequence[str],
        onames: Sequence[str],
        input_kws: dict | None = None,
        output_kws: dict | None = None,
    ) -> tuple[list[Input], list[Output]]:
        """Creates sequence of pairs of input and output.

        .. note:: the inputs and outputs count must be the same
        """
        if len(inames) != len(onames):
            raise CriticalError(
                (
                    f"Cannot add pairs of input/output due to different lenght of {inames=} and"
                    f" {onames=}"
                ),
                node=self._node,
            )
        inputs, outputs = [], []
        for iname, oname in zip(inames, onames):
            input, output = self._add_pair(iname, oname, input_kws, output_kws)
            inputs.append(input)
            outputs.append(output)
        return inputs, outputs

    def _add_pair(
        self,
        iname: str,
        oname: str,
        input_kws: dict | None = None,
        output_kws: dict | None = None,
    ) -> tuple[Input | tuple[Input, ...], Output | tuple[Output, ...]]:
        """Creates a pair of input and output.

        .. note:: there is no check whether the graph is closed or not.
        """
        input_kws = input_kws or {}
        output_kws = output_kws or {}
        out = self._add_output(oname, **output_kws)
        inp = self._add_input(iname, child_output=out, **input_kws)
        self._evaluator.allocated = False
        return inp, out

    #
    # Graph lifecycle
    #
    def update_types(self, update_parents: bool = True):
        if not self._evaluator.types_tainted:
            return True
        node = self._node
        if update_parents:
            node._logger.debug(
                f"Node '{node._name}': Trigger update_parents update types..."
            )
            for input in self._inputs.iter_all():
                if not input.connected():
                    raise ClosingError("Input is not connected", node=node, input=input)
                input.parent_node.update_types(update_parents)
        node._logger.debug(f"Node '{node._name}': Update types...")
        node._type_function()
        self._evaluator.types_tainted = False
        self._evaluator.needs_reallocation = True

    def allocate(self, allocate_parents: bool = True):
        if self._evaluator.allocated and not self._evaluator.needs_reallocation:
            return True
        node = self._node
        if allocate_parents:
            node._logger.debug(
                f"Node '{node._name}': Trigger allocate_parents memory allocation..."
            )
            for _input in self._inputs.iter_all():
                try:
                    parent_node = _input.parent_node
                except AttributeError as exc:
                    raise ClosingError(
                        "Parent node is not initialized", input=_input
                    ) from exc
                if not parent_node.allocate(allocate_parents):
                    return False
        node._logger.debug(f"Node '{node._name}': Allocate memory on inputs")
        input_reassigned = self._inputs.allocate()
        node._logger.debug(f"Node '{node._name}': Allocate memory on outputs")
        output_reassigned = self._outputs.allocate()
        node._logger.debug(f"Node '{node._name}': Post allocate")
        if input_reassigned or output_reassigned or self._evaluator.needs_post_allocate:
            node._post_allocate()
        self._evaluator.allocated = True
        self._evaluator.needs_reallocation = False
        return True

    def close(
        self,
        *,
        close_parents: bool = True,
        strict: bool = True,
        close_children: bool = False,
        together: Sequence[Node] = [],
    ) -> bool:
        # Caution: `together` list should not be written in!
        node = self._node
        evaluator = self._evaluator

        if evaluator.closed:
            return True
        if evaluator.invalid:
            raise ClosingError("Cannot close an invalid node!", node=node)
        node._logger.debug(f"Node '{node._name}': Trigger recursive close")
        for n in [node] + together:
            try:
                n.update_types(update_parents=close_parents)
            except ClosingError:
                if strict:
                    raise
        for n in [node] + together:
            try:
                n.allocate(allocate_parents=close_parents)
            except ClosingError:
                if strict:
                    raise
        if close_parents:
            for _input in self._inputs.iter_all():
                if not _input.parent_node.close(close_parents=close_parents):
                    return False
        for n in together:
            if not n.close(close_parents=close_parents):
                return False
        evaluator.closed = evaluator.allocated
        if strict and not evaluator.closed:
            raise ClosingError(node=node)

        if close_children:
            for output in self._outputs:
                for input in output.child_inputs:
                    input.node.close(close_children=True, strict=strict)

        node._logger.debug(
            f"Node '{node._name}': {evaluator.closed and 'closed' or 'failed to close'}"
        )
        return evaluator.closed

    def open(self, *, open_children: bool = False, force_taint: bool = False) -> bool:
        node = self._node
        evaluator = self._evaluator

        if not evaluator.closed and not force_taint:
            return True
        node._logger.debug(f"Node '{node._name}': Open")
        if open_children:
            for output in self._outputs:
                for _input in output.child_inputs:
                    if not _input.node.open(
                        force_taint=force_taint, open_children=open_children
                    ):
                        raise OpeningError(node=node, output=output)
        node.unfreeze()
        node.taint()
        evaluator.closed = False
        return not evaluator.closed
