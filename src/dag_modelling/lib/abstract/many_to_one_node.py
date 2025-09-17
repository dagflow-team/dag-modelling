from __future__ import annotations

from typing import TYPE_CHECKING

from nested_mapping.typing import properkey

from ...core.input_strategy import AddNewInputAddAndKeepSingleOutput
from ...core.node import Node
from ...core.storage import NodeStorage
from ...core.type_functions import (
    AllPositionals,
    check_inputs_equivalence,
    check_inputs_number_is_divisible_by_N,
    check_node_has_inputs,
    copy_from_inputs_to_outputs,
    evaluate_dtype_of_outputs,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from numpy.typing import NDArray

    from nested_mapping.typing import KeyLike, TupleKey


class ManyToOneNode(Node):
    """The abstract node with only one output `result`, which is the result of
    some function of all the positional inputs."""

    __slots__ = (
        "_broadcastable",
        "_check_edges_contents",
        "_input_data0",
        "_input_data_other",
        "_input_data",
        "_output_data",
    )

    _broadcastable: bool
    _check_edges_contents: bool

    _input_data0: NDArray
    _input_data_other: list[NDArray]
    _input_data: list[NDArray]
    _output_data: NDArray

    def __init__(
        self,
        *args,
        broadcastable: bool = False,
        output_name: str = "result",
        check_edges_contents: bool = False,
        **kwargs,
    ):
        kwargs.setdefault(
            "input_strategy", AddNewInputAddAndKeepSingleOutput(input_fmt=self._input_names())
        )
        super().__init__(*args, **kwargs)
        self._add_output(output_name)
        self._broadcastable = broadcastable
        self._check_edges_contents = check_edges_contents

        self._input_data0 = None  # pyright: ignore [reportAttributeAccessIssue]
        self._input_data_other = []
        self._input_data = []
        self._output_data = None  # pyright: ignore [reportAttributeAccessIssue]

    @staticmethod
    def _input_names() -> tuple[str, ...]:
        return ("input",)

    @classmethod
    def _input_block_size(cls) -> int:
        return len(cls._input_names())

    def _type_function(self) -> None:
        """A output takes this function to determine the dtype and shape."""
        check_inputs_number_is_divisible_by_N(self, self._input_block_size())
        check_node_has_inputs(self)  # at least one input
        check_inputs_equivalence(
            self, broadcastable=self._broadcastable, check_edges_contents=self._check_edges_contents
        )  # all the inputs should have same dd fields
        copy_from_inputs_to_outputs(
            self,
            AllPositionals,
            "result",
            prefer_largest_input=self._broadcastable,
            prefer_input_with_edges=True,
        )  # copy shape to result
        evaluate_dtype_of_outputs(self, AllPositionals, "result")  # eval dtype of result

    def _post_allocate(self):
        super()._post_allocate()

        self._input_data = [input._data for input in self.inputs]
        self._input_data0, self._input_data_other = self._input_data[0], self._input_data[1:]
        self._output_data = self.outputs["result"]._data

    @classmethod
    def replicate(
        cls,
        *args: NodeStorage | Any,
        name: str,
        replicate_outputs: Sequence[KeyLike] = ((),),
        replicate_inputs: Sequence[KeyLike] | None = None,
        **kwargs,
    ) -> tuple[Node | None, NodeStorage]:
        if args and replicate_inputs is not None:
            raise RuntimeError(
                "ManyToOneNode.replicate_outputs can use either `args` or `replicate_inputs`"
            )

        if args:
            return cls.replicate_from_args(
                name, *args, replicate_outputs=replicate_outputs, **kwargs
            )

        if replicate_inputs:
            return cls.replicate_from_indices(
                name,
                replicate_outputs=replicate_outputs,
                replicate_inputs=replicate_inputs,
                **kwargs,
            )

        return cls.replicate_from_indices(name, replicate_outputs=replicate_outputs, **kwargs)

    @classmethod
    def replicate_from_args(
        cls,
        fullname: str,
        *args: NodeStorage | Any,
        replicate_outputs: Sequence[KeyLike] = ((),),
        allow_skip_inputs: bool = False,
        skippable_inputs_should_contain: Sequence[KeyLike] | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> tuple[Node | None, NodeStorage]:
        storage = NodeStorage(default_containers=True)
        nodes = storage("nodes")
        outputs = storage("outputs")

        if not replicate_outputs:
            raise RuntimeError("`replicate_outputs` tuple should have at least one item")

        instance = None
        outname = ""

        path = properkey(fullname, sep=".")

        def fcn_outer_before(outkey: TupleKey):
            nonlocal outname, instance
            outname = path + outkey
            instance = cls(".".join(outname), **kwargs)
            nodes[outname] = instance

        def function(i: int, inkey: TupleKey, outkey: TupleKey):
            container = args[i]
            output = container[inkey] if inkey else container
            try:
                output >> instance  # pyright: ignore [reportUnusedExpression]
            except TypeError as e:
                raise ConnectionError(
                    f"Invalid >> types for {inkey}/{outkey}: {type(output)}/{type(instance)}"
                ) from e

        def fcn_outer_after(_):
            outputs[outname] = instance.outputs[0]

        from nested_mapping import walkkeys
        from nested_mapping.tools import match_keys

        keys_left = tuple(tuple(walkkeys(arg)) for arg in args)
        match_keys(
            keys_left,
            replicate_outputs,
            function,
            fcn_outer_before=fcn_outer_before,
            fcn_outer_after=fcn_outer_after,
            require_all_left_keys_processed=not allow_skip_inputs,
            skippable_left_keys_should_contain=skippable_inputs_should_contain,
        )

        NodeStorage.update_current(storage, strict=True, verbose=verbose)

        return (instance, storage) if len(replicate_outputs) == 1 else (None, storage)

    @classmethod
    def replicate_from_indices(
        cls,
        fullname: str,
        *,
        replicate_outputs: Sequence[KeyLike] = ((),),
        replicate_inputs: Sequence[KeyLike] = ((),),
        verbose: bool = False,
        **kwargs,
    ) -> tuple[Node | None, NodeStorage]:
        storage = NodeStorage(default_containers=True)
        nodes = storage("nodes")
        outputs = storage("outputs")
        inputs = storage("inputs")

        if not replicate_outputs:
            raise RuntimeError("`replicate_outputs` tuple should have at least one item")

        instance = None
        outname = ""

        input_names = cls._input_names()

        path = properkey(fullname, sep=".")

        def fcn_outer_before(outkey: TupleKey):
            nonlocal outname, instance, nodes
            outname = path + outkey
            instance = cls(".".join(outname), **kwargs)
            nodes[outname] = instance

        def function(iarg: int, inkey: TupleKey, outkey: TupleKey):
            nonlocal inputs, instance, input_names
            for iname in input_names:
                input = instance()
                inputs[path + (iname,) + inkey] = input

        def fcn_outer_after(_):
            nonlocal outputs, outname, instance
            outputs[outname] = instance.outputs[0]

        from nested_mapping.tools import match_keys

        match_keys(
            (replicate_inputs,),
            replicate_outputs,
            function,
            fcn_outer_before=fcn_outer_before,
            fcn_outer_after=fcn_outer_after,
        )

        NodeStorage.update_current(storage, strict=True, verbose=verbose)

        return (instance, storage) if len(replicate_outputs) == 1 else (None, storage)
