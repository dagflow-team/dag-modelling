from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.typing import NDArray

from nested_mapping.typing import properkey

from ...core.input_strategy import AddNewInputAddNewOutput
from ...core.node import Node
from ...core.storage import NodeStorage
from ...core.type_functions import (
    assign_axes_from_inputs_to_outputs,
    check_node_has_inputs,
    copy_from_inputs_to_outputs,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from nested_mapping.typing import KeyLike


class OneToOneNode(Node):
    """
    The abstract node with an output for every positional input
    """

    __slots__ = ("_input_output_data",)

    _input_output_data: list[tuple[NDArray, NDArray]]

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("input_strategy", AddNewInputAddNewOutput())
        super().__init__(*args, **kwargs)

        self._input_output_data = []

    def _type_function(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        check_node_has_inputs(self)
        copy_from_inputs_to_outputs(self, slice(None), slice(None), edges=True, meshes=True)
        assign_axes_from_inputs_to_outputs(
            self,
            slice(None),
            slice(None),
            assign_meshes=True,
            ignore_assigned=True,
            ignore_Nd=True,
            merge_input_axes=False,
        )

    def _post_allocate(self):
        super()._post_allocate()
        for input, output in zip(self.inputs, self.outputs):
            self._input_output_data.append((input._data, output._data))

    @classmethod
    def replicate(
        cls,
        *args: NodeStorage | Any,
        name: str,
        replicate_outputs: Sequence[KeyLike] | None = None,
        **kwargs,
    ) -> tuple[Node | None, NodeStorage]:
        if args:
            if replicate_outputs is not None:
                raise RuntimeError(
                    "OneToOneNode.replicate_outputs can use either `args` or `replicate_outputs`"
                )

            return cls.replicate_from_args(name, *args, **kwargs)

        if replicate_outputs is None:
            replicate_outputs = ((),)

        return cls.replicate_from_indices(name, replicate_outputs=replicate_outputs, **kwargs)

    @classmethod
    def replicate_from_indices(
        cls,
        name: str,
        replicate_outputs: Sequence[KeyLike] = ((),),
        verbose: bool = False,
        **kwargs,
    ) -> tuple[Node | None, NodeStorage]:
        storage = NodeStorage(default_containers=True)
        nodes = storage("nodes")
        inputs = storage("inputs")
        outputs = storage("outputs")

        if not replicate_outputs:
            raise RuntimeError("`replicate_outputs` tuple should have at least one item")

        tuplename = (name,)
        for outkey in replicate_outputs:
            outkey = properkey(outkey)
            outname = tuplename + outkey
            instance = cls(".".join(outname), **kwargs)
            nodes[outname] = instance
            instance()

            iter_inputs = instance.inputs.iter_all_items()
            ninputs = instance.inputs.len_all()
            if ninputs > 1:
                for iname, input in iter_inputs:
                    inputs[tuplename + (iname,) + outkey] = input
            else:
                _, input = next(iter_inputs)
                inputs[tuplename + outkey] = input

            iter_outputs = instance.outputs.iter_all_items()
            noutputs = instance.outputs.len_all()
            if noutputs > 1:
                for oname, output in instance.outputs.iter_all_items():
                    outputs[tuplename + (oname,) + outkey] = output
            else:
                _, output = next(iter_outputs)
                outputs[tuplename + outkey] = output

        NodeStorage.update_current(storage, strict=True, verbose=verbose)

        return (instance, storage) if len(replicate_outputs) == 1 else (None, storage)

    @classmethod
    def replicate_from_args(
        cls,
        name: str,
        *args: NodeStorage | Any,
        verbose: bool = False,
        **kwargs,
    ) -> tuple[Node | None, NodeStorage]:
        storage = NodeStorage(default_containers=True)
        nodes = storage("nodes")
        inputs = storage("inputs")
        outputs = storage("outputs")

        from nested_mapping import walkitems

        nobjects = 0
        tuplename = (name,)
        for arg in args:
            for outkey, obj in walkitems(arg):
                outname = tuplename + outkey
                instance = cls(".".join(outname), **kwargs)
                nodes[outname] = instance

                obj >> instance

                iter_inputs = instance.inputs.iter_all_items()
                ninputs = instance.inputs.len_all()
                if ninputs > 1:
                    for iname, input in iter_inputs:
                        inputs[tuplename + (iname,) + outkey] = input
                else:
                    _, input = next(iter_inputs)
                    inputs[tuplename + outkey] = input

                outputs[outname] = instance.outputs[0]
                iter_outputs = instance.outputs.iter_all_items()
                noutputs = instance.outputs.len_all()
                if noutputs > 1:
                    for oname, output in instance.outputs.iter_all_items():
                        outputs[tuplename + (oname,) + outkey] = output
                else:
                    _, output = next(iter_outputs)
                    outputs[tuplename + outkey] = output

                nobjects += 1

        NodeStorage.update_current(storage, strict=True, verbose=verbose)

        return (instance, storage) if nobjects == 1 else (None, storage)
