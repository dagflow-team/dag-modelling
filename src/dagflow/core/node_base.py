from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence
from typing import TYPE_CHECKING

from nested_mapping import NestedMapping

from ..core.labels import repr_pretty
from ..tools.logger import logger  # TODO: bad thing due to Node has logger
from .exception import ConnectionError, InitializationError
from .input import Inputs
from .input_strategy import InputStrategyBase
from .output import Output, Outputs

if TYPE_CHECKING:
    from ..parameters import Parameter


class NodeBase:
    __slots__ = ("inputs", "outputs", "_input_strategy")
    inputs: Inputs
    outputs: Outputs

    def __init__(self, inputs=None, outputs=None, input_strategy=None):
        self.input_strategy = input_strategy
        self.inputs = Inputs(inputs)
        self.outputs = Outputs(outputs)

    @property
    def input_strategy(self):
        return self._input_strategy

    @input_strategy.setter
    def input_strategy(self, new_input_strategy):
        if new_input_strategy is None:
            # initialize with default strategy
            self._input_strategy = InputStrategyBase()
        elif isinstance(new_input_strategy, InputStrategyBase):
            # if `new_input_strategy` is an certain implementation of the input strategy
            self._input_strategy = new_input_strategy
            self._input_strategy.node = self
        elif issubclass(new_input_strategy, InputStrategyBase):
            # if `new_input_strategy` is an type (not instance!) inherited from `InputStrategyBase`
            self._input_strategy = new_input_strategy(node=self)
        else:
            from .input_strategy import InputStrategies

            raise InitializationError(
                f"Wrong {new_input_strategy=}! Must be in {InputStrategies}", node=self
            )

    def __getitem__(self, key):
        if isinstance(key, (int, slice, str)):
            return self.outputs[key]
        if (y := len(key)) != 2:
            raise ValueError(f"NodeBase key should be of length 2, but given {y}!")
        ikey, okey = key
        if ikey and okey:
            if isinstance(ikey, (int, str)):
                ikey = (ikey,)
            if isinstance(okey, (int, str)):
                okey = (okey,)
            return NodeBase(
                self.inputs[ikey],
                self.outputs[okey],
                input_strategy=self._input_strategy,
            )
        if ikey:
            return self.inputs[ikey]
        if okey:
            return self.outputs[okey]
        raise ValueError("Empty keys specified")

    def get(self, key, default=None):
        try:
            return self.__getitem__(key)
        except Exception:
            return default

    def __str__(self) -> str:
        return f"→[{len(self.inputs)}]{type(self).__name__}[{len(self.outputs)}]→"

    _repr_pretty_ = repr_pretty

    def print(self):
        for i, input in enumerate(self.inputs):
            print(i, input)
        for i, output in enumerate(self.outputs):
            print(i, output)

    def __rshift__(self, other):
        """
        `self >> other`

        The method to connect `Node.outputs` to `other`.
        Now the connection is allowed only for the `Node` with only one output!
        """
        if self.outputs.len_all() == 1:
            out = tuple(self.outputs.all_edges.values())[0]
        else:
            raise ConnectionError(
                f"The connection of {type(self)} >> {type(other)} is not supported!"
                " The connection `Node >> Node` is supported only nodes with only 1 output!",
                node=self,
            )
        out >> other

    def __rrshift__(self, other: Sequence | Generator):
        """
        `other >> self`

        The method connects `Sequence[Output | Parameter | NodeBase]` to `Node`.
        The connection to the `Node` is allowed only if it contains a single output.
        """
        if not isinstance(other, (Sequence, Generator)):
            raise ConnectionError(
                f"The connection {type(other)=} >> {type(self)=} is not supported",
                node=self,
            )
        from ..parameters import Parameter

        idx_scope = self._input_strategy._idx_scope + 1
        for out in other:
            if isinstance(out, Output):
                out._connect_to_node(self, idx_scope=idx_scope, reassign_idx_scope=False)
            elif isinstance(out, Parameter):
                out._connectible_output._connect_to_node(
                    self, idx_scope=idx_scope, reassign_idx_scope=False
                )
            elif isinstance(out, NodeBase):
                outs = out.outputs
                if outs.len_all() != 1:
                    raise ConnectionError(
                        "The connection `Node >> Node` is supported for nodes with only 1 output!",
                        node=out,
                        output=outs,
                    )
                outs[0]._connect_to_node(self, idx_scope=idx_scope, reassign_idx_scope=False)
            else:
                raise ConnectionError(
                    f"The connection `Sequence[{type(out)}] >> Node` is not allowed!",
                    node=self,
                )
        self._input_strategy._idx_scope = idx_scope

    def __lshift__(self, storage: Mapping[str, Output | Parameter] | NestedMapping) -> None:
        """
        self << other

        For each not connected input try to find output with the same name in storage, then connect.
        """
        if not isinstance(storage, (Mapping, NestedMapping)):
            raise ConnectionError(
                f"Cannot connect `Node << {type(storage)}` due to such connection is not supported!",
                node=self,
            )

        from ..parameters import Parameter

        for name, inputs in self.inputs.all_edges.items():
            output = storage.get(name, None)
            if output is None:
                continue
            elif not isinstance(output, (Output, Parameter)):
                raise ConnectionError('[<<] invalid "output"', input=inputs, output=output)

            if not isinstance(inputs, (Sequence, Generator)):
                inputs = (inputs,)

            for input in inputs:
                if not input.connected():
                    # TODO: maybe we set logger as a field?
                    #       do we need this logging actually?
                    logger.debug(f"[<<] connect {name}")
                    output >> input

    #
    # Accessors
    #
    def get_data(self, key=0):
        return self.outputs[key].data

    def get_input_data(self, key):
        return self.inputs[key].data()

    def to_dict(self, **kwargs) -> dict:
        return self.outputs[0].to_dict(**kwargs)
