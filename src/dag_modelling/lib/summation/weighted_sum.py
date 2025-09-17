from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import copyto

from ...core.exception import TypeFunctionError
from ...core.type_functions import check_node_has_inputs, copy_shape_from_inputs_to_outputs, evaluate_dtype_of_outputs
from ..abstract import ManyToOneNode

if TYPE_CHECKING:
    from ...core.input import Input


class WeightedSum(ManyToOneNode):
    """Weighted sum of all the inputs together"""

    __slots__ = ("_weight",)
    _weight: Input

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs, allowed_kw_inputs=("weight",))
        self._labels.setdefault("mark", "Σᵢwᵢa⃗ᵢ")
        self._weight = self._add_input("weight", positional=False)
        self._functions_dict.update({"number": self._fcn_number, "iterable": self._fcn_iterable})

    def _type_function(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        super()._type_function()
        check_node_has_inputs(self, "weight")
        weight = self._weight
        shape = weight.dd.shape[0]
        leninp = len(self.inputs)
        if shape == 0:
            raise TypeFunctionError(
                "Cannot use WeightedSum with empty 'weight'!",
                node=self,
                input=weight,
            )
        elif shape == 1:
            self.function = self._functions_dict["number"]
        elif shape == leninp:
            self.function = self._functions_dict["iterable"]
        else:
            raise TypeFunctionError(
                f"The number of weights (={shape}) must coincide "
                f"with the number of inputs (={leninp})!",
                node=self,
                input=weight,
            )
        copy_shape_from_inputs_to_outputs(self, 0, "result")
        evaluate_dtype_of_outputs(self, slice(None), "result")

    def _fcn_number(self):
        """
        The function for one weight for all inputs:
        `len(weight) == 1`
        """
        out = self.outputs[0]._data
        weight = self._weight.data
        copyto(out, self.inputs[0].data.copy())
        for _input in self.inputs[1:]:
            out += _input.data
        out *= weight

    def _fcn_iterable(self):
        """
        The function for one weight for every input:
        `len(weight) == len(inputs)`
        """
        out = self.outputs[0]._data
        weights = self._weight.data
        copyto(out, self.inputs[0].data * weights[0])
        for _input, weight in zip(self.inputs[1:], weights[1:]):
            out += _input.data * weight
