from ...core.type_functions import AllPositionals, check_node_has_inputs, copy_dtype_from_inputs_to_outputs
from ..abstract import OneToOneNode


class ArraySum(OneToOneNode):
    """
    inputs:
        `array`: array to sum
        `array_1`: array to sum
        `...`: ...

    outputs:
        `0`, `...`: sum

    The node performs sum of elements of input arrays
    """

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "Σᵢ")

    def _type_function(self) -> None:
        check_node_has_inputs(self, AllPositionals)
        copy_dtype_from_inputs_to_outputs(self, AllPositionals, AllPositionals)
        for out in self.outputs:
            out.dd.shape = (1,)

    def _function(self):
        for indata, outdata in zip(self.inputs.iter_data(), self.outputs.iter_data_unsafe()):
            outdata[0] = indata.sum()
