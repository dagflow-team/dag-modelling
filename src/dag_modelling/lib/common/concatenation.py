from ...core.input_strategy import AddNewInputAddAndKeepSingleOutput
from ...core.type_functions import check_node_has_inputs, check_dimension_of_inputs, check_dtype_of_inputs
from ..abstract import ManyToOneNode


class Concatenation(ManyToOneNode):
    """
    Creates a node with a single data output which is a concatenated data of the inputs.
    Now supports only 1d arrays.
    """

    __slots__ = (
        "_offsets",
        "_sizes",
    )

    _offsets: tuple[int,...]
    _sizes: tuple[int,...]

    def __init__(self, name, **kwargs):
        kwargs.setdefault("input_strategy", AddNewInputAddAndKeepSingleOutput(output_fmt="result"))
        super().__init__(name, **kwargs)
        self._offsets = ()
        self._sizes = ()

    def _type_function(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        check_node_has_inputs(self)
        cdtype = self.inputs[0].dd.dtype
        check_dtype_of_inputs(self, slice(None), dtype=cdtype)
        check_dimension_of_inputs(self, slice(None), 1)
        _output = self.outputs["result"]

        offset = 0
        offsets, sizes = [], []
        for input in self.inputs:
            offsets.append(offset)
            newsize = input.dd.shape[0]
            offset += newsize
            sizes.append(newsize)
        _output.dd.shape = (offset,)
        _output.dd.dtype = cdtype

        self._offsets = tuple(offsets)
        self._sizes = tuple(sizes)

    def _function(self):
        _output = self.outputs["result"]
        data = _output._data
        for offset, _input in zip(self._offsets, self.inputs):
            size = _input.dd.shape[0]
            data[offset : offset + size] = _input.data[:]

    @property
    def sizes(self) -> list[int]:
        return self._sizes

