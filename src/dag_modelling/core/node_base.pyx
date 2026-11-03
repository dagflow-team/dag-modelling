# cython: language_level=3, boundscheck=False, wraparound=False
from .input_strategy import InputStrategyBase
from .exception import InitializationError
from .node_base_methods import _NodeBaseMethods


cdef class NodeBase:
    # cdef public fields declared in node_base.pxd.

    def __init__(self, inputs=None, outputs=None, input_strategy=None):
        _NodeBaseMethods.__init__(self, inputs, outputs, input_strategy)

    def __getitem__(self, key):
        return _NodeBaseMethods.__getitem__(self, key)

    def __str__(self):
        return _NodeBaseMethods.__str__(self)

    def __rshift__(self, other):
        return _NodeBaseMethods.__rshift__(self, other)

    def __rrshift__(self, other):
        return _NodeBaseMethods.__rrshift__(self, other)

    def __lshift__(self, storage):
        return _NodeBaseMethods.__lshift__(self, storage)

    # Cold non-dunder API via class-attribute binding.
    get = _NodeBaseMethods.get
    _repr_pretty_ = _NodeBaseMethods._repr_pretty_
    print = _NodeBaseMethods.print
    get_data = _NodeBaseMethods.get_data
    get_data_dense = _NodeBaseMethods.get_data_dense
    get_input_data = _NodeBaseMethods.get_input_data
    to_dict = _NodeBaseMethods.to_dict

    # Kept inline (not in _NodeBaseMethods) so Cython compiles a tp_getset
    # accessor instead of a Python descriptor.
    @property
    def input_strategy(self):
        return self._input_strategy

    @input_strategy.setter
    def input_strategy(self, new_input_strategy):
        if new_input_strategy is None:
            self._input_strategy = InputStrategyBase()
        elif isinstance(new_input_strategy, InputStrategyBase):
            self._input_strategy = new_input_strategy
            self._input_strategy.node = self
        elif issubclass(new_input_strategy, InputStrategyBase):
            self._input_strategy = new_input_strategy(node=self)
        else:
            from .input_strategy import InputStrategies

            raise InitializationError(
                f"Wrong {new_input_strategy=}! Must be in {InputStrategies}", node=self
            )
