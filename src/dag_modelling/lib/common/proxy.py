from numpy import copyto

from ..abstract import ManyToOneNode


class Proxy(ManyToOneNode):
    """Proxy inputs"""

    __slots__ = ("_idx",)
    _idx: int

    def __init__(self, *args, **kwargs):
        self._idx = 0
        super().__init__(*args, **kwargs)
        self._fd.needs_post_allocate = True
        self._labels.setdefault("mark", "proxy")

    def _function(self):
        self._input_nodes_callbacks[self._idx]()
        copyto(self._output_data, self._input_data[self._idx])

    def switch_input(self, idx: int) -> None:
        if self._idx == idx:
            return
        self._idx = idx
        self.taint()
