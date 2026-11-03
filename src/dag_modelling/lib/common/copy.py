from __future__ import annotations

from typing import TYPE_CHECKING, override

from ...core.data_descriptor import Layout, check_layout
from ...core.type_functions import AllPositionals, check_dimension_of_inputs
from ...methods.sparse.csr import csr_copy, dense_to_csr
from ..abstract import OneToOneNode

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...core.types import LayoutLike


class Copy(OneToOneNode):
    """Copy/identity function, with `cache=True` it freezes after each evaluation

    extra arguments:
        `layout` (str): "dense" or "sparse-csr" for the outputs, the input layout by default
        `cache` (bool): False for `Copy`, True for `Cache`; `recache()` updates the data
    """

    __slots__ = ("_layout", "_cache", "_copy")
    _cache_default = False

    _layout: LayoutLike | None
    _cache: bool
    _copy: Callable[[], None]

    def __init__(
        self, *args, layout: LayoutLike | None = None, cache: bool | None = None, **kwargs
    ):
        # not super(): Cache is a copy of this class, not a subclass
        OneToOneNode.__init__(
            self, *args, allowed_layouts={AllPositionals: (Layout.SPARSE_CSR.value,)}, **kwargs
        )
        if layout is not None:
            check_layout(layout, self)
        self._layout = layout
        self._cache = self._cache_default if cache is None else cache
        self._labels.setdefault("mark", "cache" if self._cache else "copy")
        self._evaluator._functions_dict.update(
            {
                "csr": self._fcn_csr,
                "csr_to_dense": self._fcn_csr_to_dense,
                "dense_to_csr": self._fcn_dense_to_csr,
            }
        )

    def _function(self):
        for indata, outdata in zip(self.inputs.iter_data(), self.outputs.iter_data_unsafe()):
            outdata[:] = indata

        if self._cache:
            self._evaluator.frozen = True

    def _fcn_csr(self) -> None:
        for indata, outdata in zip(self.inputs.iter_data(), self.outputs.iter_data_unsafe()):
            csr_copy(indata, outdata)

        if self._cache:
            self._evaluator.frozen = True

    def _fcn_csr_to_dense(self) -> None:
        for indata, outdata in zip(self.inputs.iter_data(), self.outputs.iter_data_unsafe()):
            indata.toarray(out=outdata)

        if self._cache:
            self._evaluator.frozen = True

    def _fcn_dense_to_csr(self) -> None:
        for indata, outdata in zip(self.inputs.iter_data(), self.outputs.iter_data_unsafe()):
            dense_to_csr(indata, outdata)

        if self._cache:
            self._evaluator.frozen = True

    @override
    def _type_function(self) -> None:
        OneToOneNode._type_function(self)
        layout_in = self._copy_layout_from_inputs()
        layout = self._layout or layout_in
        if layout != layout_in:
            if layout == Layout.SPARSE_CSR:
                check_dimension_of_inputs(self, AllPositionals, ndim=2)
            for output in self.outputs:
                output.dd.layout = layout.value if isinstance(layout, Layout) else layout
                # a first guess for csr: the arrays grow on the first fill
                output.dd.capacity = output.dd.shape[0] if layout == Layout.SPARSE_CSR else None
        name = {
            (Layout.DENSE.value, Layout.DENSE.value): "default",
            (Layout.SPARSE_CSR.value, Layout.SPARSE_CSR.value): "csr",
            (Layout.SPARSE_CSR.value, Layout.DENSE.value): "csr_to_dense",
            (Layout.DENSE.value, Layout.SPARSE_CSR.value): "dense_to_csr",
        }[layout_in, layout]
        self._evaluator.function = self._evaluator._functions_dict[name]

    def recache(self) -> None:
        self.unfreeze()
        self.touch(force_computation=True)
