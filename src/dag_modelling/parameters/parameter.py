from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

from ..core.labels import repr_pretty

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..core.node import Node
    from ..core.output import Output
    from .parameters import Parameters


class Parameter:
    __slots__ = (
        "_idx",
        "_parent",
        "_common_output",
        "_value_output",
        "_view",
        "_common_connectible_output",
        "_connectible_output",
        "_labelfmt",
        "_stack",
    )
    _parent: Parameters | None
    _idx: int
    _common_output: Output
    _value_output: Output
    _view: Node | None
    _common_connectible_output: Output
    _connectible_output: Output | None
    _labelfmt: str
    _stack: list[float] | list[int]

    def __init__(
        self,
        value_output: Output,
        idx: int | None = None,
        *,
        parent: Parameters | None,
        connectible: Output | None = None,
        labelfmt: str = "{}",
        label: Mapping = {},
        make_view: bool = True,
    ):
        self._parent = parent
        self._common_output = value_output
        self._labelfmt = labelfmt

        if connectible is not None:
            self._common_connectible_output = connectible
        else:
            self._common_connectible_output = value_output

        if idx is None:
            self._idx = 0
            self._value_output = self._common_connectible_output
            self._connectible_output = self._common_connectible_output
            self._view = None
        elif make_view:
            self._idx = idx
            label_parent = self._common_output.node.labels.copy()
            if not label:
                label = label_parent
            try:
                idxtuple = parent._names[idx]
                idxname = ".".join(idxtuple)
            except (ValueError, IndexError):
                idxname = "???"
                idxtuple = None

            with suppress(KeyError, IndexError):
                label["paths"] = [label_parent["paths"][idx]]

            from ..lib.common import View  # fmt: skip

            self._view = View(
                f"{self._common_output.node.name}.{idxname}",
                self._common_connectible_output,
                start=idx,
                length=1,
            )
            self._view.labels.inherit(
                label,
                fmtlong=f"{{}} (par {idx}: {idxname})",
                fmtextra={"graph": f"{{source.text}}\\nparameter {idx}: {idxname}"},
            )
            self._value_output = self._view.outputs[0]
            self._connectible_output = self._value_output
        else:
            self._idx = idx
            self._view = None
            self._connectible_output = None
            self._value_output = value_output
        self._stack = []

    def __str__(self) -> str:
        return f"par {self.name} v={self.value}"

    _repr_pretty_ = repr_pretty

    @property
    def value(self) -> float | int:
        return self._common_output.data[self._idx]

    @value.setter
    def value(self, value: float | int):
        return self._common_output.seti(self._idx, value)

    @property
    def output(self) -> Output:
        return self._value_output

    @property
    def is_correlated(self) -> bool:
        return self._parent is not None and self._parent.is_correlated

    @property
    def connectible(self) -> Output | None:
        return self._connectible_output

    @property
    def name(self) -> str:
        if not self._connectible_output:
            return ""

        node = self._connectible_output.node
        labels = node.labels
        return labels.path or node.name or ""

    def connected(self) -> bool:
        return self._connectible_output and self._connectible_output.connected()

    def label(self, source: str = "text") -> str:
        return self._labelfmt.format(self._value_output.node.labels[source])

    def to_dict(self, *, label_from: str = "text") -> dict:
        return {"value": self.value, "label": self.label(label_from), "flags": ""}

    def __rshift__(self, other):
        if self._connectible_output is None:
            raise RuntimeError("Cannot connect a connectible output due to it is None!")
        self._connectible_output >> other

    def push(self, other: float | int | None = None) -> float | int:
        self._stack.append(self.value)
        if other is not None:
            if not isinstance(other, (float, int)):
                raise RuntimeError(
                    f"`other` must be float|int|None, but given {other=}, {type(other)=}"
                )
            self.value = other
        return self.value

    def pop(self) -> float | int:
        with suppress(IndexError):
            self.value = self._stack.pop()
        return self.value

    def __enter__(self) -> float | int:
        return self.push()

    def __exit__(self, exc_type, exc_val, exc_tb) -> float | int:
        return self.pop()
