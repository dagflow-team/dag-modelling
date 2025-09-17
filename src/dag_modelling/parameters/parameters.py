from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from numpy import array, ndarray

from ..core.exception import InitializationError
from ..lib.common import Array
from .parameter import Parameter

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping

    from numpy.typing import ArrayLike, DTypeLike

    from ..core.node import Node
    from ..core.output import Output
    from .gaussian_parameter import NormalizedGaussianParameter


class Parameters:
    __slots__ = (
        "value",
        "_value_node",
        "_pars",
        "_names",
        "_norm_pars",
        "_is_variable",
        "_constraint",
    )
    value: Output
    _value_node: Node
    _pars: list[Parameter]
    _names: tuple[tuple[str, ...], ...]
    _norm_pars: list[NormalizedGaussianParameter]

    _is_variable: bool

    _constraint: Constraint | None

    def __init__(
        self,
        names: Sequence[Sequence[str] | str],
        value: Node,
        *,
        variable: bool | None = None,
        fixed: bool | None = None,
        close: bool = True,
        label: Mapping = {},
    ):
        self._value_node = value
        try:
            self.value = value.outputs[0]
        except IndexError as exc:
            raise InitializationError("Parameters: value node has no outputs") from exc

        if all(f is not None for f in (variable, fixed)):
            raise RuntimeError("Parameter may not be set to variable and fixed at the same time")
        if variable is not None:
            self._is_variable = variable
        elif fixed is not None:
            self._is_variable = not fixed
        else:
            self._is_variable = True

        self._constraint = None

        self._pars = []
        self._names = tuple((name,) if isinstance(name, str) else name for name in names)
        self._norm_pars = []
        if close:
            self._close()

            npars = self.value.dd.size
            if npars > 1:
                for i in range(npars):
                    ilabel = label.get(names[i], {})
                    self._pars.append(Parameter(self.value, i, label=ilabel, parent=self))
            elif npars == 1:
                self._pars.append(Parameter(self.value, label=label, parent=self))
            else:
                raise RuntimeError("Do not know how to handle 0 parameters")

    def _close(self) -> None:
        self._value_node.close(close_parents=True)

    @property
    def is_variable(self) -> bool:
        return self._is_variable

    @property
    def is_fixed(self) -> bool:
        return not self._is_variable

    @property
    def is_constrained(self) -> bool:
        return self._constraint is not None

    @property
    def is_free(self) -> bool:
        return self._constraint is None

    @property
    def is_correlated(self) -> bool:
        return False if self._constraint is None else self._constraint.is_correlated

    @property
    def parameters(self) -> list:
        return self._pars

    def outputs(self) -> tuple[Output, ...]:
        return tuple(par.output for par in self._pars)

    @property
    def norm_parameters(self) -> list:
        return self._norm_pars

    def iteritems(self) -> Generator[tuple[tuple[str, ...], Parameter], None, None]:
        yield from zip(self._names, self._pars)

    def iteritems_norm(self) -> Generator[tuple[tuple[str, ...], Parameter], None, None]:
        yield from zip(self._names, self._norm_pars)

    def _reset_pars(self) -> None:
        self._pars = []
        self._norm_pars = []

    @property
    def constraint(self) -> Constraint | None:
        return self._constraint

    def to_dict(self, *, label_from: str = "text") -> dict:
        return {
            "value": self.value.data[0],
            "label": self._value_node.labels[label_from],
            "flags": "",
        }

    def set_constraint(self, constraint: Constraint) -> None:
        if self._constraint is not None:
            raise InitializationError("Constraint already set")
        self._constraint = constraint
        # constraint._pars = self

    @staticmethod
    def from_numbers(
        value: float | int | ArrayLike,
        *,
        names: Sequence[Sequence[str] | str] = ((),),
        dtype: DTypeLike = "d",
        variable: bool | None = None,
        fixed: bool | None = None,
        label: Mapping[str, str] | None = None,
        central: float | int | ArrayLike | None = None,
        sigma: float | int | ArrayLike | None = None,
        **kwargs,
    ) -> Parameters:
        label = {"text": "parameter"} if label is None else dict(label)
        name: str = label.setdefault("name", "parameter")

        grouplabel = label.get("group", label)

        if isinstance(value, (float, int)):
            value = (value,)
        elif not isinstance(value, (Sequence, ndarray)):
            raise InitializationError(
                f"Parameters.from_numbers: Unsupported value type {type(value)}"
            )
        if len(names) != len(value):
            raise InitializationError(
                f"Parameters.from_numbers: inconsistent values ({value}) and names ({names})"
            )

        has_constraint = sigma is not None
        pars = Parameters(
            names,
            Array(
                name,
                array(value, dtype=dtype),
                label=grouplabel,
                mode="store_weak",
            ),
            label=label,
            fixed=fixed,
            variable=variable,
            close=not has_constraint,
        )

        if has_constraint:
            if central is None:
                central = value
            from .gaussian_parameters import GaussianConstraint

            pars.set_constraint(
                GaussianConstraint.from_numbers(
                    parameters=pars,
                    dtype=dtype,
                    label=grouplabel,
                    central=central,
                    sigma=sigma,
                    **kwargs,
                )
            )
            pars._close()

        return pars


class Constraint:
    __slots__ = ("_pars",)
    _pars: Parameters

    def __init__(self, parameters: Parameters):
        self._pars = parameters

    @property
    def is_correlated(self) -> bool:
        return False
