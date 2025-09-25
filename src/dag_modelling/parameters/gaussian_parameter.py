from __future__ import annotations

from typing import TYPE_CHECKING

from .parameter import Parameter

if TYPE_CHECKING:
    from ..core.output import Output


class GaussianParameter(Parameter):
    __slots__ = ("_central_output", "_sigma_output", "_normvalue_output")
    _central_output: Output
    _sigma_output: Output
    _normvalue_output: Output

    def __init__(
        self,
        value_output: Output,
        central_output: Output,
        sigma_output: Output,
        idx: int | None = None,
        *,
        normvalue_output: Output,
        **kwargs,
    ):
        super().__init__(value_output, idx, **kwargs)
        self._central_output = central_output
        self._sigma_output = sigma_output
        self._normvalue_output = normvalue_output

    def __str__(self) -> str:
        self.central
        if self.central != 0:
            return (
                f"gpar {self.name} v={self.value} ({self.normvalue}σ):"
                f" {self.central}±{self.sigma} ({self.sigma_percent:.2g}%)"
            )
        else:
            return f"gpar {self.name} v={self.value} ({self.normvalue}σ): {self.central}±{self.sigma}"

    @property
    def central(self) -> float:
        return self._central_output.data[self._idx]

    @central.setter
    def central(self, central: float):
        self._central_output.seti(self._idx, central)

    @property
    def central_output(self) -> Output:
        return self._central_output

    @property
    def sigma(self) -> float:
        return self._sigma_output.data[self._idx]

    @sigma.setter
    def sigma(self, sigma: float):
        self._sigma_output.seti(self._idx, sigma)

    @property
    def sigma_relative(self) -> float:
        return self.sigma / self.value

    @sigma_relative.setter
    def sigma_relative(self, sigma_relative: float):
        self.sigma = sigma_relative * self.value

    @property
    def sigma_percent(self) -> float:
        return 100.0 * (self.sigma / self.value)

    @sigma_percent.setter
    def sigma_percent(self, sigma_percent: float):
        self.sigma = (0.01 * sigma_percent) * self.value

    @property
    def sigma_output(self) -> Output:
        return self._sigma_output

    @property
    def normvalue(self) -> float:
        return self._normvalue_output.data[self._idx]

    @normvalue.setter
    def normvalue(self, normvalue: float):
        self._normvalue_output.seti(self._idx, normvalue)

    def to_dict(self, **kwargs) -> dict:
        dct = super().to_dict(**kwargs)
        dct.update(
            {
                "central": self.central,
                "sigma": self.sigma,
                # 'normvalue': self.normvalue,
            }
        )
        if self.is_correlated:
            dct["flags"] += "C"
        return dct


class NormalizedGaussianParameter(Parameter):
    __slots__ = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, make_view=False, **kwargs)

    def __str__(self) -> str:
        return f"ngpar {self.name} v={self.value}"

    @property
    def central(self) -> float:
        return 0.0

    @property
    def sigma(self) -> float:
        return 1.0

    @property
    def normvalue(self) -> float:
        return self.value

    @normvalue.setter
    def normvalue(self, normvalue: float):
        self.value = normvalue

    def to_dict(self, **kwargs) -> dict:
        dct = super().to_dict(**kwargs)
        dct.update(
            {
                "central": 0.0,
                "sigma": 1.0,
            }
        )
        return dct


AnyGaussianParameter = GaussianParameter | NormalizedGaussianParameter
