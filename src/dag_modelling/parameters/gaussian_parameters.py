from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import array, ndarray, zeros_like

from ..core.exception import InitializationError
from ..core.node import Node
from ..lib.common import Array
from ..core.labels import inherit_labels
from .gaussian_parameter import GaussianParameter, NormalizedGaussianParameter
from .parameters import Constraint, Parameters

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import DTypeLike

    from ..core.output import Output


class GaussianConstraint(Constraint):
    __slots__ = (
        "central",
        "sigma",
        "normvalue",
        "sigma_total",
        "normvalue_final",
        "_central_node",
        "_sigma_node",
        "_normvalue_node",
        "_cholesky_node",
        "_covariance_node",
        "_correlation_node",
        "_sigma_total_node",
        "_norm_node",
        "_is_constrained",
    )
    central: Output
    sigma: Output
    normvalue: Output
    sigma_total: Output | None

    normvalue_final: Output

    _central_node: Node
    _sigma_node: Node
    _normvalue_node: Node

    _cholesky_node: Node | None
    _covariance_node: Node | None
    _correlation_node: Node | None
    _sigma_total_node: Node | None

    _norm_node: Node

    _is_constrained: bool

    def __init__(
        self,
        central: Node,
        *,
        parameters: Parameters,
        sigma: Node | None = None,
        covariance: Node | None = None,
        correlation: Node | None = None,
        constrained: bool | None = None,
        free: bool | None = None,
        provide_covariance: bool = False,
        **_,
    ):
        super().__init__(parameters=parameters)
        self._central_node = central

        self._cholesky_node = None
        self._covariance_node = None
        self._correlation_node = None
        self._sigma_total_node = None

        if all(f is not None for f in (constrained, free)):
            raise InitializationError(
                "GaussianConstraint may not be set to constrained and free at the same time"
            )
        if constrained is not None:
            self._is_constrained = constrained
        elif free is not None:
            self._is_constrained = not free
        else:
            self._is_constrained = True

        if sigma is not None and covariance is not None:
            raise InitializationError(
                'GaussianConstraint: got both "sigma" and "covariance" as arguments'
            )
        if correlation is not None and sigma is None:
            raise InitializationError(
                'GaussianConstraint: got "correlation", but no "sigma" as arguments'
            )

        value_node = parameters._value_node
        self._sigma_total_node = sigma
        if sigma is not None:
            self.sigma_total = sigma.outputs[0]
        if correlation is not None:
            from ..lib.linalg import Cholesky
            from ..lib.statistics import CovmatrixFromCormatrix

            self._correlation_node = correlation
            self._covariance_node = CovmatrixFromCormatrix(f"V({value_node.name})")
            self._cholesky_node = Cholesky(f"L({value_node.name})")
            self._sigma_node = self._cholesky_node

            self._sigma_total_node >> self._covariance_node.inputs["sigma"]
            correlation >> self._covariance_node
            self._covariance_node >> self._cholesky_node
        elif sigma is not None:
            self._sigma_node = sigma
            if provide_covariance:
                from ..lib.arithmetic import Square

                self._covariance_node = Square(f"σ²({value_node.name})")
                self._sigma_node >> self._covariance_node
        elif covariance is not None:
            from ..lib.linalg import Cholesky

            self._cholesky_node = Cholesky(f"L({value_node.name})")
            self._sigma_node = self._cholesky_node
            self._covariance_node = covariance

            covariance >> self._cholesky_node

            # Todo, add square root of the diagonal of the covariance matrix
            self.sigma_total = None
            self._sigma_total_node = None
        else:
            # TODO: no sigma/covariance AND central means normalized=value?
            raise InitializationError(
                'GaussianConstraint: got no "sigma" and no "covariance" arguments'
            )

        self.central = self._central_node.outputs[0]
        self.sigma = self._sigma_node.outputs[0]

        if (mark := value_node.labels.mark) is not None:
            normmark = f"norm({mark})"
        else:
            normmark = "norm"
        self._normvalue_node = Array(
            f"normal unit: {value_node.name}",
            zeros_like(self.central._data),
            mark=normmark,
            mode="store_weak",
        )
        self._normvalue_node.labels.inherit(self._pars._value_node.labels, fields_exclude={"paths"})
        self.normvalue = self._normvalue_node.outputs[0]

        from ..lib.statistics import NormalizeCorrelatedVarsTwoWays  # fmt: skip
        self._norm_node = NormalizeCorrelatedVarsTwoWays(
            f"[norm] {value_node.name}", immediate=True
        )
        self.central >> self._norm_node.inputs["central"]
        self.sigma >> self._norm_node.inputs["matrix"]

        fmts = {
            "_cholesky_node": ("Cholesky: {}", "L({})"),
            "_covariance_node": ("Covariance: {}", "V({})"),
            "_normvalue_node": ("{} (normal value)", "{}"),
        }
        for nodename in ("_cholesky_node", "_covariance_node", "_norm_node", "_sigma_node"):
            if cnode := getattr(self, nodename):
                cnode.labels.inherit(self._pars._value_node.labels, fields=("index_values",))
        for nodename in ("_normvalue_node", ): # TODO, clean inheritance labelling
            if (cnode := getattr(self, nodename)) is not None:
                fmtlong, fmtshort = fmts[nodename]
                cnode.labels.inherit(
                    self._pars._value_node.labels,
                    fmtlong=fmtlong,
                    fmtshort=fmtshort,
                )
        for nodename in ("_cholesky_node", "_covariance_node"):
            if (cnode := getattr(self, nodename)) is not None:
                fmtlong, fmtshort = fmts[nodename]
                cnode.labels.inherit(
                    self._pars._value_node.labels,
                    fmtlong=fmtlong,
                    fmtshort=fmtshort,
                    fields_exclude={"paths"},
                )

        (parameters.value, self.normvalue) >> self._norm_node
        self.normvalue_final = self._norm_node.outputs["normvalue"]

        self._norm_node.close(close_parents=True)
        self._norm_node.touch()

        value_output = self._pars.value
        self._pars._reset_pars()
        npars = value_output.dd.size
        for i in range(npars):
            ci = None if npars == 1 else i
            self._pars._pars.append(
                GaussianParameter(
                    value_output,
                    self.central,
                    self.sigma_total,
                    ci,
                    normvalue_output=self.normvalue,
                    connectible=self._norm_node.outputs[0],
                    parent=parameters,
                )
            )
            self._pars._norm_pars.append(
                NormalizedGaussianParameter(self.normvalue, ci, parent=parameters, labelfmt="{}")
            )

    @property
    def is_constrained(self) -> bool:
        return self._is_constrained

    @property
    def is_free(self) -> bool:
        return not self._is_constrained

    @property
    def is_correlated(self) -> bool:
        return self._covariance_node is not None

    @staticmethod
    def from_numbers(
        *,
        central: float | Sequence[float],
        sigma: float | Sequence[float],
        label: dict[str, str] | None = None,
        dtype: DTypeLike = "d",
        correlation: ndarray | Node | Sequence[Sequence[float | int]] | None = None,
        **kwargs,
    ) -> GaussianConstraint:
        label = {"text": "gaussian parameter"} if label is None else dict(label)
        name = label.setdefault("name", "parameter")

        if isinstance(central, (float, int)):
            central = (central,)
        if isinstance(sigma, (float, int)):
            sigma = (sigma,)

        node_central = Array(
            f"{name}_central",
            array(central, dtype=dtype),
            label=inherit_labels(
                label, fmtlong="central: {}", fmtshort="c({})", fields_exclude={"paths"}
            ),
        )

        node_sigma = Array(
            f"{name}_sigma",
            array(sigma, dtype=dtype),
            label=inherit_labels(
                label, fmtlong="sigma: {}", fmtshort="σ({})", fields_exclude={"paths"}
            ),
        )

        match correlation:
            case ndarray() | list() | tuple():
                node_cor = Array(
                    f"{name}_correlation",
                    array(correlation, dtype=dtype),
                    label=inherit_labels(
                        label,
                        fmtlong="correlations: {}",
                        fmtshort="C({})",
                        fields_exclude={"paths"},
                    ),
                )
            case Node():
                node_cor = correlation
                node_cor.labels.inherit(
                    label, fmtlong="correlations: {}", fmtshort="C({})", fields_exclude={"paths"}
                )
            case None:
                node_cor = None
            case _:
                raise RuntimeError(f"Unknown type for correlation: {type(correlation).__name__}")

        return GaussianConstraint(
            central=node_central, sigma=node_sigma, correlation=node_cor, **kwargs
        )


def GaussianParameters(
    names: Sequence[Sequence[str] | str], value: Node, *args, **kwargs
) -> Parameters:
    pars = Parameters(names, value, close=False)
    pars.set_constraint(GaussianConstraint(*args, parameters=pars, **kwargs))
    pars._close()

    return pars
