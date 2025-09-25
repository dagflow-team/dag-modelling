from __future__ import annotations

from collections import defaultdict
from collections.abc import Generator
from contextlib import suppress
from typing import TYPE_CHECKING, Sequence

from nested_mapping import NestedMapping
from nested_mapping.typing import KeyLike, TupleKey, properkey

from ...core.exception import ConnectionError
from ...core.meta_node import MetaNode
from ...core.node import Node, Output
from ...core.storage import NodeStorage
from ...parameters import GaussianParameter, NormalizedGaussianParameter
from ..arithmetic import Sum
from ..calculus import Jacobian
from ..linalg import MatrixProductDDt, MatrixProductDVDt

if TYPE_CHECKING:
    from typing import Mapping

CovarianceMatrixParameterType = (
    NormalizedGaussianParameter
    | GaussianParameter
    | Sequence[NormalizedGaussianParameter]
    | Sequence[GaussianParameter]
    | Sequence[Sequence[NormalizedGaussianParameter]]
    | Sequence[Sequence[GaussianParameter]]
    | Sequence[NestedMapping]
    | NestedMapping
)


class CovarianceMatrixGroup(MetaNode):
    __slots__ = (
        "_dict_jacobian",
        "_dict_cov_pars",
        "_dict_cov_syst_part",
        "_dict_cov_syst",
        "_cov_sum_syst",
        "_parameters",
        "_ignore_duplicated_paramters",
        "_store_to",
        "_jacobian_kwargs",
    )

    _dict_jacobian: dict[str, list[Jacobian]]
    _dict_cov_pars: dict[str, list[Node | Output]]
    _dict_cov_syst_part: dict[str, list[Node]]
    _dict_cov_syst: dict[str, Node]

    _cov_sum_syst: Node | None

    _parameters: set[GaussianParameter | NormalizedGaussianParameter]
    _ignore_duplicated_paramters: bool

    _store_to: TupleKey | None

    _jacobian_kwargs: Mapping

    def __init__(
        self,
        *,
        # labels: Mapping = {},
        store_to: KeyLike | None = None,
        ignore_duplicated_parameters: bool = False,
        jacobian_kwargs: Mapping = {},
    ):
        super().__init__()

        self._dict_jacobian = defaultdict(list)
        self._dict_cov_syst_part = defaultdict(list)
        self._dict_cov_syst = {}

        self._cov_sum_syst = None

        self._parameters = set()
        self._ignore_duplicated_paramters = ignore_duplicated_parameters

        self._store_to = properkey(store_to, sep=".") if store_to is not None else None

        self._jacobian_kwargs = jacobian_kwargs

    def get_parameters_count(self) -> int:
        return len(self._parameters)

    def add_covariance_for(
        self,
        name: str,
        parameter_groups: CovarianceMatrixParameterType,
        *,
        parameter_covariance_matrices: Sequence | None = None,
        # labels: Mapping = {},
    ) -> Node:
        if name in self._dict_jacobian:
            raise RuntimeError(f"Covariance group {name} already defined")

        storage = NodeStorage(default_containers=True) if self._store_to is not None else None

        jacobians = self._dict_jacobian[name]
        matrices = self._dict_cov_syst_part[name]
        parameter_groups_clean = self._get_parameter_groups(parameter_groups)
        npars_total = 0
        ngroups = len(parameter_groups_clean)
        for i, pars in enumerate(parameter_groups_clean):
            self._check_pars_unique(pars)
            npars = len(pars)
            npars_total += npars

            jacobian = Jacobian(
                f"Jacobian ({npars} pars): {name}",
                parameters=pars,
                **self._jacobian_kwargs,
            )
            jacobian()
            self._add_node(jacobian, kw_inputs={"input": "model"}, merge_inputs=("model",))
            self.inputs.make_positional("model", index=0)
            jacobians.append(jacobian)

            pars_covmat = None
            if parameter_covariance_matrices is not None:
                with suppress(IndexError):
                    pars_covmat = parameter_covariance_matrices[i]

            if pars_covmat:
                vsyst_part = MatrixProductDVDt.from_args(
                    f"V syst ({npars}): {name} ({i})", left=jacobian, square=pars_covmat
                )
            else:
                vsyst_part = MatrixProductDDt.from_args(
                    f"V syst ({npars}): {name} ({i})", matrix=jacobian
                )
            matrices.append(vsyst_part)

            if storage is not None:
                if ngroups > 1:
                    storage[
                        ("nodes",) + self._store_to + ("jacobians", name, f"jacobian_{i:02d}")
                    ] = jacobian
                    storage[
                        ("nodes",) + self._store_to + ("covmat_vsyst_parts", name, f"vsyst_{i:02d}")
                    ] = vsyst_part

                    storage[
                        ("outputs",) + self._store_to + ("jacobians", name, f"jacobian_{i:02d}")
                    ] = jacobian.outputs[0]
                    storage[
                        ("outputs",)
                        + self._store_to
                        + ("covmat_vsyst_parts", name, f"vsyst_{i:02d}")
                    ] = vsyst_part.outputs[0]
                else:
                    storage[("nodes",) + self._store_to + ("jacobians", name)] = jacobian
                    storage[("outputs",) + self._store_to + ("jacobians", name)] = jacobian.outputs[
                        0
                    ]

        if len(matrices) > 1:
            vsyst = Sum.from_args(f"V syst ({npars_total}): {name}", *matrices)
            for matrix in matrices:
                self._add_node(matrix)
        else:
            vsyst = matrices[0]

        self._add_node(vsyst)

        self._dict_cov_syst[name] = vsyst
        if storage is not None:
            storage[("nodes",) + self._store_to + ("covmat_syst", name)] = vsyst
            storage[("outputs",) + self._store_to + ("covmat_syst", name)] = vsyst.outputs[0]

            NodeStorage.update_current(storage, strict=True)

        return vsyst

    def add_covariance_sum(
        self,
        name: str = "sum",
        # *,
        # labels: Mapping = {},
    ) -> Node:
        if self._cov_sum_syst is not None:
            raise RuntimeError("Sum of covariance matrices already computed")

        npars = len(self._parameters)

        vsyst_part = list(self._dict_cov_syst.values())
        if len(vsyst_part) > 1:
            self._cov_sum_syst = Sum.from_args(f"V syst sum ({npars}): {name}", *vsyst_part)
            self._add_node(self._cov_sum_syst)
        else:
            self._cov_sum_syst = vsyst_part[0]

        if self._store_to:
            storage = NodeStorage(default_containers=True)

            storage[("nodes",) + self._store_to + ("covmat_syst", name)] = self._cov_sum_syst
            storage[("outputs",) + self._store_to + ("covmat_syst", name)] = (
                self._cov_sum_syst.outputs[0]
            )

            NodeStorage.update_current(storage, strict=True)

        return self._cov_sum_syst

    def compute_jacobians(self):
        for jacobians in self._dict_jacobian.values():
            for jacobian in jacobians:
                jacobian.compute()

    def update_matrices(self):
        self.compute_jacobians()

        for cov_syst in self._dict_cov_syst.values():
            cov_syst.touch()

        if self._cov_sum_syst:
            self._cov_sum_syst.touch()

    def _get_parameter_groups(
        self,
        parameter_groups: CovarianceMatrixParameterType,
    ) -> Sequence[Sequence[NormalizedGaussianParameter]] | Sequence[Sequence[GaussianParameter]]:
        match parameter_groups:
            case NormalizedGaussianParameter() | GaussianParameter():
                return ((parameter_groups,),)  # pyright: ignore [reportReturnType]
            case [NormalizedGaussianParameter() | GaussianParameter(), *_]:
                return (parameter_groups,)  # pyright: ignore [reportReturnType]
            case NestedMapping():
                return (tuple(parameter_groups.walkvalues()),)  # pyright: ignore [reportReturnType]
            case [[NormalizedGaussianParameter() | GaussianParameter(), *_], *_]:
                return parameter_groups
            case [NestedMapping(), *_]:
                return tuple(tuple(group.walkvalues()) for group in parameter_groups)  # pyright: ignore [reportReturnType]

        raise RuntimeError("Invalid parameter_groups type")

    def _check_pars_unique(
        self, pars: Sequence[NormalizedGaussianParameter] | Sequence[GaussianParameter]
    ):
        for par in pars:
            if not self._ignore_duplicated_paramters and par in self._parameters:
                break

            self._parameters.add(par)
        else:
            return

        raise RuntimeError(f"CovarianceMatrixGroup: duplicated parameter {par!s}")

    def __rrshift__(self, other: Output | Node | Sequence | Generator):
        """
        `other >> self`

        The node has a complicate structure (namely, dictionaries of nodes),
        so default connection and input strategies in the MetaNode do not work.

        The method connects `Output` or `Node` with all the nodes from `_dict_jacobian`.
        If `other` is a `Sequence` or `Generator`, the method connects zipped objects
        from `other` and `_dict_jacobian`.

        Also the method is used for `Output >> self` in the `Output.__rshift__`.
        """
        if isinstance(other, (Output, Node)):
            other >> tuple(self._dict_jacobian.values())  # pyright:ignore
        elif isinstance(other, (Sequence, Generator)):
            for lhs, rhs in zip(other, tuple(self._dict_jacobian.values())):
                lhs >> rhs
        else:
            raise ConnectionError(f"Cannot connect {other=} to node", node=self)
