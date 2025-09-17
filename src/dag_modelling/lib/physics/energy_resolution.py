from __future__ import annotations

from typing import TYPE_CHECKING

from ...core.exception import ConnectionError
from ...core.meta_node import MetaNode
from ...core.storage import NodeStorage
from ..axis.bin_center import BinCenter
from ..hist.hist_smear_normal_matrix_b_c import HistSmearNormalMatrixBC
from ..physics.energy_resolution_sigma_rel_a_b_c import EnergyResolutionSigmaRelABC

if TYPE_CHECKING:
    from collections.abc import Mapping

    from nested_mapping.typing import KeyLike

    from ...core.node import Node


class EnergyResolution(MetaNode):
    __slots__ = (
        "_energy_resolution_matrix_bc_list",
        "_energy_resolution_sigma_rel_abc_list",
        "_bin_center_list",
    )

    _energy_resolution_matrix_bc_list: list[Node]
    _energy_resolution_sigma_rel_abc_list: list[Node]
    _bin_center_list: list[Node]

    def __init__(self, *, bare: bool = False, labels: Mapping = {}):
        super().__init__()
        self._energy_resolution_matrix_bc_list = []
        self._energy_resolution_sigma_rel_abc_list = []
        self._bin_center_list = []
        if bare:
            return

        self.add_energy_resolution_sigma_rel_abc(
            name="EnergyResolutionSigmaRelABC",
            label=labels.get("EnergyResolutionSigmaRelABC", {}),
        )
        self.add_bin_center("BinCenter", labels.get("BinCenter", {}))
        self.add_energy_resolution_matrix_bc(
            "EnergyResolution", labels.get("EnergyResolution", {})
        )
        self._bind_outputs()

    def add_energy_resolution_sigma_rel_abc(
        self,
        name: str = "EnergyResolutionSigmaRelABC",
        label: Mapping = {},
    ) -> EnergyResolutionSigmaRelABC:
        _energy_resolution_sigma_rel_abc = EnergyResolutionSigmaRelABC(
            name=name, label=label
        )
        self._energy_resolution_sigma_rel_abc_list.append(
            _energy_resolution_sigma_rel_abc
        )
        self._add_node(
            _energy_resolution_sigma_rel_abc,
            kw_inputs=["Energy", "a_nonuniform", "b_stat", "c_noise"],
            kw_outputs=["RelSigma"],
            merge_inputs=["Energy"],
        )
        return _energy_resolution_sigma_rel_abc

    def add_bin_center(self, name: str = "BinCenter", label: Mapping = {}) -> BinCenter:
        _bin_center = BinCenter(name, label=label)
        _bin_center._add_pair("Edges", "Energy")
        self._bin_center_list.append(_bin_center)
        self._add_node(
            _bin_center,
            kw_inputs=["Edges"],
            kw_outputs=["Energy"],
            merge_inputs=["Edges"],
            missing_inputs=True,
            also_missing_outputs=True,
        )
        return _bin_center

    def add_energy_resolution_matrix_bc(
        self,
        name: str = "EnergyResolution",
        label: Mapping = {},
    ) -> HistSmearNormalMatrixBC:
        _energy_resolution_matrix_bc = HistSmearNormalMatrixBC(name, label=label)
        self._energy_resolution_matrix_bc_list.append(_energy_resolution_matrix_bc)
        self._add_node(
            _energy_resolution_matrix_bc,
            kw_inputs=["RelSigma", "Edges", "EdgesOut"],
            kw_outputs=["SmearMatrix"],
            merge_inputs=["Edges"],
            missing_inputs=True,
            also_missing_outputs=True,
        )
        return _energy_resolution_matrix_bc

    def _bind_outputs(self) -> None:
        if not (
            (l1 := len(self._bin_center_list))
            == (l2 := len(self._energy_resolution_matrix_bc_list))
            == (l3 := len(self._energy_resolution_sigma_rel_abc_list))
        ):
            raise ConnectionError(
                f"Cannot bind outputs! Nodes must be triplets of (BinCenter, HistSmearNormalMatrixBC, EnergyResolutionSigmaRelABC), but current lengths are {l1}, {l2}, {l3}!",
                node=self,
            )
        for (
            _bin_center,
            _energy_resolution_sigma_rel_abc,
            _energy_resolution_matrix_bc,
        ) in zip(
            self._bin_center_list,
            self._energy_resolution_sigma_rel_abc_list,
            self._energy_resolution_matrix_bc_list,
        ):
            (
                _bin_center.outputs["Energy"]
                >> _energy_resolution_sigma_rel_abc.inputs["Energy"]
            )
            (
                _energy_resolution_sigma_rel_abc._rel_sigma
                >> _energy_resolution_matrix_bc.inputs["RelSigma"]
            )

    # TODO: check this again; what should be in replicate_outputs argument: all the nodes or only main?
    @classmethod
    def replicate(
        cls,
        *,
        names: Mapping[str, str] = {
            "EnergyResolutionSigmaRelABC": "sigma_rel",
            "HistSmearNormalMatrixBC": "matrix",
            "Edges": "e_edges",
            "EdgesOut": "e_edges_out",
            "BinCenter": "e_bincenter",
        },
        path: str | None = None,
        labels: Mapping = {},
        replicate_outputs: tuple[KeyLike, ...] = ((),),
        verbose: bool = False,
    ) -> tuple[EnergyResolution, NodeStorage]:
        storage = NodeStorage(default_containers=True)
        nodes = storage("nodes")
        inputs = storage("inputs")
        outputs = storage("outputs")

        instance = cls(bare=True)
        key_energy_resolution_matrix_bc = (
            names.get("HistSmearNormalMatrixBC", "HistSmearNormalMatrixBC"),
        )
        key_energy_resolution_sigma_rel_abc = (
            names.get("EnergyResolutionSigmaRelABC", "EnergyResolutionSigmaRelABC"),
        )
        key_bin_center = (names.get("BinCenter", "BinCenter"),)
        key_edges0 = (names.get("Edges", "Edges"),)
        key_edges_out0 = (names.get("EdgesOut", "EdgesOut"),)

        tpath = tuple(path.split(".")) if path else ()
        key_energy_resolution_matrix_bc = tpath + key_energy_resolution_matrix_bc
        key_energy_resolution_sigma_rel_abc = (
            tpath + key_energy_resolution_sigma_rel_abc
        )
        key_bin_center = tpath + key_bin_center
        key_edges = tpath + key_edges0

        _energy_resolution_sigma_rel_abc = instance.add_energy_resolution_sigma_rel_abc(
            names.get("EnergyResolutionSigmaRelABC", "EnergyResolutionSigmaRelABC"),
            labels.get("EnergyResolutionSigmaRelABC", {}),
        )
        nodes[key_energy_resolution_sigma_rel_abc] = _energy_resolution_sigma_rel_abc
        for iname, input in _energy_resolution_sigma_rel_abc.inputs.iter_kw_items():
            inputs[key_energy_resolution_sigma_rel_abc + (iname,)] = input
        outputs[key_energy_resolution_sigma_rel_abc] = (
            _energy_resolution_sigma_rel_abc.outputs["RelSigma"]
        )

        _bin_center = instance.add_bin_center("BinCenter", labels.get("BinCenter", {}))
        nodes[key_bin_center] = _bin_center
        inputs[key_edges] = _bin_center.inputs[0]
        outputs[key_bin_center] = (out_bincenter := _bin_center.outputs[0])

        out_relsigma = _energy_resolution_sigma_rel_abc.outputs["RelSigma"]
        out_bincenter >> _energy_resolution_sigma_rel_abc.inputs["Energy"]

        label_int = labels.get("EnergyResolution", {})
        for key in replicate_outputs:
            if isinstance(key, str):
                key = (key,)
            name = ".".join(key_energy_resolution_matrix_bc + key)
            eres = instance.add_energy_resolution_matrix_bc(name, label_int)
            nodes[key_energy_resolution_matrix_bc + key] = eres
            inputs[key_energy_resolution_matrix_bc + key_edges0 + key] = eres.inputs[
                "Edges"
            ]
            inputs[key_energy_resolution_matrix_bc + key_edges_out0 + key] = (
                eres.inputs["EdgesOut"]
            )
            outputs[key_energy_resolution_matrix_bc + key] = eres.outputs[0]

            out_relsigma >> eres.inputs["RelSigma"]

        NodeStorage.update_current(storage, strict=True, verbose=verbose)
        return instance, storage
