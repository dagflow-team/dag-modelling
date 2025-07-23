from __future__ import annotations

from typing import TYPE_CHECKING

from nested_mapping.typing import KeyLike, properkey, strkey

from ...core.meta_node import MetaNode
from ...core.storage import NodeStorage
from .integrator_core import IntegratorCore
from .integrator_sampler import IntegratorSampler

if TYPE_CHECKING:
    from collections.abc import Mapping

    from nested_mapping.typing import Key

    from .integrator_sampler import ModeType


class Integrator(MetaNode):
    __slots__ = ("_sampler",)

    _sampler: IntegratorSampler

    def __init__(
        self,
        mode: ModeType,
        *,
        bare: bool = False,
        dropdim: bool = True,
        labels: Mapping = {},
    ):
        super().__init__()
        if bare:
            return

        self._init_sampler(mode, "sampler", labels.get("sampler", {}))
        node = self._add_integrator("integrator", labels.get("integrator", {}), dropdim=dropdim)
        self._leading_node = node

    def _init_sampler(self, mode: ModeType, name: str = "sampler", label: Mapping = {}):
        self._sampler = IntegratorSampler(name, mode=mode, label=label)

        self._add_node(
            self._sampler,
            kw_inputs=["orders_x"],
            kw_inputs_optional=["orders_y"],
            kw_outputs=["x"],
            kw_outputs_optional=["y"],
            merge_inputs=["orders_x", "orders_y"],
            missing_inputs=True,
            also_missing_outputs=True,
        )

    def _add_integrator(
        self,
        name: str = "integrator",
        label: Mapping = {},
        *,
        positionals: bool = True,
        dropdim: bool,
    ) -> IntegratorCore:
        integrator = IntegratorCore(name, dropdim=dropdim, ndim=self._sampler.ndim, label=label)
        self._sampler.outputs["weights"] >> integrator("weights")

        self._add_node(
            integrator,
            kw_inputs=["orders_x"],
            kw_inputs_optional=["orders_y"],
            merge_inputs=["orders_x", "orders_y"],
            inputs_pos=positionals,
            outputs_pos=positionals,
            missing_inputs=True,
            also_missing_outputs=True,
        )

        return integrator

    @classmethod
    def replicate(
        cls,
        mode: ModeType,
        *,
        names: Mapping[str, KeyLike] = {
            "sampler": "sampler",
            "integrator": "integral",
            "mesh_x": "sampler.mesh_x",
            "mesh_y": "sampler.mesh_y",
            "orders_x": "sampler.orders_x",
            "orders_y": "sampler.orders_y",
        },
        path: KeyLike = (),
        labels: Mapping = {},
        replicate_outputs: tuple[Key, ...] = ((),),
        verbose: bool = False,
        single_node: bool = False,
        dropdim: bool = True,
    ) -> tuple["Integrator", "NodeStorage"]:
        storage = NodeStorage(default_containers=True)
        nodes = storage("nodes")
        inputs = storage("inputs")
        outputs = storage("outputs")

        path = properkey(path)

        instance = cls(mode, bare=True)
        key_integrator = path + properkey(names.get("integrator", "integrator"))
        key_sampler = path + properkey(names.get("sampler", "sampler"))
        key_meta = key_integrator[:-1] + (f"{key_integrator[-1]}_meta",)
        key_mesh_x = path + properkey(names.get("mesh_x", "mesh_x"))
        key_mesh_y = path + properkey(names.get("mesh_y", "mesh_y"))
        key_orders_x = path + properkey(names.get("orders_x", "orders_x"))
        key_orders_y = path + properkey(names.get("orders_y", "orders_y"))
        nodes[key_meta] = instance

        instance._init_sampler(mode, strkey(key_sampler), labels.get("sampler", {}))
        outputs[key_mesh_x] = instance._sampler.outputs["x"]
        outputs[key_mesh_y] = instance._sampler.outputs["y"]
        nodes[key_sampler] = instance._sampler

        label_int = labels.get("integrator", {})
        integrator = None
        need_new_instance = not single_node
        for key in replicate_outputs:
            key = properkey(key)
            name = ".".join(key_integrator + key)

            if need_new_instance:
                integrator = instance._add_integrator(
                    name, label_int, positionals=False, dropdim=dropdim
                )
                nodes[key_integrator + key] = integrator
            elif integrator is None:
                integrator = instance._add_integrator(
                    name, label_int, positionals=False, dropdim=dropdim
                )
                nodes[key_integrator] = integrator

            # NOTE: it is needed to create an input and add to the storage
            integrator()
            inputs[key_integrator + key] = integrator.inputs[-1]
            outputs[key_integrator + key] = integrator.outputs[-1]

        inputs[key_orders_x] = instance.inputs["orders_x"]
        inputs[key_orders_y] = instance.inputs["orders_y"]

        NodeStorage.update_current(storage, strict=True, verbose=verbose)

        return instance, storage
