from __future__ import annotations

from typing import TYPE_CHECKING

from nested_mapping.typing import properkey

from ...core.meta_node import MetaNode
from ...core.node import Node
from ...core.storage import NodeStorage
from .interpolator_core import InterpolatorCore
from .segment_index import SegmentIndex

if TYPE_CHECKING:
    from collections.abc import Mapping

    from nested_mapping.typing import KeyLike

    from .interpolator_core import MethodType, OutOfBoundsStrategyType


class Interpolator(MetaNode):
    __slots__ = (
        "_indexer",
        "_interpolators",
    )

    _indexer: Node
    _interpolators: list[Node]

    def __init__(self, *, bare: bool = False, labels: Mapping = {}, **kwargs):
        super().__init__()
        self._interpolators = []
        if bare:
            return

        self._add_indexer("indexer", label=labels.get("indexer", {}))
        self._add_interpolator("interpolator", label=labels.get("interpolator", {}), **kwargs)

    def _add_indexer(self, name: str, *, label={}):
        self._indexer = SegmentIndex(name, label=label)
        self._add_node(
            self._indexer,
            kw_inputs=["coarse", "fine"],
            merge_inputs=["coarse", "fine"],
        )

    def _add_interpolator(
        self,
        name: str,
        method: MethodType = "linear",
        *,
        positionals: bool = True,
        underflow: OutOfBoundsStrategyType = "extrapolate",
        overflow: OutOfBoundsStrategyType = "extrapolate",
        fillvalue: float = 0.0,
        label={},
    ) -> InterpolatorCore:
        self._interpolators.append(
            interpolator := InterpolatorCore(
                name,
                method=method,
                underflow=underflow,
                overflow=overflow,
                fillvalue=fillvalue,
                label=label,
            )
        )
        self._indexer.outputs["indices"] >> interpolator("indices")

        self._add_node(
            interpolator,
            kw_inputs=["coarse", "fine"],
            merge_inputs=["coarse", "fine"],
            inputs_pos=positionals,
            outputs_pos=positionals,
            missing_inputs=True,
            also_missing_outputs=True,
        )
        return interpolator

    @classmethod
    def replicate(
        cls,
        method: MethodType = "linear",
        names: Mapping[str, str] = {
            "indexer": "indexer",
            "interpolator": "interpolator",
        },
        labels: Mapping = {},
        *,
        replicate_xcoarse: bool = False,
        # replicate_ycoarse: bool = True,
        replicate_outputs: tuple[KeyLike, ...] = ((),),
        verbose: bool = False,
        **kwargs,
    ) -> tuple["Interpolator", "NodeStorage"]:
        storage = NodeStorage(default_containers=True)
        nodes = storage("nodes")
        inputs = storage("inputs")
        outputs = storage("outputs")

        interpolators = None
        key_interpolator = None

        def newgroup(key=()):
            nonlocal interpolators, key_interpolator

            interpolators = cls(bare=True)
            interpolators._add_indexer(names.get("indexer", "indexer"), label=labels.get("indexer", {}))
            key_interpolator = (names.get("interpolator", "interpolator"),)

            key_meta = (f"{key_interpolator[0]}_meta",) + key
            nodes[key_meta] = interpolators

            key_indexer = (names.get("indexer", "indexer"),) + key
            nodes[key_indexer] = interpolators._indexer

        label_int = labels.get("interpolator", {})
        for key in replicate_outputs:
            key = properkey(key)
            if replicate_xcoarse:
                newgroup(key)
            elif interpolators is None:
                newgroup()

            name = ".".join(key_interpolator + key)
            interpolator = interpolators._add_interpolator(
                name, method, label=label_int, positionals=False, **kwargs
            )
            nodes[name] = interpolator
            inputs.create_child(key_interpolator)[("ycoarse",) + key] = interpolator.inputs["y"]
            outputs[name] = interpolator.outputs[-1]

            if replicate_xcoarse:
                inputs.create_child(key_interpolator)[("xcoarse",) + key] = interpolators.inputs["coarse"]
                inputs.create_child(key_interpolator)[("xfine",) + key] = interpolators.inputs["fine"]

        if not replicate_xcoarse:
            inputs.create_child(key_interpolator)["xcoarse"] = interpolators.inputs["coarse"]
            inputs.create_child(key_interpolator)["xfine"] = interpolators.inputs["fine"]

        NodeStorage.update_current(storage, strict=True, verbose=verbose)

        return interpolators, storage
