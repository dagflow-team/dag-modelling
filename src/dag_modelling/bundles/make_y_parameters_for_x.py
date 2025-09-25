from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import ndarray

from nested_mapping import NestedMapping

from ..core.labels import format_dict
from ..core.output import Output
from .load_parameters import label_keys, load_parameters

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray


def make_y_parameters_for_x(
    array: NDArray | Output,
    *,
    namefmt: str,
    key: str | tuple[str, ...],
    format: str | tuple[str, ...],
    values: float | tuple[float, ...],
    state: str,
    labels: Mapping | str = {},
    disable_last_one: bool = False,
    hide_nodes = False,
    **kwargs
) -> NestedMapping:
    parcfg = NestedMapping({})
    parcfg[key] = values

    data = _get_array(array)
    if disable_last_one:
        data = data[:-1]
    npars = len(data)
    index = tuple(namefmt.format(i) for i in range(npars))

    labels_storage = NestedMapping({})
    labels_loc = labels_storage.create_child(key)
    for i, (key, value) in enumerate(zip(index, data)):
        labels_loc[key] = (label_loc:=format_dict(
            labels,
            i = i,
            value = value,
            process_keys = label_keys,
        ))

        if hide_nodes:
            label_loc["node_hidden"] = i>1 and i<(npars-2)

    return load_parameters(
        format=format,
        state=state,
        parameters=parcfg.object,
        labels=labels_storage.object,
        replicate=index,
        **kwargs
    )


# TODO: move function to common place
def _get_array(object: NDArray | Output) -> NDArray:
    match object:
        case ndarray():
            return object
        case Output():
            return object.data

    raise ValueError("Invalid array type")
