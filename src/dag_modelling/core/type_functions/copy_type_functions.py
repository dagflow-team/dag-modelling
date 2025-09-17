from __future__ import annotations

from itertools import repeat
from typing import TYPE_CHECKING

from .tools_for_type_functions import (
    AllPositionals,
    LimbKey,
    MethodSequenceCaller,
    copy_dtype,
    copy_edges,
    copy_meshes,
    copy_shape,
    zip_dag,
)

if TYPE_CHECKING:
    from ..node import Node


def copy_from_inputs_to_outputs(
    node: Node,
    inputkey: LimbKey = 0,
    outputkey: LimbKey = AllPositionals,
    *,
    dtype: bool = True,
    shape: bool = True,
    edges: bool = True,
    meshes: bool = True,
    prefer_largest_input: bool = False,
    prefer_input_with_edges: bool = False,
) -> None:
    """Coping input dtype and setting for the output"""
    if not any((dtype, shape, edges, meshes)):
        return

    inputs = tuple(node.inputs.iter(inputkey))
    outputs = tuple(node.outputs.iter(outputkey))

    caller = MethodSequenceCaller()
    if dtype:
        caller.add(copy_dtype)
    if shape:
        caller.add(copy_shape)
    if edges:
        caller.add(copy_edges)
    if meshes:
        caller.add(copy_meshes)

    has_preference = prefer_input_with_edges or prefer_largest_input
    if has_preference and len(inputs) > 1:
        largest_input = inputs[0]
        largest_size = largest_input.dd.size
        found_edges = bool(largest_input.dd.axes_edges)
        for input in inputs[1:]:
            if (newsize := input.dd.size) <= largest_size and prefer_largest_input:
                continue
            if prefer_input_with_edges and found_edges and not bool(input.dd.axes_edges):
                continue
            largest_size = newsize
            largest_input = input
            found_edges = bool(input.dd.axes_edges)
        inputs = (largest_input,)

    if len(inputs) == 1:
        inputs = repeat(inputs[0], len(outputs))

    for input, output in zip_dag(inputs, outputs, strict=True):
        caller(input, output)


def copy_dtype_from_inputs_to_outputs(
    node: Node,
    inputkey: LimbKey = 0,
    outputkey: LimbKey = AllPositionals,
) -> None:
    """Coping input dtype and setting for the output"""
    inputs = tuple(node.inputs.iter(inputkey))
    outputs = tuple(node.outputs.iter(outputkey))

    if len(inputs) == 1:
        inputs = repeat(inputs[0], len(outputs))

    for input, output in zip_dag(inputs, outputs, strict=True):
        output.dd.dtype = input.dd.dtype


def copy_shape_from_inputs_to_outputs(
    node: Node,
    inputkey: str | int = 0,
    outputkey: LimbKey = AllPositionals,
) -> None:
    """Coping input shape and setting to each output"""
    inputs = tuple(node.inputs.iter(inputkey))
    outputs = tuple(node.outputs.iter(outputkey))

    if len(inputs) == 1:
        inputs = repeat(inputs[0], len(outputs))

    for input, output in zip_dag(inputs, outputs, strict=True):
        output.dd.shape = input.dd.shape
