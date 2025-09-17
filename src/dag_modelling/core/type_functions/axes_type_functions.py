from __future__ import annotations

from collections.abc import Sequence
from itertools import repeat
from typing import TYPE_CHECKING

from ..exception import TypeFunctionError
from ..input import Input
from ..output import Output
from .tools_for_type_functions import AllPositionals, LimbKey, zip_dag

if TYPE_CHECKING:
    from ..node import Node

# TODO: we have no type functions for meshes!


####################
# edges
####################
def copy_edges_from_inputs_to_outputs(
    node: Node,
    inputkey: str | int = 0,
    outputkey: LimbKey = AllPositionals,
) -> None:
    """Coping input edges and setting for the output"""
    inputs = tuple(node.inputs.iter(inputkey))
    outputs = tuple(node.outputs.iter(outputkey))

    if len(inputs) == 1:
        inputs = repeat(inputs[0], len(outputs))

    for input, output in zip_dag(inputs, outputs, strict=True):
        output.dd.axes_edges = input.dd.axes_edges


def check_edges_dimension_of_inputs(
    node: Node,
    inputkey: LimbKey = AllPositionals,
    dim: int = 1,
):
    """Checking the existence and dim of the edges of the inputs"""
    for input in node.inputs.iter(inputkey):
        edges = input.dd.axes_edges
        if len(edges) == 0:
            raise TypeFunctionError(
                f"The input must have edges, but given {edges=}!",
                node=node,
                input=input,
            )
        for edge in edges:
            if not isinstance(edge, Output):
                raise TypeFunctionError(
                    f"The input edge must be an `Output`, but given {edge=}!",
                    node=node,
                    input=input,
                )
            if edge.dd.dim != dim:
                raise TypeFunctionError(
                    f"The input edge must be a {dim}d array, but given {edge.dd.dim=}!",
                    node=node,
                    input=input,
                )


def check_edges_equivalence_of_inputs(
    node: Node,
    inputkey: LimbKey = AllPositionals,
    reference: tuple[Output, ...] | None = None,
):
    """Checking the equivalence of the edges of the inputs."""
    inputs = tuple(node.inputs.iter(inputkey))
    if reference is None:
        input0, inputs = inputs[0], inputs[1:]
        reference = input0.dd.axes_edges
    for input in inputs:
        edges = input.dd.axes_edges
        if edges != reference:
            raise TypeFunctionError(
                f"The input edge must be {reference}, but given {edges=}!",
                node=node,
                input=input,
            )


def check_dtype_of_edges(
    node: Node,
    inputkey: LimbKey = AllPositionals,
    outputkey: LimbKey = AllPositionals,
):
    """Checking of the edges type (must be `List[Output]`) of the inputs and outputs."""
    # check inputs
    for input in node.inputs.iter(inputkey):
        edges = input.dd.axes_edges
        if not isinstance(edges, tuple):
            raise TypeFunctionError(
                f"The `input.dd.axes_edges` must be `List[Output]`, but given {edges=}!",
                node=node,
                input=input,
            )
        for edge in edges:
            if not isinstance(edge, Output):
                raise TypeFunctionError(
                    f"The edge must be `Output`, but given {edge=}!",
                    node=node,
                    input=input,
                )
    # check outputs
    for output in node.outputs.iter(outputkey):
        edges = output.dd.axes_edges
        if not isinstance(edges, tuple):
            raise TypeFunctionError(
                f"The `output.dd.axes_edges` must be `List[Output]`, but given {edges=}!",
                node=node,
                output=output,
            )
        for edge in edges:
            if not isinstance(edge, Output):
                raise TypeFunctionError(
                    f"The edge must be `Output`, but given {edge=}!",
                    node=node,
                    output=output,
                )


def check_edges_consistency_with_array(node: Node, outputkey: LimbKey = AllPositionals):
    """
    Checks the dimension equivalence of edges and the output, then checks that
    `len(output) = N` and `len(edges) = N+1` for each dimension.
    Tht type function is passed if the edges are empty.
    """
    for output in node.outputs.iter(outputkey):
        dd = output.dd
        edges = dd.axes_edges
        if (y := len(edges)) > 0:
            if y != dd.dim:
                raise TypeFunctionError(
                    f"Array: the data ({dd.dim}d) and edges "
                    f"({len(edges)}d) must have the same dimension!",
                    node=node,
                    output=output,
                )
            for i, edge in enumerate(edges):
                if edge.dd.shape[0] != dd.shape[i] + 1:
                    raise TypeFunctionError(
                        f"Array: the data length+1 ({dd.shape[i]+1}) "
                        f"is not consistent with edges ({edge.dd.shape[0]})",
                        node=node,
                        output=output,
                    )


####################
# assign axes from inputs to outputs
####################
def assign_edges_from_inputs_to_outputs(
    input: Input | Sequence[Input], output: Output, *, ignore_assigned: bool = False
):
    """Assign output's edges from input's parent output"""
    dd = output.dd

    if dd.axes_edges:
        if ignore_assigned:
            return
        raise TypeFunctionError("Edges already assigned", output=output, input=input)

    if isinstance(input, Input):
        edges = [input.parent_output]
    else:
        edges = [inp.parent_output for inp in input]

    if len(dd.shape) != len(edges):
        raise TypeFunctionError(
            f"Output ndim={len(dd.shape)} is inconsistent with edges ndim={len(edges)}",
            output=output,
            input=input,
        )

    for i, (dimsize, edgeoutput) in enumerate(zip_dag(dd.shape, edges)):
        shapeedges = edgeoutput.dd.shape
        if len(shapeedges) != 1:
            raise TypeFunctionError(
                f"Edges of {i}th dimension has non-1d shape", output=output, input=input
            )
        dimedges = shapeedges[0]
        if dimsize != (dimedges - 1):
            raise TypeFunctionError(
                f"Output dimension {i} size={dimsize} is inconsistent with edges"
                f" size={len(dimedges)}",
                output=output,
                input=input,
            )

    output.dd.axes_edges = tuple(edges)


def assign_meshes_from_inputs_to_outputs(
    input: Input | Sequence[Input],
    output: Output,
    *,
    ignore_assigned: bool = False,
    overwrite_assigned: bool = False,
    ignore_inconsistent_number_of_meshes: bool = False,
):
    """Assign output's edges from input's parent output"""
    dd = output.dd

    if dd.axes_meshes:
        if ignore_assigned:
            return
        elif not overwrite_assigned:
            raise TypeFunctionError("Meshes already assigned", output=output, input=input)

    if isinstance(input, Input):
        meshes = [input.parent_output]
    else:
        meshes = [inp.parent_output for inp in input]

    naxes = len(dd.shape)
    if naxes != len(meshes) and (not overwrite_assigned or naxes != len(dd.axes_meshes)):
        if ignore_inconsistent_number_of_meshes:
            return
        raise TypeFunctionError(
            f"Output ndim={len(dd.shape)} is inconsistent with meshes ndim={len(meshes)}",
            output=output,
            input=input,
        )

    for i, meshesoutput in enumerate(meshes):
        if dd.shape != meshesoutput.dd.shape:
            raise TypeFunctionError(
                f"Output shape={dd.shape} is inconsistent with meshes"
                f" {i} shape={meshesoutput.dd.shape}",
                output=output,
                input=input,
            )

    if overwrite_assigned:
        newmeshes = list(output.dd.axes_meshes)
        for i, mesh in enumerate(meshes):
            try:
                newmeshes[i] = mesh
            except IndexError:
                newmeshes.append(mesh)
        dd.axes_meshes = tuple(newmeshes)
    else:
        dd.axes_meshes = tuple(meshes)


def assign_axes_from_inputs_to_outputs(
    node: Node,
    inputkey: LimbKey = 0,
    outputkey: LimbKey = AllPositionals,
    *,
    assign_edges: bool = False,
    assign_meshes: bool = False,
    ignore_Nd: bool = False,
    merge_input_axes: bool = False,
    **kwargs,
) -> None:
    """Set outputs' edges/meshes based on inputs (take parent_output). Process each pair."""
    if not (assign_edges ^ assign_meshes):
        raise TypeFunctionError(
            f"assign_axes_from_inputs_to_outputs: may not {assign_edges=} and {assign_meshes=}"
        )

    inputs = tuple(node.inputs.iter(inputkey))
    outputs = tuple(node.outputs.iter(outputkey))

    if merge_input_axes:
        for output in outputs:
            if assign_edges:
                assign_edges_from_inputs_to_outputs(inputs, output, **kwargs)

            if assign_meshes:
                assign_meshes_from_inputs_to_outputs(inputs, output, **kwargs)
        return

    if len(inputs) == 1:
        inputs = repeat(inputs[0], len(outputs))

    for input, output in zip_dag(inputs, outputs):
        if ignore_Nd and len(output.dd.shape) != 1:
            continue
        if assign_edges:
            assign_edges_from_inputs_to_outputs(input, output, **kwargs)

        if assign_meshes:
            assign_meshes_from_inputs_to_outputs(input, output, **kwargs)
