from __future__ import annotations

from collections.abc import Sequence
from itertools import repeat
from typing import TYPE_CHECKING

from numpy import allclose, issubdtype

from ..exception import TypeFunctionError
from ..input import Input
from .tools_for_type_functions import AllPositionals, LimbKey, shapes_are_broadcastable, zip_dag

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from ..node import Node


def check_node_has_inputs(
    node: Node,
    inputkey: str | int | slice | Sequence | None = None,
    *,
    check_named: bool = False,
) -> None:
    """Checking if the node has inputs"""
    if inputkey is None or inputkey == AllPositionals:
        try:
            node.inputs[0]
        except Exception as exc:
            if not check_named:
                raise TypeFunctionError(
                    "The node must have at least one positional input!", node=node
                ) from exc
            if node.inputs.len_kw() == 0:
                raise TypeFunctionError(
                    "The node must have at least one input!", node=node
                ) from exc
    else:
        try:
            node.inputs[inputkey]
        except Exception as exc:
            raise TypeFunctionError(
                f"The node must have the input '{inputkey}'!", node=node
            ) from exc


def check_number_of_inputs(node: Node, n: int) -> None:
    """Checking if the node has only `n` inputs"""
    if (ninp := len(node.inputs)) != n:
        raise TypeFunctionError(f"The node must have only {n} inputs, but given {ninp}!", node=node)


def check_dimension_of_inputs(node: Node, inputkey: LimbKey, ndim: int, **kwargs):
    """Checking the dimension of the input"""
    for input in node.inputs.iter(inputkey, **kwargs):
        dim = len(input.dd.shape)
        if ndim != dim:
            raise TypeFunctionError(
                f"The node supports only {ndim}d inputs. Got {dim}d!",
                node=node,
                input=input,
            )


def check_shape_of_inputs(node: Node, inputkey: LimbKey, *shapes: tuple[int, ...], **kwargs):
    """Checking the shape equivalence for inputs"""
    for input in node.inputs.iter(inputkey, **kwargs):
        shape_current = input.dd.shape
        if all(shape_current != shape for shape in shapes):
            raise TypeFunctionError(
                f"The node supports only inputs with shape=({shapes}). Got {shape_current}!",
                node=node,
                input=input,
            )


def check_size_of_inputs(
    node: Node,
    inputkey: LimbKey,
    *,
    exact: int | None = None,
    min: int | None = None,
    max: int | None = None,
    **kwargs,
):
    """Checking the shape equivalence for inputs"""
    if exact is not None:
        for input in node.inputs.iter(inputkey, **kwargs):
            size = input.dd.size
            if size != exact:
                raise TypeFunctionError(
                    f"The input size {size} is not equal to {exact}",
                    node=node,
                    input=input,
                )
    else:
        if min is None and max is None:
            raise TypeFunctionError(
                "`exact`, `min` and/or `max` must be passed into the type function!",
                node=node,
            )
        for input in node.inputs.iter(inputkey, **kwargs):
            size = input.dd.size
            if min is not None and size < min:
                raise TypeFunctionError(
                    f"The input size {size} is below {min}",
                    node=node,
                    input=input,
                )
            if max is not None and size > max:
                raise TypeFunctionError(
                    f"The input size {size} is above {max}",
                    node=node,
                    input=input,
                )


def check_dtype_of_inputs(node: Node, inputkey: LimbKey, *, dtype: DTypeLike, **kwargs):
    """Checking the dtype equivalence for inputs"""
    for input in node.inputs.iter(inputkey, **kwargs):
        dtt = input.dd.dtype
        if dtt != dtype:
            raise TypeFunctionError(
                f"The node supports only input types {dtype}. Got {dtt}!",
                node=node,
                input=input,
            )


def check_inputs_are_square_matrices(node: Node, inputkey: LimbKey):
    """Checking input is a square matrix"""
    for input in node.inputs.iter(inputkey):
        shape = input.dd.shape
        dim = len(shape)
        if dim != 2 or shape[0] != shape[1]:
            raise TypeFunctionError(
                f"The node supports only square inputs. Got {shape}!",
                node=node,
                input=input,
            )


def _check_input_block_or_diag(node: Node, input: Input, *, check_square: bool = False) -> int:
    """Check if input is a block/square matrix or a diagonal (1d) of a square matrix.
    Returns the maximal dimension."""
    shape = input.dd.shape
    dim = len(shape)
    if dim > 2:
        raise TypeFunctionError(
            f"The node supports only 1d or 2d. Got {dim}d!",
            node=node,
            input=input,
        )
    if dim == 2:
        if check_square and shape[0] != shape[1]:
            raise TypeFunctionError(
                f"The node supports only square inputs (or 1d as diagonal). Got {shape}!",
                node=node,
                input=input,
            )
    elif dim != 1:
        mtype = "square" if check_square else "matrix"
        raise TypeFunctionError(
            f"The node supports only {mtype} inputs (or 1d as diagonal). Got {shape}!",
            node=node,
            input=input,
        )

    return dim


def check_inputs_are_matrices_or_diagonals(node: Node, inputkey: LimbKey, *, check_square: bool = False) -> int:
    """Check if input is a square matrix or diagonal (1d) of a square matrix.
    Returns the maximal dimension."""
    dim_max = 0
    for input in node.inputs.iter(inputkey):
        dim = _check_input_block_or_diag(node, input, check_square=check_square)
        dim_max = max(dim, dim_max)
    return dim_max


def check_inputs_consistency_with_square_matrices_or_diagonals(node: Node, inputkey: LimbKey = AllPositionals) -> int:
    """Check if inputs are square matrices or diagonals (1d) of a square matrices of the same size.
    Returns the maximal dimension."""
    inputs = tuple(node.inputs.iter(inputkey))

    dim_max = 0
    shape0 = inputs[0].dd.shape[0]

    for input in inputs:
        dim = _check_input_block_or_diag(node, input, check_square=True)
        dim_max = max(dim, dim_max)

        shape = input.dd.shape[0]
        if shape != shape0:
            raise TypeFunctionError(
                f"All inputs should have the same size {shape0}, got {shape}",
                node=node,
                input=input,
            )

    return dim_max


def check_inputs_equivalence(
    node: Node,
    inputkey: LimbKey = AllPositionals,
    *,
    check_dtype: bool = True,
    check_shape: bool = True,
    check_edges: bool = True,
    check_edges_contents: bool = False,
    check_meshes: bool = False,
    check_meshes_contents: bool = False,
    broadcastable: bool = False,
    atol: float = 1.0e-14,
):
    """Checking the equivalence of the dtype, shape, axes_edges and axes_meshes of all the inputs"""
    inputs = tuple(node.inputs.iter(inputkey))
    input0, inputs = inputs[0], inputs[1:]

    dtype, shape, edges, meshes = (
        input0.dd.dtype,
        input0.dd.shape,
        input0.dd.axes_edges,
        input0.dd.axes_meshes,
    )
    for input in inputs:
        dd = input.dd
        dtype_inconsistent = check_dtype and dd.dtype != dtype
        if check_shape:
            shape_inconsistent = (
                not shapes_are_broadcastable(shape, dd.shape)
                if broadcastable
                else dd.shape != shape
            )
        else:
            shape_inconsistent = False

        edges_inconsistent = check_edges and dd.axes_edges and edges and dd.axes_edges != edges
        if edges_inconsistent and check_edges_contents:
            edges_inconsistent = not all(
                allclose(a.data, b.data, atol=atol) for a, b in zip_dag(edges, dd.axes_edges)
            )

        meshes_inconsistent = (
            check_meshes and dd.axes_meshes and meshes and dd.axes_meshes != meshes
        )
        if meshes_inconsistent and check_meshes_contents:
            meshes_inconsistent = not all(
                allclose(a.data, b.data, atol=atol) for a, b in zip_dag(meshes, dd.axes_meshes)
            )

        if any(
            (
                dtype_inconsistent,
                shape_inconsistent,
                edges_inconsistent,
                meshes_inconsistent,
            )
        ):
            raise TypeFunctionError(
                f"Input data [{dd.dtype=}, {dd.shape=}, {dd.axes_edges=}, {dd.axes_meshes=}]"
                f" is inconsistent with [{dtype=}, {shape=}, {edges=}, {meshes=}]",
                node=node,
                input=input,
            )


def check_inputs_have_same_dtype(node: Node, inputkey: LimbKey = AllPositionals) -> DTypeLike:
    """Checking dtypes of all the inputs are same"""
    inputs = tuple(node.inputs.iter(inputkey))
    input0, inputs = inputs[0], inputs[1:]

    dtype = input0.dd.dtype
    for input in inputs:
        if input.dd.dtype != dtype:
            raise TypeFunctionError(
                f"Input data {input.dd.dtype} is inconsistent with {dtype}",
                node=node,
                input=input,
            )
    return dtype


def check_inputs_have_same_shape(node: Node, inputkey: LimbKey = AllPositionals) -> tuple[int, ...]:
    """Checking shapes of all the inputs are same"""
    inputs = tuple(node.inputs.iter(inputkey))
    input0, inputs = inputs[0], inputs[1:]

    shape = input0.dd.shape
    for input in inputs:
        if input.dd.shape != shape:
            raise TypeFunctionError(
                f"Input data {input.dd.shape} is inconsistent with {shape}",
                node=node,
                input=input,
            )
    return shape


def check_subtype_of_inputs(node: Node, inputkey: LimbKey, *, dtype: DTypeLike):
    """Checks if the input dtype is some subtype of `dtype`."""
    for input in node.inputs.iter(inputkey):
        if not issubdtype(input.dd.dtype, dtype):
            raise TypeFunctionError(
                f"The input must be an array of {dtype}, but given '{input.dd.dtype}'!",
                node=node,
                input=input,
            )


def check_inputs_are_matrix_multipliable(
    node: Node, inputkey1: LimbKey, inputkey2: LimbKey
) -> tuple[tuple[int, int], ...]:
    """Checking that inputs from key1 and key2 may be multiplied (matrix)
    Return shapes of the multiplications.
    """
    inputs1 = tuple(node.inputs.iter(inputkey1))
    inputs2 = tuple(node.inputs.iter(inputkey2))

    len1, len2 = len(inputs1), len(inputs2)
    if len1 == len2:
        pass
    elif len1 == 1:
        inputs1 = repeat(inputs1[0], len2)
    elif len2 == 1:
        inputs2 = repeat(inputs2[0], len1)

    ret = []
    for input1, input2 in zip_dag(inputs1, inputs2, strict=True):
        shape1 = input1.dd.shape
        shape2 = input2.dd.shape
        if shape1[-1] != shape2[0]:
            raise TypeFunctionError(
                f"Inputs {shape1} and {shape2} are not multiplicable",
                node=node,
                input=input,
            )
        ret.append((shape1[0], shape2[-1]))

    return tuple(ret)


def find_max_size_of_inputs(node: Node, inputkey: LimbKey = AllPositionals) -> int:
    """Find maximum size of inputs"""
    inputs = tuple(node.inputs.iter(inputkey))
    input0, inputs = inputs[0], inputs[1:]

    size = input0.dd.size
    for input in inputs:
        if (y := input.dd.size) > size:
            size = y
    return size


def check_inputs_number_is_divisible_by_N(node: Node, N: int) -> None:
    """Check whether inputs count multiple to some N or not"""
    if N==1:
        return
    n = node.inputs.len_pos()
    if n % N != 0:
        raise TypeFunctionError(f"Node takes only {N}N inputs, but given {n}", node=node)


# NOTE: may be it will be needed later, but for now is not
# def check_if_input_sorted(
#    node: Node,
#    inputkey: LimbKey = AllPositionals,
# ):
#    """Checking if the inputs are sorted arrays"""
#    is_sorted = lambda a: all(a[:-1] <= a[1:])
#    for input in node.inputs.iter(inputkey):
#        if not is_sorted(input.data):
#            raise TypeFunctionError(
#                "The `input` must be a sorted array!",
#                node=node,
#                input=input,
#            )
