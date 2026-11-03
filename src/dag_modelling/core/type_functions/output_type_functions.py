from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import issubdtype, result_type

from ..data_descriptor import Layout
from ..exception import TypeFunctionError
from .tools_for_type_functions import AllPositionals, LimbKey

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from ..node import Node
    from ..output import Output


def set_csr_output(node: Node, output: Output, capacity: int) -> None:
    """Give `output` a sparse-csr layout of `capacity` and select the csr kernel."""
    output.dd.layout = Layout.SPARSE_CSR.value
    output.dd.capacity = capacity
    node.choose_function("csr")


def check_number_of_outputs(node: Node, n: int) -> None:
    """Checking if the node has only `n` outputs"""
    if (nout := len(node.outputs)) != n:
        raise TypeFunctionError(
            f"The node must have only {n} outputs, but given {nout}!", node=node
        )


def evaluate_dtype_of_outputs(
    node: Node,
    inputkey: LimbKey = AllPositionals,
    outputkey: LimbKey = AllPositionals,
) -> None:
    """Automatic calculation and setting dtype for the output"""
    inputs = node.inputs.iter(inputkey)
    outputs = node.outputs.iter(outputkey)

    dtype = result_type(*(inp.dd.dtype for inp in inputs))
    for output in outputs:
        output.dd.dtype = dtype


def check_subtype_of_outputs(node: Node, outputkey: LimbKey, *, dtype: DTypeLike):
    """Checks if the output dtype is some subtype of `dtype`."""
    for output in node.outputs.iter(outputkey):
        if not issubdtype(output.dd.dtype, dtype):
            raise TypeFunctionError(
                f"The output must be an array of {dtype}, but given '{output.dd.dtype}'!",
                node=node,
                output=output,
            )
