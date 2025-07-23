from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .input import Input
    from .node_base import NodeBase
    from .output import Output


class DagModellingError(RuntimeError):
    node: NodeBase | None
    input: Input | None
    output: Output | None

    def __init__(
        self,
        message: str,
        node: NodeBase | None = None,
        *,
        input: Input | None = None,
        output: Output | None = None,
        args: tuple[str, ...] | None = None,
    ):
        if node:
            message = f"{message} [node={getattr(node, 'name', node)}]"
        if input:
            message = f"{message} [input={getattr(input, 'name', input)}]"
        if output:
            message = f"{message} [output={getattr(output, 'name', output)}]"
        super().__init__(message)
        self.node = node
        self.input = input
        self.output = output

        if node is not None and hasattr(node, "_exception"):
            node._exception = message if args is None else "\\n".join((message,) + args)


class CriticalError(DagModellingError):
    pass


class NoncriticalError(DagModellingError):
    pass


class InitializationError(CriticalError):
    def __init__(self, message: str | None = None, *args, **kwargs):
        if not message:
            message = "Wrong initialization!"
        super().__init__(message, *args, **kwargs)


class AllocationError(CriticalError):
    def __init__(self, message: str | None = None, *args, **kwargs):
        if not message:
            message = "Unable to allocate memory!"
        super().__init__(message, *args, **kwargs)


class ClosingError(CriticalError):
    def __init__(self, message: str | None = None, *args, **kwargs):
        if not message:
            message = "An exception occured during closing procedure!"
        super().__init__(message, *args, **kwargs)


class OpeningError(CriticalError):
    def __init__(self, message: str | None = None, *args, **kwargs):
        if not message:
            message = "An exception occured during opening procedure!"
        super().__init__(message, *args, **kwargs)


class ClosedGraphError(CriticalError):
    def __init__(self, message: str | None = None, *args, **kwargs):
        if not message:
            message = "Unable to modify a closed graph!"
        super().__init__(message, *args, **kwargs)


class UnclosedGraphError(CriticalError):
    def __init__(self, message: str | None = None, *args, **kwargs):
        if not message:
            message = "The graph is not closed!"
        super().__init__(message, *args, **kwargs)


class TypeFunctionError(CriticalError):
    def __init__(self, message: str | None = None, *args, **kwargs):
        if not message:
            message = "An exception occurred during type function processing!"
        super().__init__(message, *args, **kwargs)


class ReconnectionError(CriticalError):
    def __init__(self, message: str | None = None, *args, **kwargs):
        if not message:
            message = "The object is already connected!"
        super().__init__(message, *args, **kwargs)


class ConnectionError(CriticalError):
    def __init__(self, message: str | None = None, *args, **kwargs):
        if not message:
            message = "An exception occurred during connection!"
        super().__init__(message, *args, **kwargs)


class CalculationError(CriticalError):
    def __init__(self, message: str | None = None, *args, **kwargs):
        if not message:
            message = "An exception occurred during calculation!"
        super().__init__(message, *args, **kwargs)
