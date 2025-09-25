from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Sequence

from ..core.node import Node
from ..core.output import Output

if TYPE_CHECKING:
    from collections.abc import Callable
    from ..parameters import Parameter

    from numpy.typing import NDArray


def make_fcn(
    node_or_output: Node | Output,
    parameters: Sequence[Parameter] | Mapping[str, Parameter],
    safe: bool = True,
) -> Callable:
    """Retrun a function, which takes the parameter values as arguments and retruns the result of the node evaluation.

    Supports positional and key-word arguments. Posiotion of parameter is
    determined by index in the `parameters` list.

    Parameters
    ----------
    node_or_output : Node | Output
        A node (or output), depending (explicitly or implicitly) on the parameters.
    parameters : Sequence[Parameter] | Mapping[str, Parameter]
        The names of the set of parameters that the function will depend on.
    safe : bool
        If `safe=True`, the parameters will be resetted to old values after evaluation.
        If `safe=False`, the parameters will be setted to the new values.

    Returns
    -------
    Callable
        Function that depends on set of parameters with `parameters` names.
    """
    # to avoid extra checks in the function, we prepare the corresponding getter here
    output, outputs = None, None
    if isinstance(node_or_output, Output):
        output = node_or_output
    elif isinstance(node_or_output, Node):
        if len(node_or_output.outputs) == 1:
            output = node_or_output.outputs[0]
        else:
            outputs = tuple(node_or_output.outputs.pos_edges_list)
    else:
        raise ValueError(
            f"`node_or_output` must be Node | Output, but given {node_or_output}, {type(node_or_output)=}!"
        )

    match safe, output:
        case True, None:

            def _get_data():  # pyright: ignore [reportRedeclaration]
                return tuple(
                    out.data.copy() for out in outputs  # pyright: ignore [reportOptionalIterable]
                )

        case False, None:

            def _get_data():  # pyright: ignore [reportRedeclaration]
                tuple(out.data for out in outputs)  # pyright: ignore [reportOptionalIterable]

        case True, Output():

            def _get_data():
                return output.data.copy()

        case False, Output():

            def _get_data():
                return output.data

    # the dict with parameters
    _pars_dict = {}
    if isinstance(parameters, Mapping):
        _pars_dict = dict(parameters)
    elif isinstance(parameters, Sequence):
        _pars_dict = {f"par{idx}": parameter for idx, parameter in enumerate(parameters)}
    else:
        raise RuntimeError(f"Couldn't obtain {type(parameters)=}")

    if not safe:

        def fcn_not_safe(
            *args: float | int, **kwargs: float | int
        ) -> NDArray | tuple[NDArray, ...] | None:
            if len(args) > len(_pars_dict):
                raise RuntimeError(
                    f"Too many posiitional values are provided: {len(args)} [>{len(_pars_dict)}]"
                )
            for par, val in zip(_pars_dict.values(), args):
                par.value = val

            for name, val in kwargs.items():
                _pars_dict[name].value = val
            node_or_output.touch()
            return _get_data()

        return fcn_not_safe

    def fcn_safe(*args: float | int, **kwargs: float | int) -> NDArray | tuple[NDArray, ...] | None:
        if len(args) > len(_pars_dict):
            raise RuntimeError(
                f"Too many posiitional values are provided: {len(args)} [>{len(_pars_dict)}]"
            )

        pars = []
        for par, val in zip(_pars_dict.values(), args):
            par.push(val)
            pars.append(par)

        for name, val in kwargs.items():
            _pars_dict[name].push(val)
            pars.append(_pars_dict[name])
        node_or_output.touch()
        res = _get_data()
        for par in pars:
            par.pop()
        node_or_output.touch()
        return res

    return fcn_safe
