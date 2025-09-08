from __future__ import annotations

from typing import TYPE_CHECKING

from nested_mapping import NestedMapping

from ..parameters import Parameter
from .node import Node
from .output import Output

if TYPE_CHECKING:
    from collections.abc import Callable, KeysView

    from numpy.typing import NDArray


def _find_par_permissive(storage: NestedMapping, name: str) -> Parameter | None:
    for key, par in storage.walkitems():
        if key[-1] == name and isinstance(par, Parameter):
            return par


def _collect_pars_permissive(
    storage: NestedMapping, par_names: list[str] | tuple[str, ...] | KeysView
) -> dict[str, Parameter]:
    res = {}
    for name in par_names:
        if (par := _find_par_permissive(storage, name)) is not None:
            res[name] = par
    return res


def make_fcn(
    node_or_output: Node | Output,
    storage: NestedMapping,
    safe: bool = True,
    par_names: list[str] | tuple[str, ...] | None = None,
) -> Callable:
    """Retruns a function, which takes the parameter values as arguments and
    retruns the result of the node evaluation.

    :param node_or_output: A node (or output), depending (explicitly or implicitly) on the parameters
    :type node: class:`dag_modelling.core.node.Node` | class:`dag_modelling.core.output.Output`
    :param storage: A storage with parameters
    :type storage: class:`nested_mapping.NestedMapping` (including `dag_modelling.core.storage.NodeStorage`)
    :param safe: If `safe=True`, the parameters will be resetted to old values after evaluation.
    If `safe=False`, the parameters will be setted to the new values
    :type safe: bool
    :param par_names: The short names of the set of parameters for presearch
    :type par_names: list[str] | tuple[str,...] | None
    :rtype: function
    """
    if not isinstance(storage, NestedMapping):
        raise ValueError(f"`storage` must be NestedMapping, but given {storage}, {type(storage)=}!")

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
        raise ValueError(f"`node` must be Node | Output, but given {node}, {type(node)=}!")

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

    # the dict with parameters found from the presearch
    _pars_dict = _collect_pars_permissive(storage, par_names) if par_names else {}
    _pars_list = list(_pars_dict.values())

    def _get_parameter_by_name(name: str) -> Parameter:
        """Gets a parameter from the parameters dict, which stores the
        parameters found from the "fuzzy" search, or try to get the parameter
        from the storage, supposing that the name is the precise key in the
        storage."""
        try:
            return _pars_dict[name]
        except KeyError as exc:
            try:
                return storage[name]
            except KeyError:
                raise RuntimeError(f"There is no parameter '{name}' in the {storage=}!") from exc

    if not safe:

        def fcn_not_safe(
            *args: float | int, **kwargs: float | int
        ) -> NDArray | tuple[NDArray, ...] | None:
            if len(args) > len(_pars_list):
                raise RuntimeError(
                    f"Too much parameter values provided: {len(args)} [>{len(_pars_list)}]"
                )
            for par, val in zip(_pars_dict.values(), args):
                par.value = val

            for name, val in kwargs.items():
                par = _get_parameter_by_name(name)
                par.value = val
            node_or_output.touch()
            return _get_data()

        return fcn_not_safe

    def fcn_safe(*args: float | int, **kwargs: float | int) -> NDArray | tuple[NDArray, ...] | None:
        if len(args) > len(_pars_list):
            raise RuntimeError(
                f"Too much parameter values provided: {len(args)} [>{len(_pars_list)}]"
            )

        pars = []
        for par, val in zip(_pars_dict.values(), args):
            par.push(val)
            pars.append(par)

        for name, val in kwargs.items():
            par = _get_parameter_by_name(name)
            par.push(val)
            pars.append(par)
        node_or_output.touch()
        res = _get_data()
        for par in pars:
            par.pop()
        node_or_output.touch()
        return res

    return fcn_safe
