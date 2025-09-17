from __future__ import annotations

from typing import TYPE_CHECKING

from nested_mapping import NestedMapping

from ..parameters import Parameter
from ..core.node import Node
from ..core.output import Output

if TYPE_CHECKING:
    from collections.abc import Callable, KeysView

    from numpy.typing import NDArray


def _find_par(storage: NestedMapping, name: str) -> Parameter | None:
    """Find parameter in storage (permissive).

    More than one parameter might contain `name` in their paths.
    It will return only first item.

    Parameters
    ----------
    storage : NestedMapping
        A storage with parameters.
    name : str
        Name of parameter (might be with periods).

    Returns
    -------
    Parameter | None
        Parameter that contains `name` in path of parameter in storage.

    """
    for key, par in storage.walkjoineditems():
        if key == name and isinstance(par, Parameter):
            return par


def _collect_pars(
    storage: NestedMapping, par_names: list[str] | tuple[str, ...] | KeysView
) -> dict[str, Parameter]:
    """Collect parameters with `par_names` in dictionary (parameter name, parameter).

    Parameters
    ----------
    storage : NestedMapping
        A storage with parameters.
    par_names : list[str] | tuple[str, ...] | KeysView
        The names of the set of parameters that the function will depend on.

    Returns
    -------
    dict[str, Parameter]
        Dictionary that contains pairs (parameter name, parameters).
    """
    res = {}
    for name in par_names:
        if (par := _find_par(storage, name)) is not None:
            res[name] = par
    return res


def make_fcn(
    node_or_output: Node | Output,
    storage: NestedMapping,
    par_names: list[str] | tuple[str],
    safe: bool = True,
    mapper: dict[str, str] | None = None,
) -> Callable:
    """Retrun a function, which takes the parameter values as arguments and retruns the result of the node evaluation.
    
    Supports positional and key-word arguments. Posiotion of parameter is
    determined by index in the `par_names` list.

    Parameters
    ----------
    node_or_output : Node | Output
        A node (or output), depending (explicitly or implicitly) on the parameters.
    storage : NestedMapping
        A storage with parameters.
    par_names : list[str] | tuple[str, ...] | None
        The names of the set of parameters that the function will depend on.
    safe : bool
        If `safe=True`, the parameters will be resetted to old values after evaluation.
        If `safe=False`, the parameters will be setted to the new values.
    mapper : dict[str, str] | None
        The mapping of original parameter names to short names.

    Returns
    -------
    Callable
        Function that depends on set of parameters with `par_names` names.
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

    # the dict with parameters
    _pars_dict = _collect_pars(storage, par_names) if par_names else {}
    if mapper:
        _pars_dict = {mapper.get(parname, parname): parvalue for parname, parvalue in _pars_dict.items()}
    _pars_list = list(_pars_dict.values())

    def _get_parameter_by_name(name: str) -> Parameter:
        """Get a parameter from the parameters dict, which stores the parameters found from the "fuzzy" search.

        Parameters
        ----------
        name : str
            Name of parameter from `par_names` or `mapper`.

        Returns
        -------
        Parameter
            Parameter from `_pars_dict`.
        """
        try:
            return _pars_dict[name]
        except KeyError:
            raise KeyError(f"Parameter '{name}' was not passed to `par_names` or `mapper`, only {_pars_dict.keys()} are allowed!")

    if not safe:

        def fcn_not_safe(
            *args: float | int, **kwargs: float | int
        ) -> NDArray | tuple[NDArray, ...] | None:
            if len(args) > len(_pars_list):
                raise RuntimeError(
                    f"Too much parameter values provided: {len(args)} [>{len(_pars_list)}]"
                )
            if len(args) + len(kwargs)  > len(_pars_list):
                raise RuntimeError(
                    f"Possible overwritting of parameters: {len(args) + len(kwargs)} were passed, but only {len(_pars_list)} are allowed"
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
        if len(args) + len(kwargs) > len(_pars_list):
            raise RuntimeError(
                f"Possible overwritting of parameters: {len(args) + len(kwargs)} were passed, but only {len(_pars_list)} are allowed"
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
