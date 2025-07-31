from __future__ import annotations

from typing import TYPE_CHECKING

from ..parameters import Parameter
from .node import Node
from .output import Output
from .storage import NestedMapping, NodeStorage

if TYPE_CHECKING:
    from collections.abc import Callable, KeysView

    from numpy.typing import NDArray


def _find_par_permissive(storage: NodeStorage | NestedMapping, name: str) -> Parameter | None:
    for key, par in storage.walkitems():
        if key[-1] == name and isinstance(par, Parameter):
            return par


def _collect_pars_permissive(
    storage: NodeStorage | NestedMapping, par_names: list[str] | tuple[str, ...] | KeysView
) -> dict[str, Parameter]:
    res = {}
    for name in par_names:
        if (par := _find_par_permissive(storage, name)) is not None:
            res[name] = par
    return res


def make_fcn(
    node: Node | Output,
    storage: NodeStorage | NestedMapping,
    safe: bool = True,
    par_names: list[str] | tuple[str, ...] | None = None,
) -> Callable:
    """
    Retruns a function, which takes the parameter values as arguments
    and retruns the result of the node evaluation.

    :param node: A node (or output), depending (explicitly or implicitly) on the parameters
    :type node: class:`dag_modelling.core.node.Node` | class:`dag_modelling.core.output.Output`
    :param storage: A storage with parameters
    :type storage: class:`dag_modelling.core.storage.NodeStorage`
    :param safe: If `safe=True`, the parameters will be resetted to old values after evaluation.
    If `safe=False`, the parameters will be setted to the new values
    :type safe: bool
    :param par_names: The short names of the set of parameters for presearch
    :type par_names: list[str] | tuple[str,...] | None
    :rtype: function
    """
    if not isinstance(storage, (NodeStorage, NestedMapping)):
        raise ValueError(
            f"`storage` must be NodeStorage | NestedMapping, but given {storage}, {type(storage)=}!"
        )

    # to avoid extra checks in the function, we prepare the corresponding getter here
    if isinstance(node, Output):
        _get_data = (lambda: node.data.copy()) if safe else (lambda: node.data)
    elif isinstance(node, Node):
        outputs = node.outputs
        if (nout := len(outputs)) == 0:
            _get_data = lambda: None
        elif nout == 1:
            _get_data = (lambda: outputs[0].data.copy()) if safe else (lambda: outputs[0].data)
        elif safe:
            _get_data = lambda: tuple(out.data.copy() for out in outputs)
        else:
            _get_data = lambda: tuple(out.data for out in outputs)
    else:
        raise ValueError(f"`node` must be Node | Output, but given {node}, {type(node)=}!")

    # the dict with parameters found from the presearch
    _parsdict = _collect_pars_permissive(storage, par_names) if par_names else {}

    def _get_parameter(name: str) -> Parameter:
        """
        Gets a parameter from the parameters dict,
        which stores the parameters found from the "fuzzy" search,
        or try to get the parameter from the storage,
        supposing that the name is the precise key in the storage
        """
        try:
            par = _parsdict[name]
        except KeyError as exc:
            try:
                par = storage[name]
            except KeyError:
                raise RuntimeError(f"There is no parameter '{name}' in the {storage=}!") from exc
        return par

    if not safe:

        def fcn_unsafe(**kwargs) -> NDArray | tuple[NDArray, ...] | None:
            for name, val in kwargs.items():
                par = _get_parameter(name)
                par.value = val
            node.touch()
            return _get_data()

        return fcn_unsafe

    def fcn_safe(**kwargs) -> NDArray | tuple[NDArray, ...] | None:
        pars = []
        for name, val in kwargs.items():
            par = _get_parameter(name)
            par.push(val)
            pars.append(par)
        node.touch()
        res = _get_data()
        for par in pars:
            par.pop()
        node.touch()
        return res

    return fcn_safe
