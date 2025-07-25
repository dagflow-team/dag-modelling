from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from ..tools.schema import LoadYaml

if TYPE_CHECKING:
    from collections.abc import Callable, Container, Sequence
    from typing import Any


def format_latex(k: str, s: str | Any, /, *args, protect_latex: bool = True, **kwargs) -> str:
    if not isinstance(s, str):
        return s

    if "{" not in s:
        return s

    should_protect_latex = protect_latex and k in {"latex", "axis", "xaxis", "rootaxis"}
    if should_protect_latex and "{" in s and "{{" not in s:  # }} }
        return s

    return s.format(*args, **kwargs)


def format_dict(
    dct: Mapping | str,
    /,
    *args,
    process_keys: tuple | list | set | None = None,
    protect_latex: bool = True,
    **kwargs,
) -> dict:
    if isinstance(dct, str):
        return {"text": format_latex("", dct, *args, **kwargs)}

    ret = {}
    for key, value in dct.items():
        match value:
            case dict():
                ret[key] = format_dict(
                    value, *args, process_keys=process_keys, protect_latex=protect_latex, **kwargs
                )
            case str():
                if process_keys and key not in process_keys:
                    continue
                ret[key] = format_latex(key, value, *args, protect_latex=protect_latex, **kwargs)
            case _:
                ret[key] = value

    return ret


def mapping_append_lists(dct: dict, key: str, lst: list):
    if not isinstance(dct, dict):
        return

    def patch(dct):
        oldlist = dct.get(key, [])
        newlist = list(lst)
        for name in oldlist:
            if name not in newlist:
                newlist.append(name)
        dct[key] = newlist

    has_subdicts = False
    for v in dct.values():
        if isinstance(v, dict):
            mapping_append_lists(v, key, lst)
            has_subdicts = True

    if not has_subdicts:
        patch(dct)


def repr_pretty(self, p, cycle):
    """Pretty repr for IPython.

    To be used as __repr__ method
    """
    p.text("..." if cycle else str(self))


def _make_formatter(fmt: str | Callable | dict | None) -> Callable:
    if isinstance(fmt, str):
        return fmt.format
    elif isinstance(fmt, dict):

        def formatter(s, **_):
            return fmt.get(s, s)

        return formatter
    elif fmt is None:

        def formatter(s, **_):
            return s

        return formatter

    return fmt


def apply_substitutions(
    s: str,
    substitutions: Mapping[str, str] = {},
    *,
    full_string: bool = False,
) -> str:
    if not substitutions or not s or not isinstance(s, str):
        return s

    if substitutions:
        if full_string:
            for pattern, substitution in substitutions.items():
                if s == pattern:
                    return substitution
        else:
            for pattern, substitution in substitutions.items():
                s = s.replace(pattern, substitution)

    return s


class Labels:
    """A class to treat labels.

    Attributes
    ----------
    _text : str
        for plain text (terminal).
    _graph : str
        for the graph, or text_unit.
    _latex : str
        for latex output, or to replace plot_title.
    _mark : str
        for short mark on the graphiz graph.
    _axis : str
        used when the output is associated to the axis.
    _rootaxis : str
        used when the output is associated to the axis (root latex).
    _xaxis : str
        for own X axis, when it is not provided.
    _yaxis : str
        for own Y axis, when it is not provided.
    _plot_title : str
        for plot title, will be replaced by latex if not found.
    _root_title : str
        for canvas title (root), will be replaced by plot_title with \→# substitution.
    _unit : str
        for text unit.
    _latex_unit : str
        for latex text unit.
    _xunit : str
        for text unit for own X axis.
    _yunit : str
        for text unit for own Y axis.
    """

    __slots__ = (
        "_name",
        "_index_values",
        "_index_dict",
        "_text",
        "_graph",
        "_latex",
        "_mark",
        "_axis",
        "_xaxis",
        "_yaxis",
        "_plot_title",
        "_root_title",
        "_rootaxis",
        "_unit",
        "_latex_unit",
        "_xunit",
        "_yunit",
        "_paths",
        "_plotoptions",
        "_node_hidden",
    )

    _name: str | None
    _index_values: list[str]
    _index_dict: dict[str, tuple[str, int]]
    _text: str | None
    _graph: str | None
    _latex: str | None
    _axis: str | None
    _xaxis: str | None
    _yaxis: str | None
    _plot_title: str | None
    _root_title: str | None
    _rootaxis: str | None
    _unit: str | None
    _latex_unit: str | None
    _xunit: str | None
    _yunit: str | None
    _mark: str | None
    _paths: list[str]
    _plotoptions: dict[str, Any]
    _node_hidden: bool | None

    def __init__(self, label: Mapping[str, str] | str | Path | None = None):
        for slot in self.__slots__:
            setattr(self, slot, None)
        self._paths = []
        self._index_values = []
        self._index_dict = {}
        self._node_hidden = False
        self._plotoptions = {}

        if isinstance(label, str):
            if label.endswith(".yaml"):
                self._update_from(label)
            else:
                self._text = label
        elif isinstance(label, Path):
            self._update_from(str(label))
        elif isinstance(label, dict):
            self.update(label)

    def __str__(self):
        return str(
            {
                slot.removeprefix("_"): v
                for slot in self.__slots__
                if (v := getattr(self, slot)) is not None
            }
        )

    _repr_pretty_ = repr_pretty

    def _update_from(self, path: str):
        d = LoadYaml(path)
        self.update(d)

    def update(self, d: Mapping[str, str | dict[str, str]]):
        match d:
            case {"group": {} as group, **rest} if not rest:
                d = group
                for k, v in d.items():
                    d[k] = v.format(space_key="", key_space="", key="", index=())

        for k, v in d.items():
            setattr(self, k, v)

    def format(self, *args, **kwargs):
        for name in (
            "text",
            "graph",
            "latex",
            "axis",
            "xaxis",
            "yaxis",
            "plot_title",
            "root_title",
            "rootaxis",
            "unit",
            "latex_unit",
            "xunit",
            "yunit",
        ):
            aname = f"_{name}"
            oldvalue = getattr(self, aname)
            newvalue = format_latex(name, oldvalue, *args, **kwargs)
            setattr(self, aname, newvalue)

    @property
    def name(self) -> str | None:
        return self._name

    @property
    def index_dict(self) -> dict[str, tuple[str, int]]:
        return self._index_dict

    def index_in_mask(
        self, accepted_items: Mapping[str, str | int | Container[str | int]] | None
    ) -> bool:
        if accepted_items is None:
            return True

        index = self.index_dict
        for category, accepted_list in accepted_items.items():
            if (idxnum := index.get(category)) is None:
                continue

            if isinstance(accepted_list, str):
                if idxnum[0] != accepted_list:  # key value
                    return False
            elif isinstance(accepted_list, int):
                if idxnum[1] != accepted_list:  # key index
                    return False
            elif (
                idxnum[0] not in accepted_list  # key value
                and idxnum[1] not in accepted_list  # key index
            ):
                return False

        return True

    @index_dict.setter
    def index_dict(self, index_dict: dict[str, tuple[str, int]]):
        self._index_dict = index_dict

    def build_index_dict(self, index: Mapping[str, Sequence[str]] = {}):
        if not index or self.index_dict:
            return

        if not self.index_values and self.paths:
            path = self.paths[0]
            self.index_values = path.split(".")

        to_remove = []
        index_values = self.index_values
        for value in index_values:
            for category, possible_values in index.items():
                try:
                    pos = possible_values.index(value)
                except ValueError:
                    continue
                else:
                    self.index_dict[category] = value, pos
                    break
            else:
                to_remove.append(value)
        for v in to_remove:
            index_values.remove(v)

    @property
    def index_values(self) -> list[str]:
        return self._index_values

    @index_values.setter
    def index_values(self, index_values: list[str]):
        self._index_values = list(index_values)

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def text(self) -> str | None:
        return self._text

    @text.setter
    def text(self, value: str):
        self._text = value

    @property
    def text_unit(self) -> str | None:
        if unit := self.unit:
            text = self._text
            if isinstance(text, str) and "\n" in text:
                return text.replace("\n", f"[{unit}] \n", 1)
            else:
                return f"{text} [{unit}]"

        return self._text

    @property
    def graph(self) -> str | None:
        if self._graph:
            return self._graph

        return self.text_unit

    @graph.setter
    def graph(self, value: str | None):
        self._graph = value

    @property
    def latex(self) -> str | None:
        return self._latex or self._text

    def get_latex(self, *, substitutions: Mapping[str, str] = {}) -> str:
        if self._latex:
            return apply_substitutions(self._latex, substitutions)

        return apply_substitutions((self._text or "").replace(r"\n", "\n") or "", substitutions)

    @latex.setter
    def latex(self, value: str):
        self._latex = value

    @property
    def latex_unit(self) -> str | None:
        if unit := self.latex_unit:
            return f"{self.latex} [{unit}]"

        return self.latex

    @property
    def plot_title(self) -> str | None:
        return self._plot_title or self._latex or self._text.replace(r"\n", "\n")

    def get_plot_title(self, *, substitutions: Mapping[str, str] = {}) -> str | None:
        return self._plot_title or self.get_latex(substitutions=substitutions)

    @plot_title.setter
    def plot_title(self, value: str | None):
        self._plot_title = value

    @property
    def root_title(self) -> str | None:
        if self._root_title is not None:
            return self._root_title
        return _latex_to_root(self.plot_title)

    @root_title.setter
    def root_title(self, value: str | None):
        self._root_title = value

    def get_root_title(self, **kwargs) -> str | None:
        if self._root_title is not None:
            return self._root_title

        return _latex_to_root(self.get_plot_title(**kwargs))

    @property
    def rootaxis(self) -> str | None:
        return _latex_to_root(self.axis) if self._rootaxis is None else self._rootaxis

    @rootaxis.setter
    def rootaxis(self, value: str | None):
        self._rootaxis = value

    @property
    def rootaxis_unit(self) -> str | None:
        return _latex_to_root(self.axis_unit) if self._rootaxis is None else self._rootaxis

    @property
    def axis(self) -> str | None:
        return self._axis or self.plot_title

    @axis.setter
    def axis(self, value: str | None):
        self._axis = value

    @property
    def axis_unit(self) -> str | None:
        if unit := self.latex_unit:
            return f"{self.axis} [{unit}]"

        return self.axis

    @property
    def xaxis(self) -> str | None:
        return self._xaxis

    @property
    def xaxis_unit(self) -> str | None:
        if unit := self.xunit:
            return f"{self.xaxis} [{unit}]"

        return self.xaxis

    @xaxis.setter
    def xaxis(self, value: str | None):
        self._xaxis = value

    @property
    def yaxis(self) -> str | None:
        return self._yaxis

    @property
    def yaxis_unit(self) -> str | None:
        if unit := self.yunit:
            return f"{self.yaxis} [{unit}]"

        return self.yaxis

    @yaxis.setter
    def yaxis(self, value: str | None):
        self._yaxis = value

    @property
    def unit(self) -> str | None:
        return self._unit

    @unit.setter
    def unit(self, value: str | None):
        self._unit = value

    @property
    def latex_unit(self) -> str | None:
        return self._latex_unit or self.unit

    @latex_unit.setter
    def latex_unit(self, value: str | None):
        self._latex_unit = value

    @property
    def xunit(self) -> str | None:
        return self._xunit

    @xunit.setter
    def xunit(self, value: str | None):
        self._xunit = value

    @property
    def yunit(self) -> str | None:
        return self._yunit

    @yunit.setter
    def yunit(self, value: str | None):
        self._yunit = value

    @property
    def mark(self) -> str | None:
        return self._mark

    @mark.setter
    def mark(self, value: str):
        self._mark = value

    @property
    def path(self) -> str | None:
        try:
            return self._paths[0]
        except IndexError:
            return None

    @property
    def paths(self) -> list[str]:
        return self._paths

    @paths.setter
    def paths(self, value: list[str]):
        self._paths = value

    @property
    def plotable(self) -> bool:
        if not self._plotoptions:
            return True

        return self._plotoptions.get("method") != "none"

    @property
    def plotoptions(self) -> dict[str, Any]:
        return self._plotoptions

    @plotoptions.setter
    def plotoptions(self, value: Mapping[str, Any] | str):
        match value:
            case str():
                self._plotoptions = {"method": value.lower()}
            case Mapping():
                self._plotoptions = dict(value)
            case _:
                raise ValueError(value)

    @property
    def node_hidden(self) -> bool:
        return bool(self._node_hidden)

    @node_hidden.setter
    def node_hidden(self, value: bool):
        self._node_hidden = value

    def items(self):
        for k in self.__slots__:
            yield k, getattr(self, k)

    def __getitem__(self, k: str) -> str | None:
        return getattr(self, k)

    def __setitem__(self, k: str, v: str) -> None:
        setattr(self, k, v)

    def get(self, k: str, default: str) -> str | None:
        return getattr(self, k, default)

    def setdefault(self, k: str, default: str) -> str | None:
        if (value := getattr(self, k, None)) is not None:
            return value

        setattr(self, k, default)
        return default

    def setdefaults(self, labels: dict):
        for k, v in labels.items():
            if getattr(self, k) is None:
                setattr(self, k, v)

    def copy(self) -> Labels:
        l = Labels()
        for slot in self.__slots__:
            setattr(l, slot, getattr(self, slot))
        return l

    def inherit(
        self,
        source: Labels | Mapping,
        fmtlong: str | Callable | None = None,
        fmtshort: str | Callable | None = None,
        fields: Sequence[str] = [],
        fields_exclude: Container[str] = [],
        fmtextra: Mapping[str, str] = {},
    ):
        fmtlong = _make_formatter(fmtlong)
        fmtshort = _make_formatter(fmtshort)

        if fields:
            inherit = tuple(f"_{s}" for s in fields)
        else:
            inherit = (
                "_text",
                "_graph",
                "_latex",
                "_mark",
                "_axis",
                "_plot_title",
                "_index_values",
                "_index_dict",
                "_paths",
                "_node_hidden",
            )
        kshort = {"_mark"}
        for _key in inherit:
            if _key[1:] in fields_exclude:
                continue
            if isinstance(source, Labels):
                label = getattr(source, _key, None)
            elif isinstance(source, Mapping):
                key = _key[1:]
                label = source.get(key, None)
            else:
                raise ValueError(source)
            if label is None:
                continue
            match label:
                case str():
                    formatter = fmtshort if _key in kshort else fmtlong
                    newv = formatter(label, source=source)
                    if newv is not None:
                        self[_key] = newv
                case tuple() | {} | []:
                    self[_key] = type(label)(label)
                case _:
                    self[_key] = label

        for key, fmt in fmtextra.items():
            _key = f"_{key}"
            if getattr(self, _key) is not None:
                continue

            formatter = _make_formatter(fmt)
            self[_key] = formatter(source=source)


def inherit_labels(
    source: dict,
    destination: dict | None = None,
    *,
    fmtlong: str | Callable,
    fmtshort: str | Callable,
    fields_exclude: Container[str] = [],
) -> dict:
    if destination is None:
        destination = {}

    fmtlong = _make_formatter(fmtlong)
    fmtshort = _make_formatter(fmtshort)

    kshort = {"mark"}
    kskip = {"key", "name"}
    for k, v in source.items():
        if k in fields_exclude:
            continue
        if k in kskip:
            continue
        if isinstance(v, str):
            newv = fmtshort(v) if k in kshort else fmtlong(v)
            if newv is not None:
                destination[k] = newv
        else:
            destination[k] = v

    return destination


def _latex_to_root(text: str | None) -> str | None:
    return (
        text.replace(r"\rm ", "")
        .replace(r"\overline", r"\bar")
        .replace(r"\\", " | ")
        .replace("\\", "#")
        .replace("$", "")
        if text
        else text
    )
