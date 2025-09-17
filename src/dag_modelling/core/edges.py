from collections.abc import Sequence
from typing import Any

from .exception import CriticalError
from .iter import IsIterable


class EdgeContainer:
    __slots__ = (
        "_kw_edges",
        "_pos_edges",
        "_pos_edges_list",
        "_nonpos_edges",
        "_all_edges",
        "_dtype",
    )
    _kw_edges: dict[str, Any]
    _pos_edges_list: list
    _pos_edges: dict
    _nonpos_edges: dict[str, Any]
    _all_edges: dict[str, Any]

    def __init__(self, iterable=None):
        self._kw_edges = {}
        self._pos_edges_list = []
        self._pos_edges = {}
        self._nonpos_edges = {}
        self._all_edges = {}
        self._dtype = None
        if iterable:
            self.add(iterable)

    def __str__(self):
        _type = self._dtype.__name__ if self._dtype else "Edge"
        return f"{_type}: {tuple(self._all_edges)}"

    def add(
        self,
        value: Any,
        *,
        name: str | None = None,
        positional: bool = True,
        keyword: bool = True,
        merge: bool = False,
    ):
        if positional == keyword == False:
            raise RuntimeError("Edge should be at least positional or a keyword")
        if positional and merge:
            raise RuntimeError("May not merge positional limbs")

        if IsIterable(value):
            for v in value:
                self.add(v, positional=positional, keyword=keyword)
            return self
        if self._dtype:
            if not isinstance(value, self._dtype):
                raise RuntimeError(
                    f"The type {type(value)} of the data doesn't correpond " f"to {self._dtype}!"
                )
        else:
            self._dtype = type(value)
        name = name or value.name
        if not name:
            raise RuntimeError("May not add objects with undefined name")
        if name in self._all_edges and not (name in self._kw_edges and merge):
            raise RuntimeError(f"May not add duplicated items: {name}")

        if keyword:
            if merge:
                prevlimb = self._kw_edges.get(name, ())
                value = prevlimb + (value,)
            self._kw_edges[name] = value

        self._all_edges[name] = value
        if positional:
            self._pos_edges_list.append(value)
            self._pos_edges[name] = value
        else:
            self._nonpos_edges[name] = value
        return self

    def make_positional(self, name: str, *, index: int | None = None) -> Any:
        try:
            limb = self._kw_edges[name]
        except KeyError as exc:
            raise RuntimeError(f"Invalid keyword limb {name}") from exc

        if limb in self._pos_edges_list:
            raise RuntimeError(f"Limb {name} is already positional")

        if index is None:
            self._pos_edges_list.append(limb)
        else:
            try:
                self._pos_edges_list[index] = limb
            except IndexError as e:
                if index==len(self._pos_edges_list):
                    self._pos_edges_list.append(limb)
                else:
                    raise RuntimeError(f"EdgeContainer.make_positional: invalid index {index}") from e


        self._pos_edges[name] = limb
        del self._nonpos_edges[name]
        return limb

    def make_positionals(self, *names) -> list[Any]:
        return [self.make_positional(name) for name in names]

    def allocate(self) -> bool:
        """returns True if any data was reassigned"""
        data_reassigned = False
        for edge in self._all_edges.values():
            data_reassigned |= edge.allocate()

        return data_reassigned

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._kw_edges[key]
        elif isinstance(key, (int, slice)):
            return self._pos_edges_list[key]
        elif isinstance(key, Sequence):
            return tuple(self.__getitem__(k) for k in key)
        raise TypeError(f"Unsupported key type: {type(key).__name__}")

    @property
    def kw(self) -> dict:
        return self._kw_edges

    @property
    def kw_edges(self) -> dict:
        return self._kw_edges

    @property
    def all_edges(self) -> dict:
        return self._all_edges

    @property
    def pos_edges(self) -> dict:
        return self._pos_edges

    @property
    def pos_edges_list(self) -> list:
        return self._pos_edges_list

    @property
    def nonpos_edges(self) -> dict:
        return self._nonpos_edges

    def get(self, key, default=None):
        try:
            return self.__getitem__(key)
        except Exception:
            return default

    def has_key(self, key: str) -> bool:
        return key in self._kw_edges

    def get_pos(self, idx: int):
        """Get positional leg"""
        return self._pos_edges_list[idx]

    iat = get_pos

    def key(self, arg):
        for key, value in self._kw_edges.items():
            if value is arg:
                return key

        raise ValueError()

    def index(self, arg):
        return self._pos_edges_list.index(arg)

    def get_kw(self, key: str):
        """Return keyword leg"""
        return self._kw_edges[key]

    kat = get_kw

    def len_pos(self) -> int:
        """Returns a number of the positional limbs"""
        return len(self._pos_edges_list)

    __len__ = len_pos

    def len_kw(self) -> int:
        """Returns a number of the keyword limbs"""
        return len(self._kw_edges)

    def len_all(self) -> int:
        """Returns a number of the all limbs"""
        return len(self._all_edges)

    def __iter__(self):
        return iter(self._pos_edges_list)

    def iter_all(self):
        return iter(self._all_edges.values())

    def iter_all_items(self):
        return iter(self._all_edges.items())

    def iter_kw(self):
        return iter(self._kw_edges.values())

    def iter_kw_items(self):
        return iter(self._kw_edges.items())

    def iter_nonpos(self):
        return iter(self._nonpos_edges.values())

    def iter_data(self):
        for edge in self._pos_edges_list:
            yield edge.data

    def iter_data_unsafe(self):
        for edge in self._pos_edges_list:
            yield edge._data

    def iter(
        self,
        key: int | str | slice | Sequence,
        *,
        include_kw: bool = False,
        exclude_pos: bool = False,
    ):
        if not include_kw and exclude_pos:
            raise RuntimeError(
                "EdgeContainer.iter(): unable to set {include_kw=} and {exclude_pos=}"
            )
        if isinstance(key, int):
            yield self._pos_edges_list[key]
        elif isinstance(key, str):
            yield self._kw_edges[key]
        elif isinstance(key, slice):
            if key == slice(None):
                if include_kw:
                    if exclude_pos:
                        yield from self.iter_nonpos()
                    else:
                        yield from self._all_edges.values()
                else:
                    yield from self._pos_edges_list
            else:
                yield from self._pos_edges_list[key]
        elif isinstance(key, Sequence):
            for subkey in key:
                if isinstance(subkey, int):
                    yield self._pos_edges_list[subkey]
                elif isinstance(subkey, str):
                    yield self._kw_edges[subkey]
                elif isinstance(subkey, slice):
                    yield from self._pos_edges_list[subkey]
                else:
                    raise CriticalError(f"Invalid subkey type {type(subkey).__name__}")
        else:
            raise CriticalError(f"Invalid key type {type(key).__name__}")

    def __contains__(self, name):
        return name in self._all_edges

    def _replace(self, old, new):
        replaced = False

        for k, v in self._kw_edges.items():
            if old is v:
                self._kw_edges[k] = new
                replaced = True

        for i, v in enumerate(self._pos_edges_list):
            if old is v:
                self._pos_edges_list[i] = new
                replaced = True

        if not replaced:
            raise CriticalError("Unable to replace an output/input (not found)")
