from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from os import R_OK, access
from pathlib import Path
from typing import TYPE_CHECKING

from schema import And, Or, Schema, SchemaError, Use

from .logger import INFO1, logger

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Any

IsStrSeq = Or((str,), And([str], Use(tuple)))
IsStrSeqOrStr = Or(IsStrSeq, And(str, Use(lambda s: (s,))))


def IsReadable(filename: str):
    """Returns True if the file is readable"""
    return access(filename, R_OK)


IsFilename = Or(str, And(Path, Use(str)))

IsReadableFilename = And(IsFilename, IsReadable)

IsFilenameSeqOrFilename = Or([IsFilename], (IsFilename,), And(IsFilename, Use(lambda s: (s,))))

IsReadableFilenameSeqOrFilename = Or(
    [IsReadableFilename],
    (IsReadableFilename,),
    And(IsReadableFilename, Use(lambda s: (s,))),
)


def IsFilewithExt(*exts: str):
    """Returns a function that retunts True if the file extension is consistent"""
    return lambda filename: filename.endswith(exts)


def AllFileswithExt(*exts: str):
    """Returns a function that retunts True if the file extensions are consistent"""

    def checkfilenames(filenames: Sequence[str]):
        if not filenames:
            return False
        if isinstance(filenames, str):
            filename = filenames
            filenames = (filenames,)
        else:
            filename = filenames[0]
        for ext in exts:
            if filename.endswith(ext):
                break
        else:
            return False
        return all(filename.endswith(ext) for filename in filenames)

    return checkfilenames


def LoadFileWithExt(*, key: str | Mapping | None = None, update: bool = False, **kwargs: Callable):
    """Returns a function that retunts True if the file extension is consistent"""

    def checkfilename(filename_or_dict: str | Mapping):
        if key is not None:
            if not isinstance(filename_or_dict, Mapping):
                raise SchemaError("Expect dictionary as a filename")
            dct = filename_or_dict.copy()
            filename = dct.pop(key)
        else:
            if not isinstance(filename_or_dict, str):
                raise SchemaError("Expect str as a filename")
            filename, dct = filename_or_dict, None
        for ext, loader in kwargs.items():
            if filename.endswith(f".{ext}"):
                break
        else:
            raise SchemaError(
                f"Do not know how to load {filename}: no extension handler provided"
                f" ({', '.join(kwargs.keys())})"
            )

        ret = loader(filename)
        if update and dct is not None:
            ret.update(dct)

        return ret

    return checkfilename


from pathlib import Path

from yaml import Loader, load


def LoadYaml(fname: Path | str):
    if isinstance(fname, Path):
        fname = str(fname)
    with open(fname) as file:
        ret = load(file, Loader)

    logger.log(INFO1, f"Read: {fname}")
    return ret


import runpy


def LoadPy(fname: Path | str, variable: str, *, type: type | None = None):
    if isinstance(fname, Path):
        fname = str(fname)
    logger.log(INFO1, f"Read: {fname} ({variable})")
    dct = runpy.run_path(fname)

    try:
        ret = dct[variable]
    except KeyError as e:
        raise RuntimeError(f"Variable {variable} is not provided in file {fname}") from e


    if type is not None and not isinstance(ret, type):
        raise RuntimeError(
            f"Variable {variable} has wrong type ({type(ret).__name__}). Expect {type.__name__}"
        )

    return ret


def MakeLoaderPy(variable: str):
    return lambda fname: LoadPy(fname, variable)


from nested_mapping import NestedMapping


class NestedSchema:
    __slots__ = ("_schema", "_processdicts")
    _schema: Schema | object
    _processdicts: bool

    def __init__(self, /, schema: Schema | object, *, processdicts: bool = False):
        self._schema = schema
        self._processdicts = processdicts

    def validate(self, data: Any) -> Any:
        if not isinstance(data, dict):
            return self._schema.validate(data)

        if self._processdicts:
            return {key: self._process_dict((key,), subdata) for key, subdata in data.items()}

        dtin = NestedMapping(data)
        dtout = NestedMapping({})
        for key, subdata in dtin.walkitems():
            dtout[key] = self._process_element(key, subdata)

        return dtout.object

    def _process_element(self, key, subdata: Any) -> Any:
        try:
            return self._schema.validate(subdata, _is_event_schema=False)
        except SchemaError as err:
            key = ".".join(str(k) for k in key)
            raise SchemaError(f'Key "{key}" has invalid value "{subdata}":\n{err.args[0]}') from err

    def _process_dict(self, key, data: Any) -> Any:
        if not isinstance(data, dict):
            return self._schema.validate(data)

        with suppress(SchemaError):
            return self._schema.validate(data, _is_event_schema=False)

        return {
            subkey: self._process_dict(key + (subkey,), subdata) for subkey, subdata in data.items()
        }
