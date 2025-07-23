from collections.abc import Callable, Mapping
from pathlib import Path

from numpy import asarray
from schema import And, Optional, Or, Schema, Use

from nested_mapping.tools.map import make_reorder_function
from nested_mapping.typing import strkey

from ..core.storage import NodeStorage
from ..lib.common import Array
from ..tools.logger import INFO3, logger
from ..tools.schema import (
    AllFileswithExt,
    IsFilenameSeqOrFilename,
    IsStrSeqOrStr,
    LoadFileWithExt,
    LoadYaml,
)
from .file_reader import FileReader, file_readers, iterate_filenames_and_objectnames

_schema_cfg = Schema(
    {
        "name": str,
        "filenames": And(IsFilenameSeqOrFilename, AllFileswithExt(*file_readers.keys())),
        Optional("dtype", default=None): Or("d", "f"),
        Optional("replicate_outputs", default=((),)): Or((IsStrSeqOrStr,), [IsStrSeqOrStr]),
        Optional("replicate_files", default=((),)): Or((IsStrSeqOrStr,), [IsStrSeqOrStr]),
        Optional("skip", default=None): And(
            Or(((str,),), [[str]]), Use(lambda l: tuple(set(k) for k in l))
        ),
        Optional("key_order", default=None): Or(
            ((str,), (str,)),
            [[str], [str]],
            (int,),
            [int],
        ),
        Optional("output_key_order", default=None): Or(
            ((str,), (str,)),
            [[str], [str]],
            (int,),
            [int],
        ),
        Optional("name_function", default=lambda: lambda st, tpl: st): Or(
            Callable, And({str: str}, Use(lambda dct: lambda st, tpl: dct.get(st, st)))
        ),
    }
)

_schema_loadable_cfg = And(
    {"load": Or(str, And(Path, Use(str))), Optional(str): object},
    Use(
        LoadFileWithExt(yaml=LoadYaml, key="load", update=True),
        error="Failed to load {}",
    ),
    _schema_cfg,
)


def _validate_cfg(cfg):
    if isinstance(cfg, dict) and "load" in cfg:
        return _schema_loadable_cfg.validate(cfg)
    else:
        return _schema_cfg.validate(cfg)


def load_array(acfg: Mapping | None = None, *, array_kwargs: Mapping = {}, **kwargs) -> NodeStorage:
    acfg = dict(acfg or {}, **kwargs)
    cfg = _validate_cfg(acfg)

    name = (cfg["name"],)
    filenames = cfg["filenames"]
    keys = cfg["replicate_outputs"]
    file_keys = cfg["replicate_files"]
    name_function = cfg["name_function"]
    skip = cfg["skip"]
    key_order = cfg["key_order"]
    output_key_order = cfg["output_key_order"]
    dtype = cfg["dtype"]

    data = {}
    reorder_output_key = make_reorder_function(output_key_order)
    for _, filename, _, key in iterate_filenames_and_objectnames(
        filenames, file_keys, keys, skip=skip, key_order=key_order
    ):
        skey = strkey(key)
        logger.log(INFO3, f"Process {skey}")

        array = FileReader.array[filename, name_function(skey, key)]
        output_key = reorder_output_key(key)
        data[output_key] = asarray(array, dtype)

    storage = NodeStorage(default_containers=True)
    with storage:
        for key, array in data.items():
            Array.replicate(name=".".join(name + key), array=array, **array_kwargs)

    NodeStorage.update_current(storage, strict=True)

    return storage
