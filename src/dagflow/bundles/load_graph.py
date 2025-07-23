from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from numpy import allclose, asarray
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

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from nested_mapping.typing import TupleKey

_schema_cfg = Schema(
    {
        "name": str,
        "filenames": And(IsFilenameSeqOrFilename, AllFileswithExt(*file_readers.keys())),
        Optional("merge_x", default=False): bool,
        Optional("x", default="x"): str,
        Optional("y", default="y"): str,
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


def _load_graph_data(
    acfg: Mapping | None = None, **kwargs
) -> tuple[tuple, tuple, NDArray | None, dict[TupleKey, tuple[NDArray, NDArray]]]:
    acfg = dict(acfg or {}, **kwargs)
    cfg = _validate_cfg(acfg)

    name = cfg["name"]
    filenames = cfg["filenames"]
    keys = cfg["replicate_outputs"]
    file_keys = cfg["replicate_files"]
    name_function = cfg["name_function"]
    skip = cfg["skip"]
    key_order = cfg["key_order"]
    output_key_order = cfg["output_key_order"]
    dtype = cfg["dtype"]

    xname = name, cfg["x"]
    yname = name, cfg["y"]

    meshes_list: list[NDArray] = []
    data: dict[TupleKey, tuple[NDArray, NDArray]] = {}
    reorder_output_key = make_reorder_function(output_key_order)
    for _, filename, _, key in iterate_filenames_and_objectnames(
        filenames, file_keys, keys, skip=skip, key_order=key_order
    ):
        skey = strkey(key)
        logger.log(INFO3, f"Process {skey}")

        x, y = FileReader.graph[filename, name_function(skey, key)]
        x = asarray(x, dtype)
        y = asarray(y, dtype)

        output_key = reorder_output_key(key)
        data[output_key] = x, y
        meshes_list.append(x)

    if cfg["merge_x"]:
        x0 = meshes_list[0]
        for xi in meshes_list[1:]:
            if not allclose(x0, xi, atol=0, rtol=0):
                raise RuntimeError("load_graph: inconsistent x axes, unable to merge.")

        return xname, yname, x0, data

    return xname, yname, None, data


def load_graph(acfg: Mapping | None = None, **kwargs) -> NodeStorage:
    xname, yname, mesh_common, data = _load_graph_data(acfg, **kwargs)

    storage = NodeStorage(default_containers=True)
    with storage:
        if mesh_common is not None:
            mesh, _ = Array.replicate(name=strkey(xname), array=mesh_common)
        else:
            mesh = None

        for key, (x, y) in data.items():
            if mesh_common is None:
                xkey = strkey(xname + key)
                mesh, _ = Array.replicate(name=xkey, array=x)
            ykey = strkey(yname + key)
            Array.replicate(name=ykey, array=y, meshes=mesh)

    NodeStorage.update_current(storage, strict=True)

    return storage


def load_graph_data(acfg: Mapping | None = None, **kwargs) -> NodeStorage:
    xname, yname, mesh_common, data = _load_graph_data(acfg, **kwargs)

    storage = NodeStorage(default_containers=True)
    data_storage = storage("data")
    if mesh_common is None:
        for key, (x, y) in data.items():
            data_storage[xname + key] = x
    else:
        data_storage[xname] = mesh_common

    for key, (x, y) in data.items():
        data_storage[yname + key] = y

    NodeStorage.update_current(storage, strict=True)

    return storage
