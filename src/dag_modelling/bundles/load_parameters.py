from __future__ import annotations

from contextlib import suppress
from math import fabs
from pathlib import Path
from typing import TYPE_CHECKING

from schema import And, Optional, Or, Schema, SchemaError, Use

from nested_mapping import NestedMapping
from nested_mapping.typing import properkey

from ..core.exception import InitializationError
from ..core.labels import format_dict, inherit_labels, mapping_append_lists
from ..lib.common import Array
from ..lib.summation import ElSumSq
from ..parameters import Parameters
from ..core.storage import NodeStorage
from ..tools.schema import IsStrSeqOrStr, LoadFileWithExt, LoadYaml, MakeLoaderPy, NestedSchema

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping, Sequence


class ParsCfgHasProperFormat:
    __slots__ = ()

    def validate(self, data: dict) -> dict:
        form = data["format"]
        nelements = 1 if isinstance(form, str) else len(form)

        dtin = NestedMapping(data)
        for key, subdata in dtin("parameters").walkitems():
            if isinstance(subdata, tuple):
                if len(subdata) == nelements:
                    continue
            elif nelements == 1:
                continue

            key = ".".join(str(k) for k in key)
            raise SchemaError(
                f'Key "{key}" has  value "{subdata}"" inconsistent with format "{form}"'
            )

        return data


IsNumber = Or(float, int, error='Invalid number "{}", expect int of float')
IsNumberOrTuple = Or(
    IsNumber, (IsNumber,), And([IsNumber], Use(tuple)), error="Invalid number/tuple {}"
)
label_keys = {"text", "latex", "graph", "mark", "name", "index_values", "node_hidden"}
IsLabel = Or(
    {
        "text": str,
        Optional("latex"): str,
        Optional("graph"): str,
        Optional("mark"): str,
        Optional("name"): str,
        Optional("node_hidden"): bool,
    },
    And(str, Use(lambda s: {"text": s}), error="Invalid string: {}"),
)
IsValuesDict = NestedSchema(IsNumberOrTuple)
IsLabelsDict = NestedSchema(IsLabel, processdicts=True)


def IsFormatOk(format):
    if not isinstance(format, (tuple, list)):
        return format == "value"

    match format:
        case "value" | ["value"]:
            return True
        case [*valcent, "sigma_absolute" | "sigma_relative" | "sigma_percent"]:
            match valcent:
                case ["value" | "central"] | ["value", "central"] | ["central", "value"]:
                    return True

    return False


def CheckCorrelationSizes(cfg):
    nnames = len(cfg["names"])
    matrix = cfg["matrix"]
    nrows = len(matrix)

    return False if nrows != nnames else all(nnames == len(row) for row in matrix)


IsCorrelationsDict = And(
    {
        "names": Or((str,), And([str], Use(tuple))),
        "matrix_type": Or("correlation", "covariance"),
        "matrix": [[IsNumber]],
    },
    CheckCorrelationSizes,
)
IsNestedCorrelationsDict = NestedSchema(IsCorrelationsDict, processdicts=True)

IsFormat = Schema(IsFormatOk, error='Invalid parameter format "{}".')
IsParsCfgDict = Schema(
    {
        "parameters": IsValuesDict,
        "labels": IsLabelsDict,
        "format": IsFormat,
        "state": Or("fixed", "variable", error="Invalid parameters state: {}"),
        Optional("path", default=""): str,
        Optional("replicate", default=((),)): (IsStrSeqOrStr,),
        Optional("keys_order", default=None): Or([[str], [str]], ((str,), (str,))),
        Optional("correlations", default={}): IsNestedCorrelationsDict,
        Optional("joint_nuisance", default=False): bool,
        Optional("sigma_visible", default=False): bool,
        Optional("ignore_keys", default=()): Or(
            ({str},),
            And(Or((str,),),[(str,)],[{str}]),
            Use(lambda keys: tuple(map(set, keys)))
        ),
    },
    # error = 'Invalid parameters configuration: {}'
)
IsProperParsCfgDict = And(IsParsCfgDict, ParsCfgHasProperFormat())
IsLoadableDict = And(
    {"load": Or(str, And(Path, Use(str))), Optional(str): object},
    Use(
        LoadFileWithExt(yaml=LoadYaml, py=MakeLoaderPy("configuration"), key="load", update=True),
        error="Failed to load {}",
    ),
    IsProperParsCfgDict,
)


def ValidateParsCfg(cfg):
    if isinstance(cfg, dict) and "load" in cfg:
        return IsLoadableDict.validate(cfg)
    else:
        return IsProperParsCfgDict.validate(cfg)


def process_var_fixed1(vcfg, _, __):
    return {"central": vcfg, "value": vcfg, "sigma": None}


def process_var_fixed2(vcfg, format, hascentral) -> dict:
    ret = dict(zip(format, vcfg))
    if hascentral:
        ret.setdefault("value", ret["central"])
    else:
        ret.setdefault("central", ret["value"])
    ret["sigma"] = None
    return ret


def process_var_absolute(vcfg, format, hascentral) -> dict:
    ret = process_var_fixed2(vcfg, format, hascentral)
    ret["sigma"] = ret["sigma_absolute"]
    return ret


def process_var_relative(vcfg, format, hascentral) -> dict:
    ret = process_var_fixed2(vcfg, format, hascentral)
    ret["sigma"] = ret["sigma_relative"] * fabs(ret["central"])
    return ret


def process_var_percent(vcfg, format, hascentral) -> dict:
    ret = process_var_fixed2(vcfg, format, hascentral)
    ret["sigma"] = 0.01 * ret["sigma_percent"] * fabs(ret["central"])
    return ret


def get_format_processor(format):
    if isinstance(format, str):
        return process_var_fixed1

    errfmt = format[-1]
    if not errfmt.startswith("sigma"):
        return process_var_fixed2

    if errfmt.endswith("_absolute"):
        return process_var_absolute
    elif errfmt.endswith("_relative"):
        return process_var_relative
    else:
        return process_var_percent


def get_label(key: tuple, labelscfg: dict | NestedMapping, *, group_only: bool = False) -> dict:
    if not group_only:
        try:
            ret = labelscfg.get_any(key)
        except KeyError:
            pass
        else:
            if isinstance(ret, NestedMapping):
                ret = ret.object
            return dict(ret)

    lcfg = None
    for n in range(1, len(key) + 1):
        subkey = key[:-n]
        try:
            lcfg = labelscfg.get_dict(subkey + ("group",))
        except KeyError:
            if not group_only:
                try:
                    lcfg = labelscfg.get_any(subkey)
                except KeyError:
                    continue

        if not subkey and (lcfg is None or "text" not in lcfg):
            return {}

        rightkey = key[n - 1 :]
        key_str = ".".join(rightkey)
        ret = format_dict(
            lcfg,
            key_str,
            key=key_str,
            index=rightkey,
            space_key=f" {key_str}",
            key_space=f"{key_str} ",
            process_keys=label_keys,
        )
        return ret

    return {}


def _add_paths_from_labels(paths: list, cfg: NestedMapping):
    for _, cfg_label in cfg.walkdicts(
        ignorekeys=("value", "central", "sigma", "sigma_percent", "variable")
    ):
        paths.extend(cfg_label.get("paths"))


def iterate_varcfgs(
    cfg: NestedMapping,
) -> Generator[tuple[tuple[str, ...], NestedMapping], None, None]:
    parameterscfg = cfg.get_dict("parameters")
    labelscfg = cfg.get_dict("labels")
    form = cfg["format"]

    hascentral = "central" in form
    process = get_format_processor(form)

    for key, varcfg in parameterscfg.walkitems():
        varcfg = process(varcfg, form, hascentral)
        varcfg["label"] = get_label(key, labelscfg)
        yield key, varcfg


def check_correlations_consistent(cfg: NestedMapping) -> None:
    parscfg = cfg.get_dict("parameters")
    for key, corrcfg in cfg.get_dict("correlations").walkdicts():
        # processed_cfgs.add(varcfg)
        names = corrcfg["names"]
        try:
            parcfg = parscfg(key)
        except KeyError as e:
            raise InitializationError(f"Failed to obtain parameters for {key}") from e

        inames = tuple(parcfg.walkjoinedkeys())
        if names != inames:
            raise InitializationError(
                f'Keys in {".".join(key)} are not consistent with names: {inames} and {names}'
            )


def load_parameters(
    acfg: Mapping | None = None,
    *,
    nuisance_location: str | Sequence[str] | None = "statistic.nuisance.parts",
    **kwargs,
) -> NestedMapping:
    acfg = dict(acfg or {}, **kwargs)
    cfg = ValidateParsCfg(acfg)
    cfg = NestedMapping(cfg)

    return _load_parameters(cfg, nuisance_location=nuisance_location)


def _load_parameters(
    cfg: NestedMapping,
    *,
    nuisance_location: str | Sequence[str] | None = "statistic.nuisance.parts",
) -> NestedMapping:
    pathstr = cfg["path"]
    path = tuple(pathstr.split(".")) if pathstr else ()

    state = cfg["state"]

    ret = NestedMapping(
        {
            "parameters": {
                "all": {},
                "constant": {},
                "variable": {},
                "free": {},
                "central": {},
                "constrained": {},
                "normalized": {},
            },
            "correlations": {},
            "parameter_groups": {
                "all": {},
                "constant": {},
                "variable": {},
                "free": {},
                "constrained": {},
            },
        },
        sep=".",
    )

    check_correlations_consistent(cfg)

    subkeys = cfg["replicate"]
    from nested_mapping.tools.map import make_reorder_function

    reorder_key = make_reorder_function(cfg["keys_order"])

    varcfgs = NestedMapping({})
    normpars = {}
    for key_general, varcfg in iterate_varcfgs(cfg):
        varcfg.setdefault(state, True)

        label_general = NestedMapping(varcfg["label"])

        for subkey in subkeys:
            key = reorder_key(key_general + subkey)
            key_str = ".".join(key)
            subkey_str = ".".join(subkey)

            label_local = label_general.object
            if subkey:
                with suppress(KeyError):
                    label_local = label_general(subkey).object

            label = format_dict(
                label_local.copy(),
                index=subkey,
                key=subkey_str,
                space_key=f" {subkey_str}",
                key_space=f"{subkey_str} ",
                process_keys=label_keys,
            )
            varcfg_sub = varcfg.copy()
            varcfg_sub["label"] = label
            pname = ".".join(path + (key_str,))
            label["name"] = pname
            label["paths"] = [pname]
            label["index_values"] = key
            label.setdefault("text", key_str)

            varcfgs[key] = varcfg_sub

    processed_cfgs = set()
    pars = NestedMapping({})
    for key, corrcfg in cfg.get_dict("correlations").walkdicts():
        label = get_label(key, cfg.get_dict("labels"))
        label_mat0 = get_label(key, cfg.get_dict("labels"), group_only=True)

        matrixtype = corrcfg["matrix_type"]
        matrix = corrcfg["matrix"]
        mark_matrix = "C" if matrixtype == "correlation" else "V"
        label_mat = inherit_labels(
            label_mat0,
            fmtlong=f"{matrixtype.capitalize()} matrix: {{}}",
            fmtshort=mark_matrix + "({})",
        )
        label_mat["mark"] = mark_matrix
        label_mat = format_dict(
            label_mat,
            key="",
            index=(),
            space_key="",
            key_space="",
            process_keys=label_keys,
        )
        matrix_array = Array(matrixtype, matrix, label=label_mat)

        for subkey in subkeys:
            fullkey = key + subkey
            subkey_str = ".".join(subkey)
            try:
                varcfg = varcfgs(fullkey)
            except KeyError as e:
                raise InitializationError(f"Failed to obtain parameters for {fullkey}") from e

            kwargs = {matrixtype: matrix_array}
            kwargs["value"] = (vvalue := [])
            kwargs["central"] = (vcentral := [])
            kwargs["sigma"] = (vsigma := [])
            kwargs["names"] = (names := [])
            paths = []
            for name, vcfg in varcfg.walkdicts(ignorekeys=("label",)):
                vvalue.append(vcfg["value"])
                vcentral.append(vcfg["central"])
                vsigma.append(vcfg["sigma"])
                names.append(name)
                processed_cfgs.add(fullkey + name)
            _add_paths_from_labels(paths, varcfg)

            labelsub = format_dict(
                dict(label, name=".".join(fullkey)),
                subkey=subkey_str,
                index=subkey,
                key=subkey_str,
                space_key=f" {subkey_str}",
                key_space=f"{subkey_str} ",
                process_keys=label_keys,
            )
            mapping_append_lists(labelsub, "index_values", subkey)
            mapping_append_lists(labelsub, "paths", paths)
            pars[fullkey] = Parameters.from_numbers(label=labelsub, **kwargs)

    ignore_keys = cfg["ignore_keys"]
    def skip_key(key):
        if key in processed_cfgs:
            return True
        for ignored_key in ignore_keys:
            if ignored_key.issubset(key):
                return True
        return False
    for key, varcfg in varcfgs.walkdicts(ignorekeys=("label",)):
        if skip_key(key):
            continue
        par = Parameters.from_numbers(**varcfg.object)
        pars[key] = par

    sigma_visible = cfg["sigma_visible"]
    for key, par in pars.walkitems():
        pathkey = path + key
        if par.is_fixed:
            targetkey = ("constant",) + pathkey
        elif par.is_constrained:
            targetkey = ("constrained",) + pathkey
        else:
            targetkey = ("free",) + pathkey

        if not par.is_fixed:
            ret[("parameters", "variable") + pathkey] = par

        ret[("parameter_group",) + targetkey] = par
        ret[("parameter_group", "all") + pathkey] = par

        ptarget = ("parameters",) + targetkey
        target_all = ("parameters", "all") + pathkey
        for subname, subpar in par.iteritems():
            ret[ptarget + subname] = subpar
            ret[target_all + subname] = subpar
            if par.is_constrained:
                ret[("parameters", "central") + pathkey] = subpar.central_output

                if sigma_visible:
                    ret[("parameters", "sigma") + pathkey] = subpar.sigma_output

        if not par.is_fixed and (constraint := par.constraint):
            normpars_i = normpars.setdefault(key[0], [])
            normpars_i.append(constraint.normvalue_final)

            ntarget = ("parameters", "normalized", path) + key
            for subname, subpar in par.iteritems_norm():
                ret[ntarget + subname] = subpar

    if state!="fixed":
        joint_nuisance = cfg["joint_nuisance"]
        nuisance_location = properkey(nuisance_location, sep=".")
        if joint_nuisance:
            ssq = ElSumSq(f"nuisance: {pathstr}")
            for outputs in normpars.values():
                outputs >> ssq
            ssq.close()
            cpath = nuisance_location + (path,)
            ret[("nodes",) + cpath] = ssq
            ret[("outputs",) + cpath] = ssq.outputs[0]
        else:
            for name, outputs in normpars.items():
                ssq = ElSumSq(f"nuisance: {pathstr}.{name}")
                outputs >> ssq
                ssq.close()
                cpath = nuisance_location + (path, name)
                ret[("nodes",) + cpath] = ssq
                ret[("outputs",) + cpath] = ssq.outputs[0]

    NodeStorage.update_current(ret, strict=True)

    return ret
