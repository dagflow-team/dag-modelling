from importlib.machinery import EXTENSION_SUFFIXES
from json import loads
from os import environ
from pathlib import Path
from subprocess import run
from sys import executable

from pytest import mark

import dag_modelling

# Compiled in setup.py
COMPILED_MODULES = ("node_eval", "input", "output", "node_base", "node")
CORE_DIR = Path(dag_modelling.__file__).parent / "core"

PROBE_TEMPLATE = f"""
from importlib import import_module
from json import dumps

from dag_modelling.core.graph import Graph
from dag_modelling.lib.arithmetic import Sum
from dag_modelling.lib.common import Array

with Graph(close_on_exit=True):
    a, b = Array("a", [1.0, 2.0]), Array("b", [3.0, 4.0])
    s = Sum("sum")
    (a, b) >> s

modules = {COMPILED_MODULES}
files = {{m: import_module("dag_modelling.core." + m).__file__ for m in modules}}
print(dumps({{"files": files, "result": s.outputs[0].data.tolist()}}))
"""


def run_probe(value: str) -> dict:
    env = dict(environ, DGM_DISABLE_CYTHON=value)
    out = run(
        [executable, "-c", PROBE_TEMPLATE], env=env, capture_output=True, text=True, check=True
    )
    return loads(out.stdout.splitlines()[-1])


def is_built(module: str) -> bool:
    return any((CORE_DIR / f"{module}{suffix}").exists() for suffix in EXTENSION_SUFFIXES)


@mark.parametrize("value", ["1", "yes", "TRUE"])
def test_pure_python_switch_on(value):
    res = run_probe(value)
    assert all(f.endswith(".py") for f in res["files"].values()), res["files"]
    assert res["result"] == [4.0, 6.0]


@mark.parametrize("value", ["", "0", "no", "False"])
def test_pure_python_switch_off(value):
    res = run_probe(value)
    for module, file in res["files"].items():
        assert file.endswith(".py") != is_built(module), res["files"]
    assert res["result"] == [4.0, 6.0]
