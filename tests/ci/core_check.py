"""CI check: the loaded dag-modelling core must match DGM_CORE (cython or python).

DGM_CORE is set by the CI matrix, not read by the package.
"""

from importlib.machinery import EXTENSION_SUFFIXES
from os import environ

import dag_modelling.core.node_eval as node_eval

expected = environ["DGM_CORE"]
loaded = "cython" if node_eval.__file__.endswith(tuple(EXTENSION_SUFFIXES)) else "python"
print(f"DGM_CORE={expected}, loaded {loaded}: {node_eval.__file__}")
if loaded != expected:
    raise SystemExit(f"Expected {expected} core, but {loaded} is loaded")
