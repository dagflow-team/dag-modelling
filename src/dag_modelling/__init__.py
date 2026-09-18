from importlib.metadata import version
from os import environ

if environ.get("DGM_DISABLE_CYTHON", "").lower() not in {"", "0", "no", "false"}:
    from ._disable_cython import disable_cython

    disable_cython()

__version__ = version(__name__)
