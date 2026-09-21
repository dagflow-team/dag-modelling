"""Load dag_modelling.core from .py even if .so files are built (DGM_DISABLE_CYTHON=1).

Works at import time only: already imported modules stay as they are.
"""

from importlib.machinery import SOURCE_SUFFIXES, FileFinder, SourceFileLoader
from sys import meta_path

_PREFIX = "dag_modelling.core."


class PurePythonFinder:
    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if path is None or not fullname.startswith(_PREFIX):
            return None
        for entry in path:
            spec = FileFinder(entry, (SourceFileLoader, SOURCE_SUFFIXES)).find_spec(fullname)
            if spec is not None:
                return spec
        return None


def disable_cython() -> None:
    if PurePythonFinder not in meta_path:
        meta_path.insert(0, PurePythonFinder)
