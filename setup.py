"""Build script for Cython extensions in dag-modelling.

This setup.py is required because setuptools' pyproject.toml-only build
does not natively support .pyx compilation. The Cython.Build.cythonize
call discovers .pyx files and generates corresponding C sources.

If Cython is not installed, the build falls back to pure Python
(the .py file is used instead of the compiled extension).
"""
from setuptools import setup

try:
    from Cython.Build import cythonize

    ext_modules = cythonize(
        [
            "src/dag_modelling/core/node_eval.pyx",
            "src/dag_modelling/core/input.pyx",
            "src/dag_modelling/core/output.pyx",
            "src/dag_modelling/core/node_base.pyx",
            "src/dag_modelling/core/node.pyx",
        ],
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
        },
    )
except ImportError:
    # Cython is an opt-in build dependency (the "cython" extra), not part of
    # [build-system].requires, so a plain install lands here and stays pure Python.
    print("dag-modelling: Cython not available, building pure Python (no .so).")
    ext_modules = []

setup(ext_modules=ext_modules)
