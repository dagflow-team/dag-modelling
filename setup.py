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
    # Build isolation hides Cython, even with `pip install ".[cython]"`.
    print("dag-modelling: no Cython, building pure Python. To compile, use --no-build-isolation.")
    ext_modules = []

setup(ext_modules=ext_modules)
