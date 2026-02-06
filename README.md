# Summary

[![python](https://img.shields.io/badge/python-3.11-purple.svg)](https://www.python.org/)
[![pipeline](https://git.jinr.ru/dagflow-team/dag-modelling/badges/master/pipeline.svg)](https://git.jinr.ru/dagflow-team/dag-modelling/commits/master)
[![coverage report](https://git.jinr.ru/dagflow-team/dag-modelling/badges/master/coverage.svg)](https://git.jinr.ru/dagflow-team/dag-modelling/-/commits/master)
[![github](https://img.shields.io/badge/github-public-blue?logo=github)](https://github.com/dagflow-team/dag-modelling)
[![gitlab](https://img.shields.io/badge/gitlab-dev-blue?logo=gitlab)](https://git.jinr.ru/dagflow-team/dag-modelling)
[![pypi-release](https://img.shields.io/badge/pypi-release-blue?logo=pypi&logoColor=green)](https://pypi.org/project/dag-modelling)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The **DAGModelling** software is a python implementation of the dataflow programming with the lazy graph evaluation approach.

Main goals:
*  Lazy evaluated directed acyclic graph;
*  Concise connection syntax;
*  Plotting with graphviz;
*  Flexibility. The goal of DAG-Modelling is not to be efficient, but rather flexible.

The framework is intended to be used for the statistical analysis of the data of *JUNO* and *Daya Bay* neutrino oscillation experiments.

## Installation

### For users (*recommended*)

For regular use, it's best to install [the latest version of the project that's available on PyPi](https://pypi.org/project/dag-modelling/):
```bash
pip install dag-modelling
```

### For developers

We recommend that developers install the package locally in editable mode:
```bash
git clone https://github.com/dagflow-team/dag-modelling.git
cd dag-modelling
pip install -e .
```
This way, the system will track all the changes made to the source files. This means that developers won't need to reinstall the package or set environment variables, even when a branch is changed.

## Example

For example, let's consider a sum of three input nodes and then a product of the result with another array.

```python
from numpy import arange

from dag_modelling.core.graph import Graph
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.lib.common import Array
from dag_modelling.lib.arithmetic import Sum, Product

# Define a source data
array = arange(3, dtype="d")

# Check predefined Array, Sum and Product
with Graph(debug=debug) as graph:
    # Define nodes
    (in1, in2, in3, in4) = (Array(name, array) for name in ("n1", "n2", "n3", "n4"))
    s = Sum("sum")
    m = Product("product")

    # Connect nodes
    (in1, in2, in3) >> s
    (in4, s) >> m
    graph.close()

    print("Result:", m.outputs["result"].data) # must print [0. 3. 12.]
    savegraph(graph, "dag_modelling_example_1a.png")
```
The printed result must be `[0. 3. 12.]`, and the created image looks as

![graph example](example/dag_modelling_example_1a.png)


For more examples see [example/example.py](example/example.py) or [tests](tests).

Please, note, that examples are using `pygraphviz` package, which is optional and not requested by default.

## Additional modules

Supplementary python modules:
* [dgm-reactor-neutrino](https://github.com/dagflow-team/dgm-reactor-neutrino) — nodes related to reactor neutrino oscillations.
* [dgm-fit](https://github.com/dagflow-team/dgm-fit) — fitter interface.
* [Daya Bay model](https://github.com/dagflow-team/dgm-dayabay-dev) — implementation of the Daya Bay oscillation analysis, development version.

