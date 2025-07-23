# Summary

[![python](https://img.shields.io/badge/python-3.10-purple.svg)](https://www.python.org/)
[![pipeline](https://git.jinr.ru/dag-computing/dag-flow/badges/master/pipeline.svg)](https://git.jinr.ru/dag-computing/dag-flow/commits/master)
[![coverage report](https://git.jinr.ru/dag-computing/dag-flow/badges/master/coverage.svg)](https://git.jinr.ru/dag-computing/dag-flow/-/commits/master)
<!--- Uncomment here after adding docs!
[![pages](https://img.shields.io/badge/pages-link-white.svg)](http://dag-computing.pages.jinr.ru/dag-flow)
-->

The **DAGModelling** software is a python implementation of the dataflow programming with the lazy graph evaluation approach.

Main goals:
*  Lazy evaluated directed acyclic graph;
*  Concise connection syntax;
*  Plotting with graphviz;
*  Flexibility. The goal of DAG-Modelling is not to be efficient, but rather flexible.

The framework is intented to be used for the statistical analysis of the data of *JUNO* and *Daya Bay* neutrino oscillation experiments.

## Installation

### For users (*recommended*)

For regular use, it's best to install [the latest version of the project that's available on PyPi](https://pypi.org/project/dag-modelling/):
```bash
pip install dag-modelling
```

### For developers

We recommend that developers install the package locally in editable mode:
```bash
git clone https://github.com/dagmodelling-team/dag-modelling.git
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
    savegraph(graph, "dagmodelling_example_1a.png")
```
The printed result must be `[0. 3. 12.]`, and the created image looks as
![](https://raw.githubusercontent.com/dagmodelling-team/dag-modelling/refs/heads/0.9.0/example/dagmodelling_example_1a.png)


For more examples see [example/example.py](https://github.com/dagmodelling-team/dag-modelling/blob/master/example/example.py) or [tests](https://github.com/dagmodelling-team/dag-modelling/tree/master/tests).

## Additional modules

- Supplementary python modules:
    * [dagmodelling-reactornueosc](https://git.jinr.ru/dag-computing/dagmodelling-reactorenueosc) — nodes related to reactor neutrino oscillations
    * [dagmodelling-detector](https://git.jinr.ru/dag-computing/dagmodelling-detector) — nodes for the detector responce modelling
    * [dagmodelling-statistics](https://git.jinr.ru/dag-computing/dagmodelling-statistics) — statistical analysis and MC
- [Daya Bay model](https://git.jinr.ru/dag-computing/dayabay-model) — test implementation of the Daya Bay oscillation analysis

