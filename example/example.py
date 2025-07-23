#!/usr/bin/env python

from numpy import arange, copyto, result_type

from dag_modelling.core.graph import Graph
from dag_modelling.core.input_strategy import AddNewInputAddNewOutputForBlock
from dag_modelling.core.node import Node
from dag_modelling.lib.arithmetic import Product, Sum
from dag_modelling.lib.common import Array
from dag_modelling.lib.summation import WeightedSum
from dag_modelling.plot.graphviz import savegraph

array = arange(3, dtype="d")
debug = False


# Check predefined Array, Sum and Product
with Graph(debug=debug, close_on_exit=True) as graph:
    (in1, in2, in3, in4) = (Array(name, array) for name in ("n1", "n2", "n3", "n4"))
    s = Sum("sum")
    m = Product("product")

    (in1, in2, in3) >> s
    (in4, s) >> m

print("Result:", m.outputs["result"].data)
filename = "dag_modelling_example_1a.png"
savegraph(graph, filename)
print(f"Write figure: {filename}")

# Check random generated Array, Sum and Product
with Graph(debug=debug, close_on_exit=True) as graph:
    (in1, in2, in3, in4) = (Array(name, array) for name in ("n1", "n2", "n3", "n4"))
    s = Sum("sum")
    m = Product("product")

    (in1, in2, in3) >> s
    (in4, s) >> m

print("Result:", m.outputs["result"].data)
filename = "dag_modelling_example_1b.png"
savegraph(graph, filename)
print(f"Write figure: {filename}")

# Check predefined Array, two Sum's and Product
with Graph(debug=debug, close_on_exit=True) as graph:
    (in1, in2, in3, in4) = (Array(name, array) for name in ("n1", "n2", "n3", "n4"))
    s = Sum("sum")
    s2 = Sum("sum")
    m = Product("product")

    (in1, in2) >> s
    (in3, in4) >> s2
    (s, s2) >> m

print("Result:", m.outputs["result"].data)
filename = "dag_modelling_example_1b.png"
savegraph(graph, filename)
print(f"Write figure: {filename}")

# Check predefined Array, Sum, WeightedSum and Product
with Graph(debug=debug, close_on_exit=True) as graph:
    (in1, in2, in3, in4) = (Array(name, array) for name in ("n1", "n2", "n3", "n4"))
    weight = Array("weight", (2, 3))
    # The same result with other weight
    # weight = makeArray(5)("weight")
    s = Sum("sum")
    ws = WeightedSum("weightedsum")
    m = Product("product")

    (in1, in2) >> s  # [0,2,4]
    (in3, in4) >> ws
    weight >> ws("weight")
    (s, ws) >> m  # [0,2,4] * [0,5,10] = [0,10,40]

print("Result:", m.outputs["result"].data)
filename = "dag_modelling_example_1b.png"
savegraph(graph, filename)
print(f"Write figure: {filename}")


# Create a custom node
class ThreeInputsOneOutput(Node):
    """The node sums every three inputs into a new output."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("input_strategy", AddNewInputAddNewOutputForBlock())
        super().__init__(*args, **kwargs)

    def _function(self):
        for i, output in enumerate(self.outputs):
            out = output._data
            copyto(out, self.inputs[3 * i].data)
            for input in self.inputs[3 * i + 1 : (i + 1) * 3]:
                out += input.data
        return out

    @property
    def result(self):
        return [out.data for out in self.outputs]

    def _type_function(self) -> None:
        """A output takes this function to determine the dtype and shape."""
        for i, output in enumerate(self.outputs):
            inputs = self.inputs[2 * i : 2 * (1 + i)]
            output.dd.shape = inputs[0].dd.shape
            output.dd.dtype = result_type(tuple(inp.dd.dtype for inp in inputs))
        self.logger.debug(
            f"Node '{self.name}': dtype={tuple(out.dd.dtype for out in self.outputs)}, "
            f"shape={tuple(out.dd.shape for out in self.outputs)}"
        )


with Graph(debug=debug, close_on_exit=True) as graph:
    (in1, in2, in3) = (Array(name, array) for name in ("n1", "n2", "n3"))
    (in4, in5, in6) = (Array(name, (1, 0, 0)) for name in ("n4", "n5", "n6"))
    (in7, in8, in9) = (Array(name, (3, 3, 3)) for name in ("n7", "n8", "n9"))
    s = ThreeInputsOneOutput("3to1")
    (in1, in2, in3) >> s
    (in4, in5, in6) >> s
    (in7, in8, in9) >> s

print("Result:", s.result)
filename = "dag_modelling_example_1b.png"
savegraph(graph, filename)
