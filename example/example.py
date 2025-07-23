from numpy import arange
from numpy import copyto
from numpy import result_type

from dag_modelling.core.graph import Graph
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.core.input_strategy import AddNewInputAddNewOutputForBlock
from dag_modelling.lib.common import Array
from dag_modelling.lib.arithmetic import Product
from dag_modelling.lib.arithmetic import Sum
from dag_modelling.lib.summation import WeightedSum
from dag_modelling.core.node import Node

array = arange(3, dtype="d")
debug = False


class ThreeInputsOneOutput(Node):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("input_strategy", AddNewInputAddNewOutputForBlock())
        super().__init__(*args, **kwargs)

    def _function(self, _, inputs, outputs):
        for i, output in enumerate(outputs):
            out = output.data
            copyto(out, inputs[3 * i].data)
            for input in inputs[3 * i + 1 : (i + 1) * 3]:
                out += input.data
        return out

    @property
    def result(self):
        return [out.data for out in self.outputs]

    def _type_function(self) -> None:
        """A output takes this function to determine the dtype and shape"""
        for i, output in enumerate(self.outputs):
            inputs = self.inputs[2 * i : 2 * (1 + i)]
            output.dd.shape = inputs[0].dd.shape
            output.dd.dtype = result_type(tuple(inp.dd.dtype for inp in inputs))
        self.logger.debug(
            f"Node '{self.name}': dtype={tuple(out.dd.dtype for out in self.outputs)}, "
            f"shape={tuple(out.dd.shape for out in self.outputs)}"
        )


# Check predefined Array, Sum and Product
with Graph(debug=debug) as graph:
    (in1, in2, in3, in4) = (Array(name, array) for name in ("n1", "n2", "n3", "n4"))
    s = Sum("sum")
    m = Product("product")

    (in1, in2, in3) >> s
    (in4, s) >> m
    graph.close()

    print("Result:", m.outputs["result"].data)
    savegraph(graph, "dag_modelling_example_1a.png")

# Check random generated Array, Sum and Product
with Graph(debug=debug) as graph:
    (in1, in2, in3, in4) = (Array(name, array) for name in ("n1", "n2", "n3", "n4"))
    s = Sum("sum")
    m = Product("product")

    (in1, in2, in3) >> s
    (in4, s) >> m
    graph.close()

    print("Result:", m.outputs["result"].data)
    savegraph(graph, "dag_modelling_example_1b.png")

# Check predefined Array, two Sum's and Product
with Graph(debug=debug) as graph:
    (in1, in2, in3, in4) = (Array(name, array) for name in ("n1", "n2", "n3", "n4"))
    s = Sum("sum")
    s2 = Sum("sum")
    m = Product("product")

    (in1, in2) >> s
    (in3, in4) >> s2
    (s, s2) >> m
    graph.close()

    print("Result:", m.outputs["result"].data)
    savegraph(graph, "dag_modelling_example_2.png")

# Check predefined Array, Sum, WeightedSum and Product
with Graph(debug=debug) as graph:
    (in1, in2, in3, in4) = (Array(name, array) for name in ("n1", "n2", "n3", "n4"))
    weight = Array("weight", (2, 3))
    # The same result with other weight
    # weight = makeArray(5)("weight")
    s = Sum("sum")
    ws = WeightedSum("weightedsum")
    m = Product("product")

    (in1, in2) >> s  # [0,2,4]
    (in3, in4) >> ws
    {"weight": weight} >> ws  # [0,1,2] * 2 + [0,1,2] * 3 = [0,5,10]
    # NOTE: also it is possible to use the old style binding:
    # weight >> ws("weight")
    (s, ws) >> m  # [0,2,4] * [0,5,10] = [0,10,40]
    graph.close()

    print("Result:", m.outputs["result"].data)
    savegraph(graph, "dag_modelling_example_3.png")


with Graph(debug=debug) as graph:
    (in1, in2, in3) = (Array(name, array) for name in ("n1", "n2", "n3"))
    (in4, in5, in6) = (Array(name, (1, 0, 0)) for name in ("n4", "n5", "n6"))
    (in7, in8, in9) = (Array(name, (3, 3, 3)) for name in ("n7", "n8", "n9"))
    s = ThreeInputsOneOutput("3to1")
    (in1, in2, in3) >> s
    (in4, in5, in6) >> s
    (in7, in8, in9) >> s
    graph.close()

    print("Result:", s.result)
    savegraph(graph, "dag_modelling_example_4.png")
