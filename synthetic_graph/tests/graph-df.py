import numpy as np
import argparse
import timeit
from dag_modelling.core.graph import Graph
from dag_modelling.bundles.file_reader import FileReader
from dag_modelling.plot.graphviz import savegraph
from dag_modelling.lib.arithmetic import Sum
from dag_modelling.lib.common import Array
from dag_modelling.core.storage import NodeStorage


np.random.seed(33)

def make_test_graph(storage, datasize=1, width=6, length=7):
    with (
        Graph(close_on_exit=True) as graph,
        storage,
        FileReader
    ):
        nsums = 0
        prevlayer = []

        data = np.random.uniform(-100, 100, size=datasize)
        data_node, _ = Array.replicate(
            name=f"array {5}-{0}-{0}",
            array=data
        )

        for ilayer in reversed(range(length)):
            ilayer_next = ilayer - 1
            n_groups = int(width ** ilayer_next)
            thislayer = []

            for igroup in range(n_groups):
                head = Sum(name=f"sum {ilayer}-{igroup}")
                nsums += 1

                if prevlayer:
                    for array in prevlayer:
                        array >> head
                else:
                    for isource in range(width):
                        data_node >> head

                thislayer.append(head)

            prevlayer = thislayer

    savegraph(graph, "dagflow_example_0.dot", show=["type", "mark", "label", "path"],)
    return nsums, data_node, head, graph

def run_test(head, tail, runs, graph):
    def test():
        head.taint()
        tail.outputs[0].data

    average_time = timeit.timeit(test, number=runs) / runs
    savegraph(graph, "dagflow_example_1.dot", show=["type", "mark", "label", "path"],)
    print(f"Среднее время выполнения за {runs} прогонов: {average_time:.5e} секунд")
    savegraph(graph, "dagflow_example_2.dot", show=["type", "mark", "label", "path"],)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Тестирование графа.")
    parser.add_argument('--width', type=int, default=6, help='Ширина графа.')
    parser.add_argument('--length', type=int, default=5, help='Длина графа.')
    parser.add_argument('--dsize', type=int, default=1, help='Размер данных.')
    parser.add_argument('--runs', type=int, default=10, help='Количество прогонов для замера времени.')
    args = parser.parse_args()

    storage = NodeStorage()
    nsums, data_node, head, graph = make_test_graph(storage, datasize=args.dsize, width=args.width, length=args.length)
    print(f"Создано узлов: {nsums}")

    run_test(data_node, head, args.runs, graph)
