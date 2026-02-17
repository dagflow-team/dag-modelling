from __future__ import annotations

from abc import abstractmethod
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generator, Generic, TypeVar, override

from numpy import printoptions, square

from ..core.exception import UnclosedGraphError
from ..core.graph import Graph
from ..core.input import Input
from ..core.node import Node
from ..core.output import Output
from ..tools.logger import INFO1, logger

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Literal

    from numpy.typing import NDArray

try:
    import pygraphviz as G
except ImportError:
    logger.critical(
        "Unable to import `pygraphviz`. Ensure to install it with `pip install pygraphviz`."
    )
    raise


def savegraph(graph, *args, **kwargs):
    gd = GraphDot(graph, **kwargs)
    gd.savegraph(*args)


class EdgeDef:
    __slots__ = ("nodein", "nodemid", "nodeout", "edges")

    def __init__(self, nodeout, nodemid, nodein, edge):
        self.nodein = nodein
        self.nodemid = nodemid
        self.nodeout = nodeout
        self.edges = [edge]

    def append(self, edge):
        self.edges.append(edge)


class GraphDot:
    __slots__ = (
        "_graph",
        "_node_id_map",
        "_show",
        "_nodes_map_dag",
        "_nodes_open_input",
        "_nodes_open_output",
        "_edges",
        "_filter",
        "_filtered_nodes",
        "_enable_mid_node",
        "_hide_nodes_marked_hidden",
    )
    _graph: G.AGraph
    _node_id_map: dict
    _nodes_map_dag: dict[Node, G.agraph.Node]
    _filter: dict[str, list[str | int]]
    _filtered_nodes: set
    _enable_mid_node: bool
    _hide_nodes_marked_hidden: bool

    _show: set[
        Literal[
            "type",
            "mark",
            "label",
            "path",
            "index",
            "status",
            "data",
            "data_part",
            "data_summary",
        ]
    ]

    def __init__(
        self,
        graph_or_node: Graph | Node | None,
        *,
        graphattr: dict = {},
        edgeattr: dict = {},
        nodeattr: dict = {},
        show: Sequence | str = ["type", "mark", "label"],
        filter: Mapping[str, Sequence[str | int]] = {},
        label: str | None = None,
        agraph_kwargs: Mapping = {},
        enable_mid_node: bool = True,
        enable_common_attrs: bool = True,
        hide_nodes_marked_hidden: bool = True,
    ):
        if show == "full" or "full" in show:
            self._show = {
                "type",
                "mark",
                "label",
                "path",
                "index",
                "status",
                "data",
                "data_summary",
            }
        elif show == "all" or "all" in show:
            self._show = {
                "type",
                "mark",
                "label",
                "path",
                "index",
                "status",
                "data_part",
                "data_summary",
            }
        else:
            self._show = set(show)
        self._filter = {k: list(v) for k, v in filter.items()}
        self._filtered_nodes = set()
        self._enable_mid_node = enable_mid_node
        self._hide_nodes_marked_hidden = hide_nodes_marked_hidden

        graphattr = dict(graphattr)
        graphattr.setdefault("rankdir", "LR")
        graphattr.setdefault("dpi", 300)

        edgeattr = dict(edgeattr)
        edgeattr.setdefault("fontsize", 10)
        edgeattr.setdefault("labelfontsize", 9)
        edgeattr.setdefault("labeldistance", 1.2)

        nodeattr = dict(nodeattr)
        if any(s in self._show for s in ("data", "data_part")):
            nodeattr.setdefault("fontname", "Liberation Mono")

        self._node_id_map = {}
        self._nodes_map_dag = {}
        self._nodes_open_input = {}
        self._nodes_open_output = {}
        self._edges: dict[str, EdgeDef] = {}
        self._graph = G.AGraph(directed=True, strict=False, **agraph_kwargs)

        if enable_common_attrs:
            if graphattr:
                self._graph.graph_attr.update(graphattr)
            if edgeattr:
                self._graph.edge_attr.update(edgeattr)
            if nodeattr:
                self._graph.node_attr.update(nodeattr)

        if isinstance(graph_or_node, Graph):
            if label and enable_common_attrs:
                self.set_label(label)
            self._transform_graph(graph_or_node)
        elif isinstance(graph_or_node, Node):
            self._transform_from_nodes(graph_or_node)
        elif graph_or_node != None:
            raise RuntimeError("Invalid graph entry point")

    @classmethod
    def from_graph(cls, graph: Graph, *args, **kwargs) -> GraphDot:
        gd = cls(None, *args, **kwargs)
        if label := kwargs.pop("label", graph.label()):
            gd.set_label(label)
        gd._transform_graph(graph)
        return gd

    def _transform_graph(self, dag: Graph) -> None:
        for nodedag in dag._nodes:
            if self._node_is_filtered(nodedag):
                continue
            self._add_node(nodedag)
        for nodedag in dag._nodes:
            if self._node_is_filtered(nodedag):
                continue
            self._add_open_inputs(nodedag)
            self._add_edges(nodedag)
        self.update_style()

    @classmethod
    def from_object(cls, obj: Output | Node | Graph, *args, **kwargs) -> GraphDot:
        match obj:
            case Output():
                return cls.from_output(obj, *args, **kwargs)
            case Node():
                return cls.from_node(obj, *args, **kwargs)
            case Graph():
                return cls.from_graph(obj, *args, **kwargs)

        raise RuntimeError("Invalid object")

    @classmethod
    def from_output(cls, output: Output, *args, **kwargs) -> GraphDot:
        return cls.from_node(output.node, *args, **kwargs)

    @classmethod
    def from_node(cls, node: Node, *args, **kwargs) -> GraphDot:
        return cls.from_nodes((node,), *args, **kwargs)

    @classmethod
    def from_nodes(
        cls,
        nodes: Sequence[Node],
        *args,
        min_depth: int | None = None,
        max_depth: int | None = None,
        min_size: int | None = None,
        keep_direction: bool = False,
        **kwargs,
    ) -> GraphDot:
        node0 = nodes[0]

        gd = cls(None, *args, **kwargs)
        label = [node0.name]
        if min_depth is not None:
            label.append(f"{min_depth=:+d}")
        if max_depth is not None:
            label.append(f"{max_depth=:+d}")
        if min_size is not None:
            label.append(f"{min_size=:d}")
        gd.set_label(", ".join(label))

        gd._transform_from_nodes(
            nodes,
            min_depth=min_depth,
            max_depth=max_depth,
            min_size=min_size,
            keep_direction=keep_direction,
        )
        return gd

    def _transform_from_nodes(
        self,
        nodes: Sequence[Node] | Node,
        min_depth: int | None = None,
        max_depth: int | None = None,
        depth: int = 0,
        min_size: int | None = None,
        keep_direction: bool = True,
        **kwargs,
    ) -> None:
        if isinstance(nodes, Node):
            nodes = (nodes,)

        gw_kwargs = {}
        if min_size:

            def node_skip_fcn(node: Node, depth: int) -> bool:
                try:
                    o0size = node.outputs[0].dd.size
                except IndexError:
                    return False
                return depth <= 0 and not num_in_range(o0size, min_size)

            gw_kwargs["node_skip_fcn"] = node_skip_fcn

        graph_walker = GraphWalker(
            min_depth=min_depth,
            max_depth=max_depth,
            process_full_graph=not keep_direction,
            **gw_kwargs,
        )

        for node in nodes:
            logger.debug(f"Extending graph with node {node.labels.path or node.name}, {kwargs=}")
            if self._node_is_filtered(node):
                return

            graph_walker.process_from_node(node)

        for node, depth in graph_walker.nodes.items():
            self._add_node(node, depth=depth)

        for nodedag in self._nodes_map_dag:
            self._add_open_inputs(nodedag)
            self._add_edges(nodedag)

        self.update_style()

    def _add_node(self, nodedag: Node, *, depth: int | None = None) -> None:
        if nodedag in self._nodes_map_dag or self._node_is_filtered(nodedag):
            return

        styledict = {"shape": "Mrecord", "label": self.get_label(nodedag, depth=depth)}
        target = self.get_id(nodedag)
        self._graph.add_node(target, **styledict)
        nodedot = self._graph.get_node(target)
        nodedot.attr["nodedag"] = nodedag
        nodedot.attr["depth"] = depth
        self._nodes_map_dag[nodedag] = nodedot

    def _add_open_inputs(self, nodedag):
        if self._node_is_filtered(nodedag):
            return
        for input in nodedag.inputs.iter_all():
            if (
                not input.connected()
                or self._node_is_filtered(input.parent_node)
                or self._node_is_missing(input.parent_node)
            ):
                self._add_open_input(input, nodedag)

    def _add_open_input(self, input, nodedag):
        if self._node_is_filtered(nodedag):
            return
        styledict = {}
        source = self.get_id(input, "_in")
        target = self.get_id(nodedag)

        self._get_index(input, styledict, "headlabel")

        self._graph.add_node(source, label="", shape="none", **styledict)
        self._graph.add_edge(source, target, **styledict)

        nodein = self._graph.get_node(source)
        edge = self._graph.get_edge(source, target)
        nodeout = self._graph.get_node(target)

        self._nodes_open_input[input] = nodein
        self._edges[input] = EdgeDef(nodein, None, nodeout, edge)

    def _add_open_output(self, nodedag, output):
        if self._node_is_filtered(nodedag):
            return
        styledict = {}
        source = self.get_id(nodedag)
        target = self.get_id(output, "_out")
        self._get_index(output, styledict, "taillabel")

        self._graph.add_node(target, label="", shape="none", **styledict)
        self._graph.add_edge(source, target, arrowhead="empty", **styledict)
        nodein = self._graph.get_node(source)
        edge = self._graph.get_edge(source, target)
        nodeout = self._graph.get_node(target)

        self._nodes_open_output[output] = nodeout
        self._edges[output] = EdgeDef(nodein, None, nodeout, edge)

    def _add_edges(self, nodedag):
        if self._node_is_filtered(nodedag):
            return
        for _, output in enumerate(nodedag.outputs.iter_all()):
            if output.connected():
                if len(output.child_inputs) > 1:
                    self._add_edges_multi_alot(nodedag, output)
                # elif len(output.child_inputs) > 1:
                #     self._add_edges_multi_few(iout, nodedag, output)
                else:
                    self._add_edge(nodedag, output, output.child_inputs[0])
            else:
                self._add_open_output(nodedag, output)

            if output.dd.axes_edges:
                self._add_edge_hist(output)
            if output.dd.axes_meshes:
                self._add_mesh(output)

    def _add_edges_multi_alot(self, nodedag, output):
        if self._node_is_filtered(nodedag):
            return

        if self._enable_mid_node:
            virtual_mid_node = self.get_id(output, "_mid")

            edge_added = False
            for input in output.child_inputs:
                if self._add_edge(nodedag, output, input, vtarget=virtual_mid_node):
                    edge_added = True
                    break
            for input in output.child_inputs:
                edge_added |= self._add_edge(nodedag, output, input, vsource=virtual_mid_node)

            if edge_added:
                self._graph.add_node(
                    virtual_mid_node,
                    label="",
                    shape="cds",
                    width=0.1,
                    height=0.1,
                    color="forestgreen",
                    weight=10,
                )
        else:
            for input in output.child_inputs:
                self._add_edge(nodedag, output, input)

    def _add_edges_multi_few(self, iout: int, nodedag, output):
        if self._node_is_filtered(nodedag):
            return
        style = {"sametail": str(iout), "weight": 5}
        for input in output.child_inputs:
            self._add_edge(nodedag, output, input, style=style)
            style["taillabel"] = ""

    def _add_edge_hist(self, output: Output) -> None:
        if self._node_is_filtered(output.node):
            return
        if output.dd.edges_inherited:
            return

        for eoutput in output.dd.axes_edges:
            self._add_edge(eoutput.node, eoutput, output, style={"style": "dashed"})

    def _add_mesh(self, output: Output) -> None:
        if self._node_is_filtered(output.node):
            return
        if output.dd.meshes_inherited:
            return
        for noutput in output.dd.axes_meshes:
            self._add_edge(noutput.node, noutput, output, style={"style": "dotted"})

    def _get_index(self, leg, styledict: dict, target: str):
        if isinstance(leg, Input):
            container = leg.node.inputs
            connected = leg.connected()
        else:
            container = leg.node.outputs
            connected = True

        if container.len_all() < 2 and connected:
            return

        idx = ""
        try:
            idx = container.index(leg)
        except ValueError:
            pass
        else:
            idx = str(idx)

        if not connected:
            try:
                idx2 = container.key(leg)
            except ValueError:
                pass
            else:
                idx = f"{idx}: {idx2}" if idx else idx2
        if idx:
            styledict[target] = str(idx)

    def _add_edge(
        self,
        nodedag,
        output,
        input,
        *,
        vsource: str | None = None,
        vtarget: str | None = None,
        style: dict | None = None,
    ) -> bool:
        if self._node_is_missing(input.node):
            return False
        if self._node_is_missing(nodedag):
            return False
        styledict = style or {}

        if vsource is not None:
            source = vsource
            styledict["arrowtail"] = "none"
        else:
            source = self.get_id(nodedag)
            self._get_index(output, styledict, "taillabel")

        if vtarget is not None:
            target = vtarget
            styledict["arrowhead"] = "none"
        else:
            target = self.get_id(input.node)
            self._get_index(input, styledict, "headlabel")

        self._graph.add_edge(source, target, **styledict)

        nodein = self._graph.get_node(source)
        edge = self._graph.get_edge(source, target)
        nodeout = self._graph.get_node(target)

        edgedef = self._edges.get(input, None)
        if edgedef is None:
            self._edges[input] = EdgeDef(nodein, None, nodeout, edge)
        else:
            edgedef.append(edge)

        return True

    def _node_is_missing(self, node: Node) -> bool:
        return node not in self._nodes_map_dag

    def _node_is_filtered(self, node: Node) -> bool:
        if node in self._filtered_nodes:
            return True

        if self._hide_nodes_marked_hidden and node.labels.node_hidden:
            self._filtered_nodes.add(node)
            return True

        if not node.labels.index_in_mask(self._filter):
            self._filtered_nodes.add(node)
            return True

        return False

    def _set_style_node(self, node, attr):
        if node is None:
            attr["color"] = "gray"
            return

        try:
            if node.invalid:
                attr["color"] = "black"
            elif node.tainted:
                attr["color"] = "red"
            elif node.frozen_tainted:
                attr["color"] = "blue"
            elif node.frozen:
                attr["color"] = "cyan"
            elif node.immediate:
                attr["color"] = "green"
            else:
                attr["color"] = "forestgreen"

            if node.exception is not None:
                attr["color"] = "magenta"
        except AttributeError:
            attr["color"] = "yellow"

        if attr.get("depth") == "0":
            attr["penwidth"] = 2

    def _set_style_edge(self, obj, attrin, attr, attrout):
        if isinstance(obj, Input):
            if obj.connected():
                node = obj.parent_output.node
            else:
                node = None
                self._set_style_node(node, attrin)
        else:
            node = obj.node
            self._set_style_node(node, attrout)

        self._set_style_node(node, attr)

        if isinstance(obj, Input):
            allocated_on_input = obj.owns_buffer
            try:
                allocated_on_output = obj.parent_output.owns_buffer
            except AttributeError:
                allocated_on_output = True
        elif isinstance(obj, Output):
            allocated_on_input = False
            allocated_on_output = obj.owns_buffer
        attr.update({"dir": "both", "arrowsize": 0.5})
        attr["arrowhead"] = attr["arrowhead"] or allocated_on_input and "dotopen" or "odotopen"
        attr["arrowtail"] = attr["arrowtail"] or allocated_on_output and "dot" or "odot"

        if node:
            if node.frozen:
                attrin["color"] = "gray"
            elif attr["color"] == "gray":
                del attr["color"]

    def update_style(self):
        for nodedag, nodedot in self._nodes_map_dag.items():
            self._set_style_node(nodedag, nodedot.attr)

        for obj, edgedef in self._edges.items():
            for edge in edgedef.edges:
                self._set_style_edge(obj, edgedef.nodein.attr, edge.attr, edgedef.nodeout.attr)

    def set_label(self, label: str):
        self._graph.graph_attr["label"] = label

    def savegraph(self, fname: Path | str, *, quiet: bool = False):
        if not isinstance(fname, Path):
            fname = Path(fname)
        if not quiet:
            logger.log(INFO1, f"Write: {fname}")
        if fname.suffix == ".dot":
            self._graph.write(fname)
        else:
            self._graph.layout(prog="dot")
            self._graph.draw(fname)

        if not self._nodes_map_dag:
            logger.warning(f"No nodes saved for {fname}")

    def get_id(self, obj, suffix: str = "") -> str:
        name = type(obj).__name__
        omap = self._node_id_map.setdefault(name, {})
        onum = omap.setdefault(obj, len(omap))
        return f"{name}_{onum}{suffix}"

    def get_label(self, node: Node, *, depth: int | None = None) -> str:
        text = node.labels.graph or node.name
        try:
            out0 = node.outputs[0]
        except IndexError:
            shape0 = ""
            dtype0 = ""
            hasedges = False
            hasnodes = False
            out0 = None
        else:
            hasedges = bool(out0.dd.axes_edges)
            hasnodes = bool(out0.dd.axes_meshes)
            shape0 = out0.dd.shape
            if shape0 is None:
                shape0 = "?"
            shape0 = "x".join(str(s) for s in shape0)

            dtype0 = out0.dd.dtype
            dtype0 = "?" if dtype0 is None else dtype0.char
        nout_pos = len(node.outputs)
        nout_nonpos = node.outputs.len_all() - nout_pos
        nout = []
        if nout_pos:
            nout.append(f"{nout_pos}p")
        if nout_nonpos:
            nout.append(f"{nout_nonpos}k")
        nout = "+".join(nout) or "0"

        nin_pos = len(node.inputs)
        nin_nonpos = node.inputs.len_all() - nin_pos
        nin = []
        if nin_pos:
            nin.append(f"{nin_pos}p")
        if nin_nonpos:
            nin.append(f"{nin_nonpos}k")
        nin = "+".join(nin) or "0"

        nlimbs = f"{nin}→{nout}"

        left, right = [], []
        br_left, br_right = ("\\{", "\\}") if hasedges else ("[", "]")
        if hasnodes:
            br_right += "…"
        if shape0:
            info_type = f"{br_left}{shape0}{br_right}{dtype0}\\n{nlimbs}"
        else:
            info_type = f"{nlimbs}"
        if "type" in self._show:
            left.append(info_type)
        if "mark" in self._show and (mark := node.labels.mark) is not None:
            left.append(mark)
        if depth is not None:
            left.append(f"d: {depth:+d}".replace("-", "−"))
        if "label" in self._show:
            right.append(text)
        if "path" in self._show and (paths := node.labels.paths):
            if len(paths) > 1:
                right.append(f"path[{len(paths)}]: {paths[0]}, …")
            else:
                right.append(f"path: {paths[0]}")
        if "index" in self._show and (index := node.labels.index_values):
            right.append(f'index: {", ".join(index)}')
        if "status" in self._show:
            status = []
            with suppress(AttributeError):
                if node.types_tainted:
                    status.append("types_tainted")
            with suppress(AttributeError):
                if node.tainted:
                    status.append("tainted")
            with suppress(AttributeError):
                if node.frozen:
                    status.append("frozen")
            with suppress(AttributeError):
                if node.frozen_tainted:
                    status.append("frozen_tainted")
            with suppress(AttributeError):
                if node.invalid:
                    status.append("invalid")
            with suppress(AttributeError):
                if not node.closed:
                    status.append("open")
            if status:
                right.append(status)

        show_data = "data" in self._show
        show_data_part = "data_part" in self._show
        show_data_summary = "data_summary" in self._show
        need_data = show_data or show_data_part or show_data_summary
        if need_data and out0 is not None:
            tainted = "tainted" if out0.tainted else "updated"
            data = None
            try:
                data = out0.data
            except UnclosedGraphError:
                data = out0._data
            except Exception:
                right.append("cought exception")
                data = out0._data

            if show_data_summary and data is not None:
                sm = data.sum()
                sm2 = square(data).sum()
                mn = data.min()
                mx = data.max()
                avg = data.mean()
                block = [
                    f"Σ={sm:.2g}",
                    f"Σ²={sm2:.2g}",
                    f"avg={avg:.2g}",
                    f"min={mn:.2g}",
                    f"max={mx:.2g}",
                    f"{tainted}",
                ]
                right.append(block)

            if show_data_part:
                right.append(_format_data(data, part=True))

            if show_data:
                right.append(_format_data(data))

        if getattr(node, "exception", None) is not None:
            if node.closed:
                logger.log(INFO1, f"Exception: {node.exception}")
            right.append(node.exception)

        return self._combine_labels((left, right))

    def _combine_labels(self, labels: Sequence | str) -> str:
        if isinstance(labels, str):
            return labels

        slabels = [self._combine_labels(l) for l in labels]
        return f"{{{'|'.join(slabels)}}}"


def num_in_range(num: int, minnum: int | None, maxnum: int | None = None) -> bool:
    if minnum is not None and num < minnum:
        return False
    return maxnum is None or num <= maxnum


def _get_lead_mid_trail(array: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    lead = array[:3]
    nmid = (array.shape[0] - 1) // 2 - 1
    mid = array[nmid : nmid + 3]
    tail = array[-3:]
    return lead, mid, tail


def _format_1d(array: NDArray) -> str:
    if array.size < 13:
        with printoptions(precision=6):
            return str(array)

    with printoptions(threshold=17, precision=2):
        lead, mid, tail = _get_lead_mid_trail(array)

        leadstr = str(lead)[:-1]
        midstr = str(mid)[1:-1]
        tailstr = str(tail)[1:]
        return f"{leadstr} ... {midstr} ... {tailstr}"


def _format_2d(array: NDArray) -> str:
    n0 = array.shape[0]
    if n0 < 13:
        contents = "\n".join(map(_format_1d, array))
        return f"[{contents}]"

    lead, mid, tail = _get_lead_mid_trail(array)

    leadstr = _format_2d(lead)[:-1]
    midstr = _format_2d(mid)[1:-1]
    tailstr = _format_2d(tail)[1:]
    return f"{leadstr}\n...\n{midstr}\n...\n{tailstr}"


def _format_data(data: NDArray | None, part: bool = False) -> str:
    if data is None:
        return "None"
    if part:
        if data.size < 13 or data.ndim > 2:
            with printoptions(precision=6):
                datastr = str(data)
        elif data.ndim == 1:
            datastr = _format_1d(data)
        else:
            datastr = _format_2d(data)
    else:
        datastr = str(data)
    return datastr.replace("\n", "\\l") + "\\l"


NodeT = TypeVar("NodeT")
OutputT = TypeVar("OutputT")
InputT = TypeVar("InputT")


class NodeHandlerBase(Generic[NodeT, OutputT, InputT]):
    __slots__ = ()

    @abstractmethod
    def iter_inputs(self, node: NodeT) -> Generator[InputT, None, None]:
        pass

    @abstractmethod
    def iter_outputs(self, node: NodeT) -> Generator[OutputT, None, None]:
        pass

    @abstractmethod
    def iter_meshes_edges(self, node: NodeT) -> Generator[OutputT, None, None]:
        pass


class NodeHandlerDGM(NodeHandlerBase[Node, Output, Input]):
    __slots__ = ()

    @override
    def iter_inputs(self, node: Node) -> Generator[Input, None, None]:
        yield from node.inputs.iter_all()

    @override
    def iter_outputs(self, node: Node) -> Generator[Output, None, None]:
        yield from node.outputs.iter_all()

    @override
    def iter_meshes_edges(self, node: Node) -> Generator[Output, None, None]:
        for output in node.outputs.iter_all():
            yield from output.dd.axes_edges
            yield from output.dd.axes_edges


class GraphWalker:
    __slots__ = (
        "nodes",
        "_queue_nodes",
        "_queue_meshes_edges",
        "edges",
        "open_outputs",
        "open_inputs",
        "_current_depth",
        "min_depth",
        "max_depth",
        "_enable_process_backward",
        "_enable_process_forward",
        "_enable_process_meshes_edges",
        "_enable_process_full_graph",
        "_node_skip_fcn",
        "_node_handler",
    )

    nodes: dict[Node, int]
    _queue_nodes: dict[Node, int]
    _queue_meshes_edges: dict[Node, int]
    edges: dict[Input, tuple[Node, Node]]
    open_outputs: list[Output]
    open_inputs: list[Input]

    _current_depth: int

    min_depth: int | None
    max_depth: int | None

    _enable_process_backward: bool
    _enable_process_forward: bool
    _enable_process_backward: bool
    _enable_process_meshes_edges: bool
    _enable_process_full_graph: bool

    _node_skip_fcn: Callable[[Node], bool]

    _node_handler: NodeHandlerBase

    def __init__(
        self,
        *,
        min_depth: int | None,
        max_depth: int | None,
        process_backward: bool = True,
        process_forward: bool = True,
        process_meshes_edges: bool = False,
        process_full_graph: bool = False,
        node_skip_fcn: Callable[[Node], bool] = lambda _: False,
        node_handler: NodeHandlerBase | None
    ):
        self.nodes = {}
        self._queue_nodes = {}
        self._queue_meshes_edges = {}
        self.edges = {}
        self.open_outputs = []
        self.open_inputs = []
        self._current_depth = 0
        self.min_depth = min_depth
        self.max_depth = max_depth
        self._enable_process_backward = process_backward
        self._enable_process_forward = process_forward
        self._enable_process_meshes_edges = process_meshes_edges
        self._enable_process_full_graph = process_full_graph

        self._node_skip_fcn = node_skip_fcn

        self._node_handler = node_handler or NodeHandlerDGM()

        growth_disabled = not process_backward and not process_forward
        if growth_disabled and process_full_graph:
            raise RuntimeError("GraphWalker got conflicting arguments: "
                               f"{process_full_graph=}, {process_backward=} and {process_forward=}")
        if not process_backward and process_meshes_edges:
            raise RuntimeError("GraphWalker got conflicting arguments: "
                               f"{process_backward=} and {process_meshes_edges=}")

    def process_from_node(
        self,
        node: Node,
        *,
        depth: int | None = 0,
        process_full_graph: bool | None = None,
        process_initial_node: bool = True,
    ):
        """Main function

        1. Add the node to the queue.
        2. Go strictly backward from the node. Add nodes to the queue.
        3. Go strictly forward from the node. Add nodes to the queue.
        4. For each node in the queue:
            a. Optionally: process nodes forward/backward.
            b. Push queue to the list of nodes. The queue is copied to the queue of meshes/edges.
            c. Optionally: process meshes/edges.
            d. Repeat as long as new nodes are added to the queue.
        """
        if process_full_graph is None:
            process_full_graph = self._enable_process_full_graph

        self.current_depth = depth
        depth = self.current_depth
        if process_initial_node and not self._add_node_to_queue(node):
            return

        if self._enable_process_backward:
            self._build_queue_nodes_backward_from(node, depth=depth)

        if self._enable_process_forward:
            self._build_queue_nodes_forward_from(node, depth=depth)

        while self.has_queue:
            if self._enable_process_full_graph:
                self._process_nodes_from_queue()
            else:
                self._push_queue_to_storage()

            if self._enable_process_meshes_edges:
                self._process_queue_meshes_edges()
            else:
                self._queue_meshes_edges = {}

    @property
    def has_queue(self) -> bool:
        return bool(self._queue_nodes or self._queue_meshes_edges)

    @property
    def current_depth(self) -> int:
        return self._current_depth

    @current_depth.setter
    def current_depth(self, value: int | None):
        if value is not None:
            self._current_depth = value

    def _push_queue_to_storage(self):
        self.nodes.update(self._queue_nodes)
        self._queue_meshes_edges.update(self._queue_nodes)
        self._queue_nodes = {}

    def _process_nodes_from_queue(self):
        while self._queue_nodes:
            self._process_nodes_from_queue_once()

    def _process_nodes_from_queue_once(self):
        """Goes forward/backward from each node from the queue."""
        if not self._queue_nodes:
            return
        queue = self._queue_nodes.copy()
        self._push_queue_to_storage()

        for node, self.current_depth in queue.items():
            if self._may_not_add_node(node):
                continue
            self.process_from_node(
                node,
                depth=self.current_depth,
                process_full_graph=False,
                process_initial_node=False,
            )
        return queue

    def _depth_outside_limits(self) -> bool:
        if self.min_depth is not None and self.current_depth < self.min_depth:
            return True

        if self.max_depth is not None and self.current_depth > self.max_depth:
            return True

        return False

    def _node_already_added(self, node: Node) -> bool:
        return node in self._queue_nodes or node in self.nodes

    def _may_not_add_node(self, node: Node) -> bool:
        return (
            self._depth_outside_limits()
            or self._node_already_added(node)
            or self._node_skip_fcn(node)
        )

    def _add_node_to_queue(self, node: Node) -> bool:
        if self._depth_outside_limits() or self._node_already_added(node):
            return False

        self._queue_nodes[node] = self._current_depth

        return True

    def _build_queue_nodes_backward_from(self, node: Node, depth: int | None = None):
        """Go strictly backward from the node.
        Add nodes to queue.
        Stop if depth is too low.
        """
        self.current_depth = depth
        self.current_depth -= 1
        if self._depth_outside_limits():
            return
        for input in self._node_handler.iter_inputs(node):
            try:
                parent_node = input.parent_node
            except AttributeError:
                self.open_inputs[input]
                continue

            if not input in self.edges:
                self.edges[input, (parent_node, node)]  # pyright: ignore [reportArgumentType]
            if not self._add_node_to_queue(parent_node):
                continue
            self._build_queue_nodes_backward_from(parent_node)

    def _build_queue_nodes_forward_from(self, node: Node, *, depth: int | None = None):
        """Go strictly forward from the node.
        Add nodes to queue.
        Stop if depth is too high.
        """
        self.current_depth = depth
        self.current_depth += 1
        if self._depth_outside_limits():
            return
        for output in self._node_handler.iter_outputs(node):
            if output.child_inputs:
                for child_input in output.child_inputs:
                    child_node = child_input.node
                    if not self._add_node_to_queue(child_node):
                        continue
                    self._build_queue_nodes_forward_from(child_node)
            else:
                self.open_outputs[output]

    def _process_queue_meshes_edges(self):
        """Add nodes of meshes and edges to the queue, if they are not already present."""
        for node, self._current_depth in self._queue_meshes_edges.items():
            self.current_depth -= 1
            if self._depth_outside_limits():
                continue
            for output in self._node_handler.iter_meshes_edges(node):
                self._add_node_to_queue(output.node)

        self._queue_meshes_edges = {}
