from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

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
    GraphDot = None

    def savegraph(*args, **kwargs):
        pass

else:

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
        )
        _graph: G.AGraph
        _node_id_map: dict
        _nodes_map_dag: dict[Node, G.agraph.Node]
        _filter: dict[str, list[str | int]]
        _filtered_nodes: set

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

            if graphattr:
                self._graph.graph_attr.update(graphattr)
            if edgeattr:
                self._graph.edge_attr.update(edgeattr)
            if nodeattr:
                self._graph.node_attr.update(nodeattr)

            if isinstance(graph_or_node, Graph):
                if label:
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
            mindepth: int | None = None,
            maxdepth: int | None = None,
            minsize: int | None = None,
            keep_direction: bool = False,
            **kwargs,
        ) -> GraphDot:
            node0 = nodes[0]

            gd = cls(None, *args, **kwargs)
            label = [node0.name]
            if mindepth is not None:
                label.append(f"{mindepth=:+d}")
            if maxdepth is not None:
                label.append(f"{maxdepth=:+d}")
            if minsize is not None:
                label.append(f"{minsize=:d}")
            gd.set_label(", ".join(label))

            gd._transform_from_nodes(
                nodes,
                mindepth=mindepth,
                maxdepth=maxdepth,
                minsize=minsize,
                keep_direction=keep_direction,
            )
            return gd

        def _transform_from_nodes(self, nodes: Sequence[Node] | Node, **kwargs) -> None:
            if isinstance(nodes, Node):
                nodes = (nodes,)

            for node in nodes:
                logger.debug(
                    f"Extending graph with node {node.labels.path or node.name}, {kwargs=}"
                )
                if self._node_is_filtered(node):
                    return

                self._add_nodes_backward_recursive(node, including_self=True, **kwargs)
                self._add_nodes_forward_recursive(node, including_self=False, **kwargs)

            for nodedag in self._nodes_map_dag:
                self._add_open_inputs(nodedag)
                self._add_edges(nodedag)

            self.update_style()

        def _add_node_only(
            self,
            node: Node,
            *,
            mindepth: int | None = None,
            maxdepth: int | None = None,
            depth: int = 0,
            minsize: int | None = None,
        ) -> bool:
            if node in self._nodes_map_dag:
                return False
            if not num_in_range(depth, mindepth, maxdepth):
                return False
            # print(f"{depth=: 2d}: {node.name}")

            try:
                o0size = node.outputs[0].dd.size
            except IndexError:
                pass
            else:
                if depth <= 0 and not num_in_range(o0size, minsize):
                    return False

            self._add_node(node, depth=depth)

            return True

        def _add_nodes_backward_recursive(
            self,
            node: Node,
            *,
            including_self: bool = False,
            mindepth: int | None = None,
            maxdepth: int | None = None,
            minsize: int | None = None,
            keep_direction: bool = False,
            depth: int = 0,
            visited_nodes: set[Node] = set(),
        ) -> None:
            if self._node_is_filtered(node):
                return
            visited_nodes.add(node)

            if including_self and not self._add_node_only(
                node, mindepth=mindepth, maxdepth=maxdepth, depth=depth, minsize=minsize
            ):
                return

            newdepth = depth - 1
            if newdepth < 0 or not keep_direction:
                for input in node.inputs.iter_all():
                    try:
                        parent_node = input.parent_node
                    except AttributeError:
                        continue
                    self._add_nodes_backward_recursive(
                        parent_node,
                        including_self=True,
                        depth=newdepth,
                        mindepth=mindepth,
                        maxdepth=maxdepth,
                        minsize=minsize,
                        keep_direction=keep_direction,
                        visited_nodes=visited_nodes,
                    )

            if not keep_direction:
                self._add_nodes_forward_recursive(
                    node,
                    including_self=False,
                    depth=depth,
                    mindepth=mindepth,
                    maxdepth=maxdepth,
                    minsize=minsize,
                    keep_direction=keep_direction,
                    ignore_visit=True,
                    visited_nodes=visited_nodes,
                )

        def _add_nodes_forward_recursive(
            self,
            node: Node,
            *,
            including_self: bool = False,
            mindepth: int | None = None,
            maxdepth: int | None = None,
            minsize: int | None = None,
            keep_direction: bool = False,
            depth: int = 0,
            visited_nodes: set[Node] = set(),
            ignore_visit: bool = False,
        ) -> None:
            if self._node_is_filtered(node):
                return
            if depth != 0 and node in visited_nodes and not ignore_visit:
                return
            visited_nodes.add(node)
            if including_self and not self._add_node_only(
                node, mindepth=mindepth, maxdepth=maxdepth, depth=depth, minsize=minsize
            ):
                return

            newdepth = depth + 1
            for output in node.outputs.iter_all():
                for child_input in output.child_inputs:
                    self._add_nodes_backward_recursive(
                        child_input.node,
                        including_self=True,
                        depth=newdepth,
                        keep_direction=keep_direction,
                        mindepth=mindepth,
                        maxdepth=maxdepth,
                        minsize=minsize,
                        visited_nodes=visited_nodes,
                    )

                    if newdepth > 0 or not keep_direction:
                        self._add_nodes_forward_recursive(
                            child_input.node,
                            depth=newdepth,
                            keep_direction=keep_direction,
                            mindepth=mindepth,
                            maxdepth=maxdepth,
                            minsize=minsize,
                            visited_nodes=visited_nodes,
                            ignore_visit=True,
                        )

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
            vnode = self.get_id(output, "_mid")

            edge_added = False
            for input in output.child_inputs:
                if self._add_edge(nodedag, output, input, vtarget=vnode):
                    edge_added = True
                    break
            for input in output.child_inputs:
                edge_added |= self._add_edge(nodedag, output, input, vsource=vnode)

            if edge_added:
                self._graph.add_node(
                    vnode,
                    label="",
                    shape="cds",
                    width=0.1,
                    height=0.1,
                    color="forestgreen",
                    weight=10,
                )

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

            if node.labels.node_hidden:
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

        def savegraph(self, fname, *, quiet: bool = False):
            if not quiet:
                logger.log(INFO1, f"Write: {fname}")
            if fname.endswith(".dot"):
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
                if len(paths)>1:
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
