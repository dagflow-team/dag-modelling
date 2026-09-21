from __future__ import annotations

from weakref import ref as weakref

from nested_mapping.typing import properkey

from ..core.labels import Labels
from ..tools.logger import Logger, get_logger
from .exception import (
    ClosedGraphError,
    CriticalError,
    DagModellingError,
    InitializationError,
    UnclosedGraphError,
)


class _NodeMethods:
    __slots__ = ()

    def __init__(
        self,
        name,
        *,
        label=None,
        graph=None,
        debug=None,
        logger=None,
        input_strategy=None,
        immediate=False,
        frozen=False,
        allowed_kw_inputs=(),
        **kwargs,
    ):
        from .graph import Graph
        from .node_base import NodeBase
        from .node_eval import NodeEval
        from .node_structure import NodeStructure

        # Not super(): this mixin isn't inserted into any MRO chain
        # (the cdef class consumes it via attribute binding), so
        # super-dispatch wouldn't see NodeBase.
        NodeBase.__init__(self, input_strategy=input_strategy)
        self._graph = None
        self._exception = None
        self._internal_buffers = {}
        self._meta_node = None

        self._name = name
        self._evaluator = NodeEval(
            children=self.outputs, parents=self.inputs, immediate=immediate, **kwargs
        )
        self._structure = NodeStructure(
            node=self,
            inputs=self.inputs,
            outputs=self.outputs,
            evaluator=self._evaluator,
            allowed_kw_inputs=tuple(allowed_kw_inputs),
        )

        self.graph = Graph.current() if graph is None else graph
        if debug is None and self.graph is not None:
            self._debug = self.graph.debug
        else:
            self._debug = bool(debug)

        self._labels = Labels(label or name)

        if isinstance(logger, Logger):
            self._logger = logger
        elif logger is not None:
            raise InitializationError(f"Cannot initialize a node with logger={logger}", node=self)
        elif self.graph is not None:
            self._logger = self.graph.logger
        else:
            self._logger = get_logger()

        self._evaluator.frozen = frozen

        self._evaluator._functions_dict = {"default": self._function}
        self._evaluator.function = self._evaluator._functions_dict["default"]

    def __str__(self):
        from .node_base import NodeBase

        return f"{{{self.name}}} {NodeBase.__str__(self)}"

    # Back-compat proxies: _functions_dict / _input_nodes_callbacks were
    # moved to NodeEval in commit afeebcf; external consumers in
    # dgm-dayabay-dev (AxisDistortionMatrixLinearLegacy, Monotonize) still
    # touch them on Node directly. TODO: remove after they migrate.
    @property
    def _functions_dict(self):
        return self._evaluator._functions_dict

    @property
    def _input_nodes_callbacks(self):
        return self._evaluator._input_nodes_callbacks

    def choose_function(self, name):
        self._evaluator.function = self._evaluator._functions_dict[name]

    @classmethod
    def from_args(cls, name, *positional_connectibles, kwargs={}, **key_connectibles):
        instance = cls(name, **kwargs)
        for connectible in positional_connectibles:
            connectible >> instance
        for key, connectible in key_connectibles.items():
            connectible >> instance.inputs[key]
        return instance

    @classmethod
    def replicate(
        cls,
        *,
        name,
        replicate_outputs=((),),
        verbose=False,
        **kwargs,
    ):
        from .storage import NodeStorage

        storage = NodeStorage(default_containers=True)
        nodes = storage("nodes")
        inputs = storage("inputs")
        outputs = storage("outputs")

        if not replicate_outputs:
            raise RuntimeError("`replicate_outputs` tuple should have at least one item")

        tuplename = (name,)
        for key in replicate_outputs:
            key = properkey(key)
            outname = tuplename + key
            instance = cls(".".join(outname), **kwargs)
            nodes[outname] = instance

            ninputs = instance.inputs.len_all()
            noutputs = instance.outputs.len_all()
            if noutputs == 0:
                instance()
                ninputs = instance.inputs.len_all()
                noutputs = instance.outputs.len_all()

            iter_inputs = instance.inputs.iter_all_items()
            if ninputs > 1:
                for iname, input in iter_inputs:
                    inputs[tuplename + (iname,) + key] = input
            elif ninputs == 1:
                _, input = next(iter_inputs)
                inputs[tuplename + key] = input

            iter_outputs = instance.outputs.iter_all_items()
            if noutputs > 1:
                for oname, output in instance.outputs.iter_all_items():
                    outputs[tuplename + (oname,) + key] = output
            else:
                _, output = next(iter_outputs)
                outputs[tuplename + key] = output

        NodeStorage.update_current(storage, strict=True, verbose=verbose)

        return (instance, storage) if len(replicate_outputs) == 1 else (None, storage)

    #
    # Properties
    #
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def allowed_kw_inputs(self):
        return self._structure._allowed_kw_inputs

    @property
    def exception(self):
        return self._exception

    @property
    def meta_node(self):
        return self._meta_node and self._meta_node()

    @meta_node.setter
    def meta_node(self, value):
        if self._meta_node is not None:
            raise RuntimeError("MetaNode may be set only once")
        self._meta_node = weakref(value)

    @property
    def logger(self):
        return self._logger

    @property
    def tainted(self):
        return self._evaluator.tainted

    @property
    def tainted_value(self):
        return self._evaluator.tainted_value

    def tainted_above_threshold(self, taint_threshold: int = 0):
        return self._evaluator.tainted_above_threshold(taint_threshold)

    @property
    def frozen_tainted(self):
        return self._evaluator.frozen_tainted

    @property
    def frozen(self):
        return self._evaluator.frozen

    @property
    def closed(self):
        return self._evaluator.closed

    @property
    def debug(self):
        return self._debug

    @property
    def allocated(self):
        return self._evaluator.allocated

    @property
    def immediate(self):
        return self._evaluator._immediate

    @property
    def invalid(self):
        return self._evaluator.invalid

    @invalid.setter
    def invalid(self, invalid):
        self._evaluator._invalidate(invalid)

    @property
    def needs_post_allocate(self):
        return self._evaluator.needs_post_allocate

    @needs_post_allocate.setter
    def needs_post_allocate(self, value):
        self._evaluator.needs_post_allocate = value

    @property
    def function(self):
        return self._evaluator.function

    @function.setter
    def function(self, value):
        self._evaluator.function = value

    def invalidate(self, invalid=True):
        return self._evaluator.invalidate(invalid)

    def invalidate_children(self, invalid=True):
        return self._evaluator.invalidate_children(invalid)

    def invalidate_parents(self, invalid=True):
        return self._evaluator.invalidate_parents(invalid)

    def taint_type(self, force_taint: bool = False) -> None:
        if self._evaluator.closed:
            raise ClosedGraphError("Unable to taint type", node=self)
        self._evaluator.taint_type(force_taint=force_taint)

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, graph):
        if graph is None:
            return
        if self._graph is not None:
            raise DagModellingError("Graph is already defined")
        self._graph = graph
        self._graph.register_node(self)

    @property
    def labels(self):
        return self._labels

    def label(self):
        return self._labels.text

    @property
    def n_calls(self):
        return self._evaluator._n_calls

    @property
    def structure(self):
        return self._structure

    #
    # I/O management
    #
    def __call__(self, name=None, *args, **kwargs):
        return self._structure.make_or_get_input(name, *args, **kwargs)

    def _make_input(self, *args, **kwargs):
        return self._structure.make_input(*args, **kwargs)

    def add_input(self, name, **kwargs):
        return self._structure.add_input(name, **kwargs)

    def _add_inputs(self, name, **kwargs):
        return self._structure._add_inputs(name, **kwargs)

    def _add_input(self, name, *, positional=True, keyword=True, **kwargs):
        return self._structure._add_input(name, positional=positional, keyword=keyword, **kwargs)

    def add_output(self, name, *, keyword=True, positional=True, **kwargs):
        return self._structure.add_output(name, keyword=keyword, positional=positional, **kwargs)

    def _add_output(self, name, *, keyword=True, positional=True, **kwargs):
        return self._structure._add_output(name, keyword=keyword, positional=positional, **kwargs)

    def _add_outputs(self, name, **kwargs):
        return self._structure._add_outputs(name, **kwargs)

    def add_pair(self, iname, oname, **kwargs):
        return self._structure.add_pair(iname, oname, **kwargs)

    def _add_pair(self, iname, oname, input_kws=None, output_kws=None):
        return self._structure._add_pair(iname, oname, input_kws, output_kws)

    def _add_pairs(self, inames, onames, input_kws=None, output_kws=None):
        return self._structure._add_pairs(inames, onames, input_kws, output_kws)

    #
    # Evaluation
    #
    def _function(self):
        pass

    def eval(self):
        if not self.closed:
            raise UnclosedGraphError("Cannot evaluate not closed node!", node=self)
        self._eval()

    def _eval(self):
        self._evaluator._eval_implementation()

    def freeze(self):
        if self._evaluator.frozen:
            return
        if self._evaluator.tainted:
            raise CriticalError("Unable to freeze tainted node!", node=self)
        self._evaluator.freeze()

    def print(self):
        print(f"Node {self._name}: →[{self.inputs.len_all()}],[{self.outputs.len_all()}]→")
        for i, _input in enumerate(self.inputs):
            print("  ", i, _input)
        for _input in self.inputs.iter_nonpos():
            print("    ", _input)
        for i, output in enumerate(self.outputs):
            print("  ", i, output)
        for output in self.outputs.iter_nonpos():
            print("    ", output)

    def to_dict(self, *, label_from="text"):
        return {
            "label": self.labels[label_from],
        }

    def _type_function(self):
        """A output takes this function to determine the dtype and shape."""
        raise DagModellingError("Unimplemented method: the method must be overridden!")

    def _post_allocate(self):
        self._evaluator._input_nodes_callbacks = []

        for input in self.inputs.iter_all():
            node = input.parent_node
            if node not in self._evaluator._input_nodes_callbacks:
                self._evaluator._input_nodes_callbacks.append(node.touch)

    def update_types(self, update_parents=True):
        return self._structure.update_types(update_parents)

    def allocate(self, allocate_parents=True):
        return self._structure.allocate(allocate_parents)

    def close(self, *, close_parents=True, strict=True, close_children=False, together=[]):
        return self._structure.close(
            close_parents=close_parents,
            strict=strict,
            close_children=close_children,
            together=together,
        )

    def open(self, *, open_children=False, force_taint=False):
        return self._structure.open(open_children=open_children, force_taint=force_taint)
