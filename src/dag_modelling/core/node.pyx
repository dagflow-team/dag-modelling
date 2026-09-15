# cython: language_level=3, boundscheck=False, wraparound=False
from __future__ import annotations

from weakref import ref as weakref

from .exception import DagModellingError
from .node_eval cimport NodeEval
from .node_base cimport NodeBase
from .node_methods import _NodeMethods

from .exception import UnclosedGraphError

cdef class Node(NodeBase):
    # cdef public fields declared in node.pxd.

    def __init__(self, *args, **kwargs):
        # NodeBase.__init__ is in NodeBase's own class dict; MRO would
        # route here first. Bypass explicitly.
        _NodeMethods.__init__(self, *args, **kwargs)

    def __str__(self):
        return _NodeMethods.__str__(self)

    def __call__(self, *args, **kwargs):
        return _NodeMethods.__call__(self, *args, **kwargs)

    # Cold non-dunder API via class-attribute binding.
    choose_function = _NodeMethods.choose_function
    # classmethods bound via vars() to keep the raw descriptor: plain
    # `_NodeMethods.from_args` access would resolve the descriptor and
    # store a method bound to _NodeMethods rather than to Node.
    from_args = vars(_NodeMethods)["from_args"]
    replicate = vars(_NodeMethods)["replicate"]

    _make_input = _NodeMethods._make_input
    add_input = _NodeMethods.add_input
    _add_inputs = _NodeMethods._add_inputs
    _add_input = _NodeMethods._add_input
    add_output = _NodeMethods.add_output
    _add_output = _NodeMethods._add_output
    _add_outputs = _NodeMethods._add_outputs
    add_pair = _NodeMethods.add_pair
    _add_pair = _NodeMethods._add_pair
    _add_pairs = _NodeMethods._add_pairs

    _function = _NodeMethods._function
    eval = _NodeMethods.eval
    _eval = _NodeMethods._eval
    freeze = _NodeMethods.freeze
    print = _NodeMethods.print
    to_dict = _NodeMethods.to_dict
    _type_function = _NodeMethods._type_function
    _post_allocate = _NodeMethods._post_allocate
    update_types = _NodeMethods.update_types
    allocate = _NodeMethods.allocate
    close = _NodeMethods.close
    open = _NodeMethods.open
    invalidate = _NodeMethods.invalidate
    invalidate_children = _NodeMethods.invalidate_children
    invalidate_parents = _NodeMethods.invalidate_parents
    label = _NodeMethods.label
    # Not on the hot path: touched only by graph allocate/close/open.
    taint_type = _NodeMethods.taint_type

    #
    # Properties - kept inline (not in _NodeMethods) so Cython compiles
    # them into the tp_getset slot instead of a Python descriptor.
    #

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

    @property
    def n_calls(self):
        return self._evaluator._n_calls

    @property
    def structure(self):
        return self._structure

    #
    # Hot path
    #

    cpdef touch(self, bint force_computation=False):
        if not force_computation:
            if not self._evaluator.tainted:
                return
            if not self._evaluator.closed:
                from .exception import UnclosedGraphError
                raise UnclosedGraphError("Cannot evaluate not closed node!", node=self)
        self._evaluator._touch_implementation()

    cpdef touch_above_threshold(self, int taint_threshold = 0, bint force_computation=False):
        if not force_computation:
            if not self._evaluator.tainted_above_threshold(taint_threshold):
                return
            if not self.closed:
                raise UnclosedGraphError("Cannot evaluate not closed node!", node=self)
        self._evaluator._touch_above_threshold_implementation(taint_threshold)

    cpdef unfreeze(self):
        if self._evaluator.unfreeze():
            self.taint()

    cpdef taint(
        self,
        bint force_taint=False,
        bint force_computation=False,
        object caller=None,
    ):
        if self._evaluator.tainted and not force_taint:
            return
        if self._evaluator.frozen:
            self._evaluator.frozen_tainted = True
            return

        self._evaluator.tainted = True
        if self._evaluator._immediate or force_computation:
            self._evaluator._touch_implementation()
        self._evaluator.taint_children(
            force_taint=force_taint, force_computation=force_computation, caller=caller
        )

    cpdef taint_below_threshold(
        self,
        int taint_threshold = 0,
        bint force_taint = False,
        bint force_computation = False,
        object caller = None,
    ):
        if self._evaluator.tainted_value > taint_threshold and not force_taint:
            return
        if self.frozen:
            self._evaluator.frozen_tainted = True
            return

        self._evaluator.tainted_value = taint_threshold + 1
        if self._evaluator._immediate or force_computation:
            self._evaluator._touch_above_threshold_implementation(taint_threshold)
        self._evaluator.taint_children_below_threshold(
            taint_threshold, force_taint=force_taint, force_computation=force_computation, caller=caller
        )

    cpdef taint_children(
        self,
        bint force_taint=False,
        bint force_computation=False,
        object caller=None,
    ):
        self._evaluator.taint_children(
            force_taint=force_taint, force_computation=force_computation, caller=caller
        )
