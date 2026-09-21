# cython: language_level=3, boundscheck=False, wraparound=False
from __future__ import annotations

from .node_eval_methods import _NodeEvalMethods


cdef class NodeEval:
    # cdef public fields declared in node_eval.pxd.

    def __init__(self, **kwargs):
        _NodeEvalMethods.__init__(self, **kwargs)

    def __str__(self):
        return _NodeEvalMethods.__str__(self)

    @property
    def children(self):
        return self._children

    @property
    def parents(self):
        return self._parents

    # Not on the hot path: touched only by graph allocate/close/open.
    taint_type = _NodeEvalMethods.taint_type
    reset_n_calls = _NodeEvalMethods.reset_n_calls

    cpdef void _invalidate(self, bint invalid):
        if invalid:
            self.invalidate()
        elif self._any_parent_invalid():
            return
        else:
            self.invalidate(False)
        self.invalidate_children(invalid)

    cdef bint _any_parent_invalid(self):
        for parent in self._parents.iter_all():
            if parent.invalid:
                return True
        return False

    cpdef void invalidate(self, bint invalid = True):
        self.invalid = invalid
        self.frozen_tainted = False
        self.frozen = False
        self.tainted = True

    cpdef void invalidate_children(self, bint invalid = True):
        for child in self._children:
            child.invalid = invalid

    cpdef void invalidate_parents(self, bint invalid = True):
        for parent in self._parents.iter_all():
            node = parent.parent_node
            node.invalidate(invalid)
            node.invalidate_parents(invalid)

    cpdef void freeze(self):
        self.frozen = True
        self.frozen_tainted = False

    cpdef bint unfreeze(self):
        if not self.frozen:
            return False
        self.frozen = False
        if self.frozen_tainted:
            self.frozen_tainted = False
            return True
        return False

    cpdef bint tainted_above_threshold(self, int taint_threshold = 0):
        return self.tainted_value > taint_threshold

    cpdef void taint_children(self, bint force_taint = False, bint force_computation = False, object caller = None):
        for child in self._children:
            child.taint_children(
                force_taint=force_taint, force_computation=force_computation, caller=caller
            )

    cpdef void taint_children_below_threshold(self, int taint_threshold = 0, bint force_taint = False, bint force_computation = False, object caller = None):
        for child in self.children:
            child.taint_children_below_threshold(
                taint_threshold,
                force_taint=force_taint,
                force_computation=force_computation,
                caller=caller,
            )

    cpdef void _eval_implementation(self):
        self._n_calls += 1
        self.function()

    cpdef void _touch_implementation(self):
        self._n_calls += 1
        self.function()
        self.tainted = False

    cpdef void _touch_above_threshold_implementation(self, int taint_threshold):
        # for debug purposes
        # replaced function is expected to process taint_threshold
        self._n_calls += 1
        self._function_stub_with_threshold(taint_threshold)
        self.tainted_value = taint_threshold

    cpdef void _function_stub_with_threshold(self, int taint_threshold):
        """An empty function stub of the Node that touches parent nodes to
        start a recursive execution of a graph (without computations)"""
        for input in self.parents.iter_all():
            input.touch_above_threshold(taint_threshold)
