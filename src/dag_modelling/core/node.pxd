# cython: language_level=3

from .node_eval cimport NodeEval
from .node_base cimport NodeBase


cdef class Node(NodeBase):
    # Slots — mirror node.py __slots__ (9 items)
    cdef public object _name
    cdef public object _labels
    cdef public object _graph
    cdef public object _logger
    cdef public object _exception
    cdef public dict _internal_buffers
    cdef public object _meta_node
    cdef public bint _debug
    cdef public NodeEval _evaluator
    cdef public object _structure

    # Hot-path methods
    cpdef touch(self, bint force_computation=*)
    cpdef touch_above_threshold(self, int taint_threshold=*, bint force_computation=*)
    cpdef unfreeze(self)
    cpdef taint(self, bint force_taint=*, bint force_computation=*, object caller=*)
    cpdef taint_below_threshold(self, int taint_threshold=*, bint force_taint=*, bint force_computation=*, object caller=*)
    cpdef taint_children(self, bint force_taint=*, bint force_computation=*, object caller=*)
