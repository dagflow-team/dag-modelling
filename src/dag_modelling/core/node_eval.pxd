# cython: language_level=3


cdef class NodeEval:
    # Flags checked on the hot taint/touch path
    cdef public bint tainted
    cdef public int tainted_value
    cdef public bint frozen
    cdef public bint frozen_tainted
    cdef public bint invalid
    cdef public bint closed
    cdef public bint allocated

    # Flags touched only during graph lifecycle. Still cdef: a cdef class
    # has no __dict__, so every field needs a C-level declaration regardless.
    cdef public bint needs_reallocation
    cdef public bint needs_post_allocate
    cdef public bint types_tainted

    # Graph references (Python objects)
    cdef public object _children   # Outputs
    cdef public object _parents    # Inputs

    # Execution
    cdef public object function    # Callable
    cdef public dict _functions_dict
    cdef public int _n_calls
    cdef public list _input_nodes_callbacks
    cdef public bint _immediate

    # Methods
    cpdef void _invalidate(self, bint invalid)
    cdef bint _any_parent_invalid(self)
    cpdef void invalidate(self, bint invalid=*)
    cpdef void invalidate_children(self, bint invalid=*)
    cpdef void invalidate_parents(self, bint invalid=*)
    cpdef void freeze(self)
    cpdef bint unfreeze(self)
    cpdef bint tainted_above_threshold(self, int taint_threshold=*)
    cpdef void taint_children(self, bint force_taint=*, bint force_computation=*, object caller=*)
    cpdef void taint_children_below_threshold(self, int taint_threshold=*, bint force_taint=*, bint force_computation=*, object caller=*)
    cpdef void _eval_implementation(self)
    cpdef void _touch_implementation(self)
    cpdef void _touch_above_threshold_implementation(self, int taint_threshold)
    cpdef void _function_stub_with_threshold(self, int taint_threshold)
