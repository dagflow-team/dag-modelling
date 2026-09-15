# cython: language_level=3


cdef class Input:
    # Data
    cdef public object _own_data        # NDArray | None
    cdef public object _own_dd          # DataDescriptor

    # Graph references (object to avoid circular cimport with Output)
    cdef public object _node            # NodeBase | None
    cdef public object _name            # str | None
    cdef public object _parent_output   # Output | None
    cdef public object _child_output    # Output | None

    # Flags (C-level bint for fast access)
    cdef public bint _allocatable
    cdef public bint _owns_buffer
    cdef public bint _debug

    # Hot-path methods
    cpdef void touch_above_threshold(self, int taint_threshold)
    cpdef void taint(self, bint force_taint=*, bint force_computation=*, object caller=*)
    cpdef taint_below_threshold(self, int taint_threshold=*, bint force_taint=*, bint force_computation=*, object caller=*)
