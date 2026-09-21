# cython: language_level=3

from .input cimport Input


cdef class Output:
    # Data
    cdef public object _data            # NDArray | None
    cdef public object _data_ro         # NDArray | None
    cdef public object _dd              # DataDescriptor
    cdef public object _labels          # Labels | None

    # Graph references
    cdef public object _node            # Node | None
    cdef public object _name            # str | None
    cdef public list _child_inputs      # list[Input]
    cdef public object _parent_input    # Input | None
    cdef public object _allocating_input  # Input | None

    # Flags (C-level bint for fast access)
    cdef public bint _allocatable
    cdef public bint _owns_buffer
    cdef public bint _forbid_reallocation
    cdef public bint _debug

    # Hot-path methods
    cpdef void taint_children(self, bint force_taint=*, bint force_computation=*, object caller=*)
    cpdef void taint_children_below_threshold(self, int taint_threshold=*, bint force_taint=*, bint force_computation=*, object caller=*)
    cpdef touch(self, bint force_computation=*)
    cpdef void touch_above_threshold(self, int taint_threshold, bint force_computation=*)
