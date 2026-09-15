# cython: language_level=3


# Must be a cdef class: Node (cdef class) inherits from it, and Cython
# requires the first base of a cdef class to be an extension type.
cdef class NodeBase:
    cdef public object inputs          # Inputs
    cdef public object outputs         # Outputs
    cdef public object _input_strategy # InputStrategyBase
