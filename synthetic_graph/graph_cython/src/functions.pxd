from types cimport NDArray

ctypedef NDArray* CFcnType(NDArray** inputs, int input_count, NDArray* data)

cdef CFcnType fcn_default
cdef CFcnType fcn_sum
cdef CFcnType fcn_sum_double_int
cdef CFcnType fcn_sum_double
cdef CFcnType fcn_product
cdef CFcnType fcn_matrix_product
cdef CFcnType fcn_sin