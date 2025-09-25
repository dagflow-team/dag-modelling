from functions cimport CFcnType
from types cimport NDArray, NDArrayConfig


ctypedef struct CNode:
    CNode** inputs
    int input_count

    NDArray* data

    CFcnType* fcn
    int tainted


cdef class Node:
    cdef public list inputs
    cdef CNode* cnode
    cdef CFcnType* fcn
    cdef list shape
    cdef int data_type
    cdef int initialized

    cdef CNode* _to_c_struct(self)
    cdef void _setup_cnode(self)
    cdef NDArray* _create_data(self)
    cdef NDArrayConfig* _create_config(self)
    @staticmethod
    cdef void* _run(CNode* node)
    @staticmethod
    cdef void* _allocate_data(int data_type, int data_size)
    @staticmethod
    cdef list _convert_to_python(NDArray* data)
    @staticmethod
    cdef list _reshape_list(list flat_list, list shape)

