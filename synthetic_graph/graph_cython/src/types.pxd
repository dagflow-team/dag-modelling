cdef enum DataType:
    TYPE_INT = 1
    TYPE_FLOAT = 2
    TYPE_DOUBLE = 3

ctypedef struct NDArrayConfig:
    int* shape
    int ndims
    int data_type

ctypedef struct NDArray:
    void* data

    NDArrayConfig* config
    
    int* strides
    int total_size