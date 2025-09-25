import cython
from libc.math cimport sin
from types cimport DataType, NDArray, NDArrayConfig


cdef NDArray* fcn_default(NDArray** inputs, int input_count, NDArray* data):
    return data

cdef NDArray* fcn_sum(NDArray** inputs, int input_count, NDArray* data):
    cdef int i, j, index
    cdef int total_size = data.total_size
    cdef int data_type = data.config.data_type
    
    if data_type == DataType.TYPE_INT:
        for index in range(total_size):
            for i in range(input_count):
                (<int*>data.data)[index] += (<int*>inputs[i].data)[index]
    elif data_type == DataType.TYPE_FLOAT:
        for index in range(total_size):
            for i in range(input_count):
                (<float*>data.data)[index] += (<float*>inputs[i].data)[index]
    else:  # TYPE_DOUBLE
        for index in range(total_size):
            for i in range(input_count):
                (<double*>data.data)[index] += (<double*>inputs[i].data)[index]
    
    return data


cdef NDArray* fcn_sum_double(NDArray** inputs, int input_count, NDArray* data):
    cdef int total_size = data.total_size
    cdef int i, j
    
    for j in range(total_size):
        for i in range(input_count):
            (<double*>data.data)[j] += (<double*>inputs[i].data)[j]
    
    return data


cdef NDArray* fcn_sum_double_int(NDArray** inputs, int input_count, NDArray* data):
    cdef int index
    cdef int total_size = data.total_size
    
    for index in range(total_size):
        (<double*>data.data)[index] += (<double*>inputs[0].data)[index]
        (<double*>data.data)[index] += (<int*>inputs[1].data)[index]
    
    return data


cdef NDArray* fcn_product(NDArray** inputs, int input_count, NDArray* data):
    cdef int i, index
    cdef int total_size = data.total_size
    cdef int data_type = data.config.data_type
    
    if data_type == DataType.TYPE_INT:
        for index in range(total_size):
            (<int*>data.data)[index] = 1
            for i in range(input_count):
                (<int*>data.data)[index] *= (<int*>inputs[i].data)[index]
    elif data_type == DataType.TYPE_FLOAT:
        for index in range(total_size):
            (<float*>data.data)[index] = 1.0
            for i in range(input_count):
                (<float*>data.data)[index] *= (<float*>inputs[i].data)[index]
    else:  # TYPE_DOUBLE
        for index in range(total_size):
            (<double*>data.data)[index] = 1.0
            for i in range(input_count):
                (<double*>data.data)[index] *= (<double*>inputs[i].data)[index]
    
    return data


cdef NDArray* fcn_matrix_product(NDArray** inputs, int input_count, NDArray* data):
    cdef int i, j, k
    cdef int data_type = data.config.data_type
    
    if input_count < 2:
        return data
    
    cdef int A_rows = inputs[0].config.shape[0]
    cdef int A_cols = inputs[0].config.shape[1] if inputs[0].config.ndims > 1 else 1
    cdef int B_rows = inputs[1].config.shape[0]
    cdef int B_cols = inputs[1].config.shape[1] if inputs[1].config.ndims > 1 else 1
    
    if A_cols != B_rows:
        return data
    
    if data_type == DataType.TYPE_INT:
        for i in range(A_rows):
            for j in range(B_cols):
                (<int*>data.data)[i * B_cols + j] = 0
                for k in range(A_cols):
                    (<int*>data.data)[i * B_cols + j] += (<int*>inputs[0].data)[i * A_cols + k] * (<int*>inputs[1].data)[k * B_cols + j]
    elif data_type == DataType.TYPE_FLOAT:
        for i in range(A_rows):
            for j in range(B_cols):
                (<float*>data.data)[i * B_cols + j] = 0.0
                for k in range(A_cols):
                    (<float*>data.data)[i * B_cols + j] += (<float*>inputs[0].data)[i * A_cols + k] * (<float*>inputs[1].data)[k * B_cols + j]
    else:  # TYPE_DOUBLE
        for i in range(A_rows):
            for j in range(B_cols):
                (<double*>data.data)[i * B_cols + j] = 0.0
                for k in range(A_cols):
                    (<double*>data.data)[i * B_cols + j] += (<double*>inputs[0].data)[i * A_cols + k] * (<double*>inputs[1].data)[k * B_cols + j]
    
    return data


cdef NDArray* fcn_sin(NDArray** inputs, int input_count, NDArray* data):
    cdef int index
    cdef int total_size = data.total_size
    cdef int data_type = data.config.data_type
    
    if data_type == DataType.TYPE_INT:
        for index in range(total_size):
            (<int*>data.data)[index] = <int>sin((<int*>inputs[0].data)[index])
    elif data_type == DataType.TYPE_FLOAT:
        for index in range(total_size):
            (<float*>data.data)[index] = <float>sin((<float*>inputs[0].data)[index])
    else:  # TYPE_DOUBLE
        for index in range(total_size):
            (<double*>data.data)[index] = sin((<double*>inputs[0].data)[index])
    
    return data