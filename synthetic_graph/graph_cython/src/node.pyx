import cython
from libc.stdlib cimport malloc, free, calloc

from functions cimport fcn_default
from types cimport DataType


cdef class Node:
    def __init__(self, data_type=DataType.TYPE_DOUBLE, shape=None, inputs=None):
        self.inputs = inputs if inputs else []
        self.data_type = data_type
        self.shape = shape if shape else [1]

        self.cnode = <CNode*>malloc(sizeof(CNode))
        self.initialized = 0

        self.fcn = fcn_default

    def __rshift__(self, other):
        other.inputs.append(self)
        return other

    def to_c_struct(self):
        self._to_c_struct()

    cdef CNode* _to_c_struct(self):
        if self.initialized:
            return self.cnode

        self._setup_cnode()
        self.cnode.fcn = self.fcn
        self.cnode.tainted = 1
        self.initialized = 1

        return self.cnode

    cdef void _setup_cnode(self):
        cdef int i
        cdef int input_count = len(self.inputs)
        cdef CNode** input_nodes = <CNode**>malloc(sizeof(CNode*) * input_count)

        cdef Node input_node
        for i in range(input_count):
            input_node = self.inputs[i]
            input_nodes[i] = <CNode*>input_node._to_c_struct()

        cdef NDArray* data = self._create_data()

        self.cnode.inputs = input_nodes
        self.cnode.input_count = input_count
        self.cnode.data = data

    cdef NDArray* _create_data(self):
        cdef NDArrayConfig* config = self._create_config()

        cdef NDArray* data = <NDArray*>malloc(sizeof(NDArray))
        data.config = config

        cdef int total_size = 1
        for i in range(config.ndims):
            total_size *= config.shape[i]
        
        data.total_size = total_size
        
        data.data = Node._allocate_data(config.data_type, total_size)
        data.strides = <int*>malloc(sizeof(int) * config.ndims)

        cdef int stride = 1
        for i in range(config.ndims - 1, -1, -1):
            data.strides[i] = stride
            stride *= config.shape[i]
        
        return data

    cdef NDArrayConfig* _create_config(self):
        cdef NDArrayConfig* config = <NDArrayConfig*>malloc(sizeof(NDArrayConfig))
        cdef int ndims = len(self.shape)
        
        config.ndims = ndims
        config.shape = <int*>malloc(sizeof(int) * ndims)
        config.data_type = self.data_type
        
        cdef int i
        for i in range(ndims):
            config.shape[i] = self.shape[i]
            
        return config

    @staticmethod
    cdef void* _run(CNode* node):
        if not node.tainted:
            return node.data

        cdef NDArray** input_results = <NDArray**>malloc(node.input_count * sizeof(NDArray*))
        cdef int i

        for i in range(node.input_count):
            input_results[i] = <NDArray*>Node._run(node.inputs[i])

        cdef NDArray* result = node.fcn(input_results, node.input_count, node.data)
        node.tainted = 0

        free(input_results)
        return result

    def run(self):
        cdef NDArray* result = <NDArray*>Node._run(self.cnode)
        return Node._convert_to_python(result)

    @staticmethod
    cdef list _convert_to_python(NDArray* array):
        cdef int i, j
        cdef NDArrayConfig* config = array.config
        cdef int* shape = config.shape
        cdef int ndims = config.ndims
        
        flat_list = []
        
        if config.data_type == DataType.TYPE_INT:
            for i in range(array.total_size):
                flat_list.append((<int*>array.data)[i])
        elif config.data_type == DataType.TYPE_FLOAT:
            for i in range(array.total_size):
                flat_list.append((<float*>array.data)[i])
        else:  # TYPE_DOUBLE
            for i in range(array.total_size):
                flat_list.append((<double*>array.data)[i])

        if ndims == 1:
            return flat_list
    
        cdef list py_shape = []
        for i in range(ndims):
            py_shape.append(shape[i])
    
        result = []
        step_size = 1
        for j in range(1, ndims):
            step_size *= py_shape[j]
        
        for i in range(0, min(len(flat_list), py_shape[0] * step_size), step_size):
            chunk = flat_list[i:i + step_size]
            result.append(Node._reshape_list(chunk, py_shape[1:]))
        
        return result

    @staticmethod
    cdef list _reshape_list(list flat_list, list shape):
        if len(shape) == 1:
            return flat_list
        
        result = []
        step_size = 1
        for dim in shape[1:]:
            step_size *= dim
        
        for i in range(0, len(flat_list), step_size):
            chunk = flat_list[i:i + step_size]
            result.append(Node._reshape_list(chunk, shape[1:]))
        
        return result

    @staticmethod
    cdef void* _allocate_data(int data_type, int data_size):
        if data_type == DataType.TYPE_INT:
            return calloc(data_size, sizeof(int))
        elif data_type == DataType.TYPE_FLOAT:
            return calloc(data_size, sizeof(float))
        else:
            return calloc(data_size, sizeof(double))

    def __dealloc__(self):
        if self.cnode:
            if self.cnode.data:
                if self.cnode.data.config:
                    if self.cnode.data.config.shape:
                        free(self.cnode.data.config.shape)
                    free(self.cnode.data.config)
                if self.cnode.data.strides:
                    free(self.cnode.data.strides)
                if self.cnode.data.data:
                    free(self.cnode.data.data)
                free(self.cnode.data)
            if self.cnode.inputs:
                free(self.cnode.inputs)
            free(self.cnode)
