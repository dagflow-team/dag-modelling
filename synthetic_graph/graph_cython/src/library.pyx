import cython

from node cimport Node
from functions cimport fcn_sum, fcn_sum_double, fcn_product, fcn_sin, fcn_sum_double_int, fcn_matrix_product
from types cimport DataType, NDArray


cdef class Input(Node):
    cdef public object py_data

    def __init__(
        self,
        data,
        data_type=DataType.TYPE_DOUBLE,
        shape=None
    ):
        self.py_data = data

        if shape is None:
            if isinstance(data, list):
                shape = self._get_shape_from_nested_list(data)
            elif hasattr(data, 'shape'):
                shape = list(data.shape)
            else:
                shape = [1]

        super().__init__(data_type, shape=shape)

    cdef void _setup_cnode(self):
        cdef NDArray* data = self._create_data()

        self._copy_data_to_array(data)

        self.cnode.inputs = cython.NULL
        self.cnode.input_count = 0
        self.cnode.data = data

    cdef void _copy_data_to_array(self, NDArray* array):
        cdef int i
        cdef int total_size = array.total_size
        
        flat_data = self._flatten_data(self.py_data)
        
        if self.data_type == DataType.TYPE_INT:
            for i in range(min(len(flat_data), total_size)):
                (<int*>array.data)[i] = int(flat_data[i])
        elif self.data_type == DataType.TYPE_FLOAT:
            for i in range(min(len(flat_data), total_size)):
                (<float*>array.data)[i] = float(flat_data[i])
        else:  # TYPE_DOUBLE
            for i in range(min(len(flat_data), total_size)):
                (<double*>array.data)[i] = float(flat_data[i])
    
    cdef list _get_shape_from_nested_list(self, data):
        shape = [len(data)]
        
        if data and isinstance(data[0], list):
            shape.extend(self._get_shape_from_nested_list(data[0]))
        
        return shape
    
    cdef list _flatten_data(self, data):
        if not isinstance(data, list) and not hasattr(data, 'shape'):
            return [data]
        
        flat_list = []
        if hasattr(data, 'shape'):
            import numpy as np
            return list(np.asarray(data).flatten())
        
        for item in data:
            if isinstance(item, list):
                flat_list.extend(self._flatten_data(item))
            else:
                flat_list.append(item)
                
        return flat_list

cdef class Sum(Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fcn = fcn_sum


cdef class SumDouble(Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fcn = fcn_sum_double

cdef class Product(Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fcn = fcn_product


cdef class MatrixProduct(Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fcn = fcn_matrix_product


cdef class SumDoubleInt(Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fcn = fcn_sum_double_int


cdef class Sin(Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fcn = fcn_sin