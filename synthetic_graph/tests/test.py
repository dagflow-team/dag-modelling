"""
Пример использования Cython DAG (Directed Acyclic Graph) системы
для создания вычислительных графов с различными операциями.
"""

from synthetic_graph.graph_cython import Input, Sum, Product, MatrixProduct, Sin, SumDoubleInt

def example_basic_arithmetic():
    """Базовые арифметические операции"""
    print("=== Базовые арифметические операции ===")
    
    # Создаем входные данные
    a = Input([1.0, 2.0, 3.0])
    b = Input([4.0, 5.0, 6.0])
    c = Input([2.0, 2.0, 2.0])
    
    # Создаем узлы операций
    sum_node = Sum(shape=[3])
    product_node = Product(shape=[3])
    
    # Строим граф: (a + b) * c
    a >> sum_node
    b >> sum_node
    sum_node >> product_node
    c >> product_node
    
    # Выполняем вычисления
    product_node.to_c_struct()
    result = product_node.run()
    print(f"(a + b) * c = {result}")
    print(f"Ожидаемый результат: [10.0, 14.0, 18.0]")
    print()

def example_matrix_operations():
    """Операции с матрицами"""
    print("=== Операции с матрицами ===")
    
    # Создаем матрицы
    matrix_a = Input([[1, 2], [3, 4]], data_type=1)
    matrix_b = Input([[5, 6], [7, 8]], data_type=1)
    
    # Матричное произведение
    mat_product = MatrixProduct(data_type=1, shape=[2, 2])
    
    matrix_a >> mat_product
    matrix_b >> mat_product
    
    mat_product.to_c_struct()
    result = mat_product.run()
    print(f"Матричное произведение A * B:")
    print(f"A = [[1, 2], [3, 4]]")
    print(f"B = [[5, 6], [7, 8]]")
    print(f"A * B = {result}")
    print(f"Ожидаемый результат: [[19, 22], [43, 50]]")
    print()

def example_trigonometric():
    """Тригонометрические функции"""
    print("=== Тригонометрические функции ===")
    
    # Создаем входные данные (углы в радианах)
    angles = Input([0.0, 1.5708, 3.14159])  # 0, π/2, π
    
    # Применяем синус
    sin_node = Sin(shape=[3])
    angles >> sin_node
    
    sin_node.to_c_struct()
    result = sin_node.run()
    print(f"sin(angles) = {result}")
    print(f"Углы: [0, π/2, π]")
    print(f"Ожидаемый результат: [0.0, 1.0, 0.0] (приблизительно)")
    print()

def example_mixed_types():
    """Операции со смешанными типами данных"""
    print("=== Операции со смешанными типами ===")
    
    # Double и int
    double_input = Input([1.5, 2.5, 3.5], data_type=3)
    int_input = Input([1, 2, 3], data_type=1)
    
    # Специальная функция для сложения double и int
    mixed_sum = SumDoubleInt(data_type=3, shape=[3])
    
    double_input >> mixed_sum
    int_input >> mixed_sum
    
    mixed_sum.to_c_struct()
    result = mixed_sum.run()
    print(f"Сложение double + int = {result}")
    print(f"Double: [1.5, 2.5, 3.5]")
    print(f"Int: [1, 2, 3]")
    print(f"Результат: {result}")
    print()

def example_complex_graph():
    """Сложный вычислительный граф"""
    print("=== Сложный вычислительный граф ===")
    
    # Входные данные
    x = Input([1.0, 2.0])
    y = Input([3.0, 4.0])
    z = Input([0.5, 0.5])
    
    # Промежуточные вычисления
    sum1 = Sum(shape=[2])  # x + y
    product1 = Product(shape=[2])  # (x + y) * z
    sin1 = Sin(shape=[2])  # sin((x + y) * z)
    
    # Строим граф: sin((x + y) * z)
    x >> sum1
    y >> sum1
    sum1 >> product1
    z >> product1
    product1 >> sin1
    
    sin1.to_c_struct()
    result = sin1.run()
    print(f"sin((x + y) * z) = {result}")
    print(f"x = [1.0, 2.0]")
    print(f"y = [3.0, 4.0]")
    print(f"z = [0.5, 0.5]")
    print(f"Результат: {result}")
    print()

def example_chaining_operations():
    """Цепочка операций с использованием оператора >>"""
    print("=== Цепочка операций ===")
    
    # Создаем цепочку: Input -> Sin -> Product
    input_data = Input([0.0, 1.0, 2.0])
    multiplier = Input([2.0, 2.0, 2.0])
    
    sin_node = Sin(shape=[3])
    product_node = Product(shape=[3])
    
    # Используем оператор >> для создания цепочки
    input_data >> sin_node
    sin_node >> product_node
    multiplier >> product_node
    
    product_node.to_c_struct()
    result = product_node.run()
    print(f"sin(input) * multiplier = {result}")
    print(f"Input: [0.0, 1.0, 2.0]")
    print(f"Multiplier: [2.0, 2.0, 2.0]")
    print(f"Результат: {result}")
    print()

def example_different_shapes():
    """Работа с различными размерностями"""
    print("=== Различные размерности ===")
    
    # Скаляр
    scalar = Input(5.0)
    scalar.to_c_struct()
    print(f"Скаляр: {scalar.run()}")
    
    # Вектор
    vector = Input([1.0, 2.0, 3.0])
    vector.to_c_struct()
    print(f"Вектор: {vector.run()}")
    
    # Матрица 2x3
    matrix = Input([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    matrix.to_c_struct()
    print(f"Матрица 2x3: {matrix.run()}")
    
    # 3D тензор
    tensor = Input([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], data_type=1)
    tensor.to_c_struct()
    print(f"3D тензор: {tensor.run()}")
    print()

if __name__ == "__main__":
    try:
        # Выполняем все примеры
        example_basic_arithmetic()
        example_matrix_operations()
        example_trigonometric()
        example_mixed_types()
        example_complex_graph()
        example_chaining_operations()
        example_different_shapes()
        
        print("Все примеры выполнены успешно!")
        
    except Exception as e:
        print(f"Ошибка при выполнении примера: {e}")
        import traceback
        traceback.print_exc()

    # Дополнительный пример для демонстрации производительности
    print("\n=== Тест производительности ===")
    import time
    
    # Создаем большие массивы
    large_data1 = Input([float(i) for i in range(10000)])
    large_data2 = Input([float(i+1) for i in range(10000)])
    
    sum_large = Sum(shape=[10000])
    large_data1 >> sum_large
    large_data2 >> sum_large
    
    sum_large.to_c_struct()
    start_time = time.time()
    result = sum_large.run()
    end_time = time.time()
    
    print(f"Время выполнения для массивов размером 10000: {end_time - start_time:.4f} секунд")
    print(f"Первые 5 элементов результата: {result[:5]}")
    print(f"Последние 5 элементы результата: {result[-5:]}")
