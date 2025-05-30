import numpy as np

def generate_bubble_sort():
    n = np.random.randint(5, 10)
    arr = np.random.randint(1, 100, size=n).tolist()
    
    code = f"""
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

arr = {arr}
sorted_arr = bubble_sort(arr.copy())
"""
    return code, 0  # índice para bubble_sort

def generate_selection_sort():
    n = np.random.randint(5, 10)
    arr = np.random.randint(1, 100, size=n).tolist()
    
    code = f"""
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

arr = {arr}
sorted_arr = selection_sort(arr.copy())
"""
    return code, 1  # índice para selection_sort

def generate_insertion_sort():
    n = np.random.randint(5, 10)
    arr = np.random.randint(1, 100, size=n).tolist()
    
    code = f"""
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

arr = {arr}
sorted_arr = insertion_sort(arr.copy())
"""
    return code, 2  # índice para insertion_sort

def generate_merge_sort():
    n = np.random.randint(5, 10)
    arr = np.random.randint(1, 100, size=n).tolist()
    
    code = f"""
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

arr = {arr}
sorted_arr = merge_sort(arr.copy())
"""
    return code, 3  # índice para merge_sort

def generate_quick_sort():
    n = np.random.randint(5, 10)
    arr = np.random.randint(1, 100, size=n).tolist()
    
    code = f"""
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)

arr = {arr}
sorted_arr = quick_sort(arr.copy())
"""
    return code, 4  # índice para quick_sort

def generate_heap_sort():
    n = np.random.randint(5, 10)
    arr = np.random.randint(1, 100, size=n).tolist()
    
    code = f"""
def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2
    
    if left < n and arr[left] > arr[largest]:
        largest = left
    
    if right < n and arr[right] > arr[largest]:
        largest = right
    
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def heap_sort(arr):
    n = len(arr)
    
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)
    
    return arr

arr = {arr}
sorted_arr = heap_sort(arr.copy())
"""
    return code, 5  # índice para heap_sort

def generate_counting_sort():
    n = np.random.randint(5, 10)
    max_val = 20  # Limitamos el rango para counting sort
    arr = np.random.randint(0, max_val, size=n).tolist()
    
    code = f"""
def counting_sort(arr):
    if not arr:
        return arr
    
    max_val = max(arr)
    count = [0] * (max_val + 1)
    
    for num in arr:
        count[num] += 1
    
    result = []
    for i in range(len(count)):
        result.extend([i] * count[i])
    
    return result

arr = {arr}
sorted_arr = counting_sort(arr.copy())
"""
    return code, 6  # índice para counting_sort

def generate_radix_sort():
    n = np.random.randint(5, 10)
    arr = np.random.randint(1, 1000, size=n).tolist()
    
    code = f"""
def counting_sort_for_radix(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10
    
    for i in range(n):
        index = arr[i] // exp
        count[index % 10] += 1
    
    for i in range(1, 10):
        count[i] += count[i - 1]
    
    i = n - 1
    while i >= 0:
        index = arr[i] // exp
        output[count[index % 10] - 1] = arr[i]
        count[index % 10] -= 1
        i -= 1
    
    for i in range(n):
        arr[i] = output[i]

def radix_sort(arr):
    max_num = max(arr)
    exp = 1
    
    while max_num // exp > 0:
        counting_sort_for_radix(arr, exp)
        exp *= 10
    
    return arr

arr = {arr}
sorted_arr = radix_sort(arr.copy())
"""
    return code, 7  # índice para radix_sort

def generate_sorting_dataset(num_samples=1000):
    """Genera un dataset especializado en algoritmos de ordenamiento"""
    generators = [
        generate_bubble_sort,
        generate_selection_sort,
        generate_insertion_sort,
        generate_merge_sort,
        generate_quick_sort,
        generate_heap_sort,
        generate_counting_sort,
        generate_radix_sort
    ]
    
    code_samples = []
    labels = []
    
    # Distribución equilibrada de algoritmos
    samples_per_algorithm = num_samples // len(generators)
    
    for generator in generators:
        for _ in range(samples_per_algorithm):
            code, label = generator()
            code_samples.append(code)
            labels.append(label)
    
    # Agregar muestras adicionales aleatoriamente para completar
    remaining_samples = num_samples - len(code_samples)
    for _ in range(remaining_samples):
        generator = np.random.choice(generators)
        code, label = generator()
        code_samples.append(code)
        labels.append(label)
    
    # Mezclar el dataset
    indices = np.random.permutation(len(code_samples))
    code_samples = [code_samples[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    return code_samples, labels

def get_algorithm_names():
    """Retorna los nombres de los algoritmos en orden"""
    return [
        'bubble_sort',
        'selection_sort', 
        'insertion_sort',
        'merge_sort',
        'quick_sort',
        'heap_sort',
        'counting_sort',
        'radix_sort'
    ]