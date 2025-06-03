import numpy as np

def generate_constant_time():
    """
    Genera un código de función con complejidad O(1).
    """
    code = """
# Tiempo constante
def constant_time(arr):
    return arr[0] if arr else None

# Ejemplo de uso
arr = [1, 2, 3]
result = constant_time(arr)
"""
    return code, {'O': 0, 'Ω': 0, 'Θ': 0}

def generate_linear_search():
    """
    Genera un código de función con búsqueda lineal y etiquetas correspondientes.
    """
    n = np.random.randint(5, 20)
    arr = np.random.randint(1, 100, size=n).tolist()
    target = np.random.choice(arr)

    code = f"""
# Búsqueda lineal
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1

# Ejemplo de uso
arr = {arr}
target = {target}
result = linear_search(arr, target)
"""
    return code, {'O': 2, 'Ω': 0, 'Θ': 2}

def generate_binary_search():
    """
    Genera un código de función con búsqueda binaria y etiquetas correspondientes.
    """
    n = np.random.randint(5, 20)
    arr = sorted(np.random.randint(1, 100, size=n).tolist())
    target = np.random.choice(arr)

    code = f"""
# Búsqueda binaria
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# Ejemplo de uso
arr = {arr}
target = {target}
result = binary_search(arr, target)
"""
    return code, {'O': 1, 'Ω': 0, 'Θ': 1}

def generate_merge_sort():
    """
    Genera un código de función con merge sort y etiquetas correspondientes.
    """
    n = np.random.randint(5, 15)
    arr = np.random.randint(1, 100, size=n).tolist()

    code = f"""
# Merge Sort
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

# Ejemplo de uso
arr = {arr}
sorted_arr = merge_sort(arr)
"""
    return code, {'O': 3, 'Ω': 3, 'Θ': 3}

def generate_bubble_sort():
    """
    Genera un código de función con bubble sort y etiquetas correspondientes.
    """
    n = np.random.randint(5, 15)
    arr = np.random.randint(1, 100, size=n).tolist()

    code = f"""
# Ordenamiento burbuja
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

# Ejemplo de uso
arr = {arr}
sorted_arr = bubble_sort(arr.copy())
"""
    # CORREGIDO: Bubble sort es O(n²) en todos los casos sin optimización
    return code, {'O': 4, 'Ω': 4, 'Θ': 4}

def generate_optimized_bubble_sort():
    """
    Genera un código de función con bubble sort optimizado.
    """
    n = np.random.randint(5, 15)
    arr = np.random.randint(1, 100, size=n).tolist()

    code = f"""
# Ordenamiento burbuja optimizado
def bubble_sort_optimized(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr

# Ejemplo de uso
arr = {arr}
sorted_arr = bubble_sort_optimized(arr.copy())
"""
    # Este SÍ tiene mejor caso O(n)
    return code, {'O': 4, 'Ω': 2, 'Θ': 4}

def generate_quick_sort():
    """
    Genera un código de función con quicksort y etiquetas correspondientes.
    """
    n = np.random.randint(5, 15)
    arr = np.random.randint(1, 100, size=n).tolist()

    code = f"""
# Quicksort
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

# Ejemplo de uso
arr = {arr}
sorted_arr = quicksort(arr)
"""
    return code, {'O': 4, 'Ω': 3, 'Θ': 3}

def generate_nested_loops():
    """
    Genera código con loops anidados simples O(n²).
    """
    n = np.random.randint(5, 15)
    
    code = f"""
# Loops anidados O(n²)
def nested_loops(n):
    result = []
    for i in range(n):
        for j in range(n):
            result.append(i * j)
    return result

# Ejemplo de uso
n = {n}
result = nested_loops(n)
"""
    return code, {'O': 4, 'Ω': 4, 'Θ': 4}

def generate_dataset(num_samples=1000):
    """
    Genera un dataset de muestras de código con sus etiquetas de complejidad.
    """
    generators = [
        generate_constant_time,
        generate_linear_search,
        generate_binary_search,
        generate_merge_sort,  # AGREGADO
        generate_bubble_sort,
        generate_optimized_bubble_sort,  # AGREGADO
        generate_quick_sort,
        generate_nested_loops  # AGREGADO
    ]

    code_samples = []
    o_labels = []
    omega_labels = []
    theta_labels = []

    for _ in range(num_samples):
        generator = np.random.choice(generators)
        code, labels = generator()
        code_samples.append(code)
        o_labels.append(labels['O'])
        omega_labels.append(labels['Ω'])
        theta_labels.append(labels['Θ'])

    return code_samples, o_labels, omega_labels, theta_labels