import numpy as np

def generate_linear_search():
    n = np.random.randint(5, 15)
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
    return code, {'O': 2, 'Ω': 0, 'Θ': 2}  # O(n), Ω(1), Θ(n)

def generate_binary_search():
    n = np.random.randint(5, 15)
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
    return code, {'O': 1, 'Ω': 0, 'Θ': 1}  # O(log n), Ω(1), Θ(log n)

def generate_bubble_sort():
    n = np.random.randint(5, 10)
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
    return code, {'O': 4, 'Ω': 2, 'Θ': 4}  # O(n²), Ω(n), Θ(n²)

def generate_quick_sort():
    n = np.random.randint(5, 10)
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
    return code, {'O': 4, 'Ω': 2, 'Θ': 3}  # O(n²), Ω(n log n), Θ(n log n)

def generate_dataset(num_samples=1000):
    generators = [
        generate_linear_search,
        generate_binary_search,
        generate_bubble_sort,
        generate_quick_sort
    ]
    
    code_samples = []
    o_labels = []
    omega_labels = []
    theta_labels = []
    
    for _ in range(num_samples):
        generator = np.random.choice(generators)
        code, complexities = generator()
        
        code_samples.append(code)
        o_labels.append(complexities['O'])
        omega_labels.append(complexities['Ω'])
        theta_labels.append(complexities['Θ'])
    
    return code_samples, o_labels, omega_labels, theta_labels