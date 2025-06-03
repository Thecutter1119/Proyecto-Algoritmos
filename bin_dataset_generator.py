import numpy as np

def generate_arbol_binario():
    valores = np.random.randint(1, 100, size=7).tolist()

    code = f"""
# Definición del nodo del árbol binario
class Nodo:
    def __init__(self, valor):
        self.valor = valor
        self.izquierda = None
        self.derecha = None

# Insertar nodos en un árbol binario simple
def insertar(nodo, valor):
    if nodo is None:
        return Nodo(valor)
    if valor < nodo.valor:
        nodo.izquierda = insertar(nodo.izquierda, valor)
    else:
        nodo.derecha = insertar(nodo.derecha, valor)
    return nodo

# Recorrido inorden (O(n))
def inorden(nodo):
    if nodo:
        inorden(nodo.izquierda)
        print(nodo.valor)
        inorden(nodo.derecha)

# Construcción del árbol y recorrido
valores = {valores}
raiz = None
for v in valores:
    raiz = insertar(raiz, v)

inorden(raiz)
"""
    return code, {'O': 3, 'Ω': 3, 'Θ': 3}  # O(n log n), Ω(n log n), Θ(n log n)


def generate_pilas():
    n = np.random.randint(5, 15)
    arr = np.random.randint(1, 100, size=n).tolist()

    code = f"""# Definición de la clase Pila
class Pila:
    def __init__(self):
        self.items = []

    def apilar(self, item):
        self.items.append(item)  # Añade un elemento al tope de la pila

    def desapilar(self):
        if not self.esta_vacia():
            return self.items.pop()  # Remueve y retorna el elemento tope
        return None

    def esta_vacia(self):
        return len(self.items) == 0

    def ver_tope(self):
        if not self.esta_vacia():
            return self.items[-1]  # Retorna el elemento tope sin removerlo
        return None

    def __str__(self):
        return str(self.items)

# Simulación de uso
pila = Pila()
arr = {arr}
for x in arr:
    pila.apilar(x)
while not pila.esta_vacia():
    pila.desapilar()
"""
    return code, {'O': 2, 'Ω': 2, 'Θ': 2}

def generate_colas():
    n = np.random.randint(5, 15)
    arr = np.random.randint(1, 100, size=n).tolist()

    code = f"""from collections import deque

class Cola:
    def __init__(self):
        self.items = deque()  # Inicializamos la cola

    def encolar(self, item):
        self.items.append(item)  # Añade un elemento al final de la cola

    def desencolar(self):
        if not self.esta_vacia():
            return self.items.popleft()  # Remueve y retorna el primer elemento
        return None

    def esta_vacia(self):
        return len(self.items) == 0

    def frente(self):
        if not self.esta_vacia():
            return self.items[0]  # Retorna el primer elemento sin removerlo
        return None

    def __str__(self):
        return str(list(self.items))

# Ejemplo de uso
cola = Cola()
arr = {arr}
for x in arr:
    cola.encolar(x)
while not cola.esta_vacia():
    cola.desencolar()
"""
    return code, {'O': 2, 'Ω': 2, 'Θ': 2}
 
    
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
    return code, {'O': 2, 'Ω': 2, 'Θ': 2}  # O(n), Ω(1), Θ(n)

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
    return code, {'O': 3, 'Ω': 3, 'Θ': 3}  # O(n log n), Ω(n log n), Θ(n log n)

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
    return code, {'O': 4, 'Ω': 4, 'Θ': 4}  # O(n²)

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
    return code, {'O': 2, 'Ω': 2, 'Θ': 2}  # O(n²)

def generate_dataset(num_samples=1000):
    generators = [
        generate_linear_search,
        generate_binary_search,
        generate_bubble_sort,
        generate_quick_sort,
        generate_colas,
        generate_pilas,
        generate_arbol_binario 

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