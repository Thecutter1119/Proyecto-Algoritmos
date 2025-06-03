from complexity_predictor import ComplexityPredictor
from combined_dataset_generator import generate_dataset
import numpy as np

def validate_code(code):
    """Valida el código de entrada"""
    try:
        compile(code, '<string>', 'exec')
        return True
    except SyntaxError:
        return False

def test_predictor(predictor):
    test_cases = [
        # Algoritmo O(1)
        {
            "code": "def constant_time(arr):\n    return arr[0] if arr else None",
            "expected": {'O': 0, 'Ω': 0, 'Θ': 0}  
        },
        
        # Algoritmo O(log n)
        {
            "code": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
            "expected": {'O': 1, 'Ω': 0, 'Θ': 1}
        },
        
        # Algoritmo O(n)
        {
            "code": "def linear_search(arr, target):\n    for i in range(len(arr)):\n        if arr[i] == target:\n            return i\n    return -1",
            "expected": {'O': 2, 'Ω': 0, 'Θ': 2}
        },
        
        # Algoritmo O(n log n)
        {
            "code": "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n\ndef merge(left, right):\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i])\n            i += 1\n        else:\n            result.append(right[j])\n            j += 1\n    result.extend(left[i:])\n    result.extend(right[j:])\n    return result",
            "expected": {'O': 3, 'Ω': 3, 'Θ': 3}
        },
        
        # Algoritmo O(n²)
        {
            "code": "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        swapped = False\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n                swapped = True\n        if not swapped:\n            break\n    return arr",
            "expected": {'O': 4, 'Ω': 2, 'Θ': 4}
        }
    ]

    print("\nProbando predictor con diferentes algoritmos:\n")
    for i, test in enumerate(test_cases, 1):
        print(f"Test caso #{i}:")
        print("Código:")
        print(test['code'].strip())
        if not validate_code(test['code']):
            print("Error: El código tiene errores de sintaxis")
            continue
            
        try:
            prediction = predictor.predict(test['code'])
            if prediction:
                print("\nPredicciones de complejidad:")
                print(f"Complejidad O: {prediction['O']} (esperado: {test['expected']['O']})")
                print(f"Complejidad Ω: {prediction['Ω']} (esperado: {test['expected']['Ω']})")
                print(f"Complejidad Θ: {prediction['Θ']} (esperado: {test['expected']['Θ']})")
            print("\n" + "-"*50 + "\n")
        except Exception as e:
            print(f"Error durante la predicción: {str(e)}")
            print("\n" + "-"*50 + "\n")

def main():
    # Configuración
    vector_size = 100
    num_samples = 2000
    epochs = 15
    batch_size = 32

    # Crear el predictor
    print("Inicializando predictor de complejidad...")
    predictor = ComplexityPredictor(vector_size=vector_size)

    # Generar dataset de entrenamiento
    print("\nGenerando dataset de entrenamiento...")
    code_samples, o_labels, omega_labels, theta_labels = generate_dataset(num_samples=num_samples)

    # Dividir en conjunto de entrenamiento y prueba
    split_idx = int(0.8 * len(code_samples))
    train_samples = code_samples[:split_idx]
    train_o = o_labels[:split_idx]
    train_omega = omega_labels[:split_idx]
    train_theta = theta_labels[:split_idx]

    test_samples = code_samples[split_idx:]
    test_o = o_labels[split_idx:]
    test_omega = omega_labels[split_idx:]
    test_theta = theta_labels[split_idx:]

    # Entrenar el modelo
    print("\nEntrenando modelos...")
    predictor.train(
        train_samples,
        train_o,
        train_omega,
        train_theta,
        epochs=epochs,
        batch_size=batch_size
    )

    # Evaluar el modelo
    print("\nEvaluando modelos...")
    evaluation = predictor.evaluate(test_samples, test_o, test_omega, test_theta)
    
    print("\nResultados de la evaluación:")
    for complexity_type, metrics in evaluation.items():
        if complexity_type in ['O', 'Ω', 'Θ']:
            print(f"\nModelo {complexity_type}:")
            print(f"  - Pérdida: {metrics['loss']:.4f}")
            print(f"  - Precisión: {metrics['accuracy']:.4f}")


    # Probar con casos de ejemplo
    test_predictor(predictor)

if __name__ == "__main__":
    main()