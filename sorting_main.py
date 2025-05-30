from sorting_predictor import SortingPredictor
from sorting_dataset_generator import generate_sorting_dataset, get_algorithm_names
import numpy as np

def test_sorting_predictor(predictor):
    """Prueba el predictor con algoritmos de ordenamiento conocidos"""
    test_cases = [
        {
            "name": "Bubble Sort",
            "code": """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

arr = [64, 34, 25, 12, 22, 11, 90]
sorted_arr = bubble_sort(arr.copy())
""",
            "expected": "bubble_sort"
        },
        {
            "name": "Merge Sort",
            "code": """
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

arr = [64, 34, 25, 12, 22, 11, 90]
sorted_arr = merge_sort(arr)
""",
            "expected": "merge_sort"
        },
        {
            "name": "Quick Sort",
            "code": """
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)

arr = [64, 34, 25, 12, 22, 11, 90]
sorted_arr = quick_sort(arr)
""",
            "expected": "quick_sort"
        },
        {
            "name": "Selection Sort",
            "code": """
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

arr = [64, 34, 25, 12, 22, 11, 90]
sorted_arr = selection_sort(arr.copy())
""",
            "expected": "selection_sort"
        }
    ]

    print("\n" + "="*60)
    print("PROBANDO PREDICTOR DE ALGORITMOS DE ORDENAMIENTO")
    print("="*60)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest #{i}: {test['name']}")
        print("-" * 40)
        
        try:
            prediction = predictor.predict(test['code'])
            if prediction:
                predicted_algorithm = prediction['algorithm']
                confidence = prediction['confidence']
                
                print(f"Algoritmo predicho: {predicted_algorithm}")
                print(f"Algoritmo esperado: {test['expected']}")
                print(f"Confianza: {confidence:.4f}")
                
                # Mostrar las 3 predicciones más probables
                sorted_probs = sorted(
                    prediction['all_probabilities'].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                print("\nTop 3 predicciones:")
                for alg, prob in sorted_probs[:3]:
                    print(f"  {alg}: {prob:.4f}")
                
                # Verificar si la predicción es correcta
                is_correct = predicted_algorithm == test['expected']
                print(f"✓ Correcto" if is_correct else "✗ Incorrecto")
            else:
                print("Error: No se pudo realizar la predicción")
                
        except Exception as e:
            print(f"Error durante la predicción: {str(e)}")
        
        print("-" * 40)

def main():
    """Función principal para entrenar y probar el predictor de ordenamiento"""
    # Configuración
    vector_size = 100
    num_samples = 1600  # 200 por cada algoritmo
    epochs = 25
    batch_size = 32

    print("="*60)
    print("PREDICTOR DE ALGORITMOS DE ORDENAMIENTO")
    print("="*60)

    # Crear el predictor
    print("\nInicializando predictor de algoritmos de ordenamiento...")
    predictor = SortingPredictor(vector_size=vector_size)

    # Generar dataset especializado
    print(f"\nGenerando dataset de {num_samples} muestras de algoritmos de ordenamiento...")
    code_samples, labels = generate_sorting_dataset(num_samples=num_samples)
    
    print(f"Dataset generado con {len(code_samples)} muestras")
    print("Algoritmos incluidos:")
    algorithm_names = get_algorithm_names()
    for i, name in enumerate(algorithm_names):
        count = labels.count(i)
        print(f"  - {name}: {count} muestras")

    # Dividir en entrenamiento y prueba
    split_idx = int(0.8 * len(code_samples))
    train_samples = code_samples[:split_idx]
    train_labels = labels[:split_idx]
    
    test_samples = code_samples[split_idx:]
    test_labels = labels[split_idx:]

    print(f"\nDivisión del dataset:")
    print(f"  - Entrenamiento: {len(train_samples)} muestras")
    print(f"  - Prueba: {len(test_samples)} muestras")

    # Entrenar el modelo
    print(f"\nEntrenando modelo por {epochs} épocas...")
    history = predictor.train(
        train_samples,
        train_labels,
        epochs=epochs,
        batch_size=batch_size
    )

    # Evaluar el modelo
    print("\nEvaluando modelo...")
    evaluation = predictor.evaluate(test_samples, test_labels)
    
    print(f"\nResultados de la evaluación:")
    print(f"  - Pérdida: {evaluation['loss']:.4f}")
    print(f"  - Precisión: {evaluation['accuracy']:.4f}")

    # Probar con casos específicos
    test_sorting_predictor(predictor)

    print("\n" + "="*60)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*60)

if __name__ == "__main__":
    main()