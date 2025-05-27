# Predictor de Complejidad Algorítmica

Este proyecto implementa un sistema de predicción de complejidad algorítmica utilizando aprendizaje profundo. El sistema es capaz de analizar código fuente y predecir sus complejidades Big-O (O), Omega (Ω) y Theta (Θ).

## Características

- Predicción de tres tipos de complejidad algorítmica:
  - Complejidad Big-O (O)
  - Complejidad Omega (Ω)
  - Complejidad Theta (Θ)
- Utiliza redes neuronales para el análisis
- Genera automáticamente datasets de entrenamiento con algoritmos comunes
- Soporta diferentes tipos de algoritmos:
  - Búsqueda lineal
  - Búsqueda binaria
  - Ordenamiento burbuja
  - Quicksort

## Requisitos

- Python 3.x
- TensorFlow
- NumPy
- AST (Python Standard Library)

## Estructura del Proyecto

- `complexity_predictor.py`: Implementación principal del predictor de complejidad
- `dataset_generator.py`: Generador de datasets de entrenamiento
- `main.py`: Script principal para ejecutar el entrenamiento y predicción

## Uso

1. Asegúrate de tener todas las dependencias instaladas:

```bash
pip install tensorflow numpy
```

2. Ejecuta el script principal:

```bash
python main.py
```

Esto generará un dataset de entrenamiento, entrenará los modelos y realizará una predicción de ejemplo.

## Clases de Complejidad Soportadas

- O(1): Tiempo constante
- O(log n): Tiempo logarítmico
- O(n): Tiempo lineal
- O(n log n): Tiempo linearítmico
- O(n²): Tiempo cuadrático
- O(2^n): Tiempo exponencial

## Cómo Funciona

1. El sistema convierte el código fuente en una representación vectorial utilizando el módulo AST de Python
2. Tres modelos de redes neuronales diferentes predicen las complejidades O, Ω y Θ
3. Cada modelo está entrenado con ejemplos de algoritmos comunes y sus complejidades conocidas
4. La predicción final incluye las tres notaciones de complejidad

## Limitaciones Actuales

- El sistema está entrenado con un conjunto limitado de patrones de algoritmos
- La precisión depende de la calidad y cantidad de datos de entrenamiento
- La vectorización del código es simplificada y podría mejorarse

## Futuras Mejoras

- Ampliar el conjunto de datos de entrenamiento
- Mejorar la vectorización del código usando técnicas más avanzadas
- Agregar soporte para más tipos de algoritmos
- Implementar análisis de complejidad espacial
