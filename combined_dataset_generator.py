import numpy as np

# Importa los generadores de ambos archivos
from dataset_generator import (
    generate_constant_time,
    generate_linear_search,
    generate_binary_search,
    generate_merge_sort,
    generate_bubble_sort,
    generate_optimized_bubble_sort,
    generate_quick_sort,
    generate_nested_loops
)

from bin_dataset_generator import (
    # Importa los generadores de bin_dataset_generator.py
    generate_colas,
    generate_pilas,
    generate_arbol_binario
)

# Dataset combinando todos los generadores
def generate_dataset(num_samples=1000):
    generators = [
        generate_constant_time,
        generate_linear_search,
        generate_binary_search,
        generate_merge_sort,
        generate_bubble_sort,
        generate_optimized_bubble_sort,
        generate_quick_sort,
        generate_nested_loops,
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
        code, labels = generator()
        code_samples.append(code)
        o_labels.append(labels['O'])
        omega_labels.append(labels['Ω'])
        theta_labels.append(labels['Θ'])

    return code_samples, o_labels, omega_labels, theta_labels
