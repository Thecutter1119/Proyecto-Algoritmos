from flask import Flask, request, render_template, jsonify
from complexity_predictor import ComplexityPredictor
from sorting_predictor import SortingPredictor
from dataset_generator import generate_dataset
from sorting_dataset_generator import generate_sorting_dataset
import ast

app = Flask(__name__)

# Configuración
vector_size = 100
num_samples = 2000
epochs = 15
batch_size = 32

# Predictores globales
complexity_predictor = None
sorting_predictor = None

def initialize_complexity_predictor():
    """Inicializa el predictor de complejidad general"""
    global complexity_predictor
    if complexity_predictor is None:
        print("Inicializando predictor de complejidad...")
        complexity_predictor = ComplexityPredictor(vector_size=vector_size)

        # Generar y entrenar con el dataset general
        code_samples, o_labels, omega_labels, theta_labels = generate_dataset(num_samples=num_samples)

        # Dividir dataset
        split_idx = int(0.8 * len(code_samples))
        train_samples = code_samples[:split_idx]
        train_o = o_labels[:split_idx]
        train_omega = omega_labels[:split_idx]
        train_theta = theta_labels[:split_idx]

        # Entrenar
        complexity_predictor.train(
            train_samples, train_o, train_omega, train_theta,
            epochs=epochs, batch_size=batch_size
        )
        print("Predictor de complejidad listo.")

def initialize_sorting_predictor():
    """Inicializa el predictor de algoritmos de ordenamiento"""
    global sorting_predictor
    if sorting_predictor is None:
        print("Inicializando predictor de ordenamiento...")
        sorting_predictor = SortingPredictor(vector_size=vector_size)

        # Generar dataset de ordenamiento
        code_samples, labels = generate_sorting_dataset(num_samples=1600)

        # Dividir dataset
        split_idx = int(0.8 * len(code_samples))
        train_samples = code_samples[:split_idx]
        train_labels = labels[:split_idx]

        # Entrenar
        sorting_predictor.train(
            train_samples, train_labels,
            epochs=20, batch_size=batch_size
        )
        print("Predictor de ordenamiento listo.")

def is_sorting_algorithm(code):
    """Detecta si el código parece ser un algoritmo de ordenamiento"""
    try:
        tree = ast.parse(code)
        
        # Buscar patrones típicos de ordenamiento
        has_nested_loops = False
        has_swapping = False
        has_comparison = False
        has_sort_keywords = False
        
        # Palabras clave relacionadas con ordenamiento
        sort_keywords = ['sort', 'bubble', 'merge', 'quick', 'heap', 'insertion', 'selection']
        
        code_lower = code.lower()
        for keyword in sort_keywords:
            if keyword in code_lower:
                has_sort_keywords = True
                break
        
        # Análisis del AST simplificado
        for node in ast.walk(tree):
            # Detectar bucles anidados
            if isinstance(node, ast.For):
                for child in ast.walk(node):
                    if isinstance(child, ast.For) and child != node:
                        has_nested_loops = True
            
            # Detectar intercambios (swapping)
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                if isinstance(node.targets[0], ast.Tuple) and len(node.targets[0].elts) == 2:
                    has_swapping = True
            
            # Detectar comparaciones
            if isinstance(node, ast.Compare):
                has_comparison = True
        
        # Heurística simple: si tiene palabras clave de ordenamiento O bucles anidados con comparaciones
        return has_sort_keywords or (has_nested_loops and has_comparison)
        
    except:
        return False

@app.route("/", methods=["GET", "POST"])
def index():
    """Ruta principal que maneja ambos tipos de predicción"""
    result = None
    code = ""
    prediction_type = "complexity"  # Por defecto

    if request.method == "POST":
        code = request.form.get("code", "")
        prediction_type = request.form.get("prediction_type", "complexity")
        
        if code:
            try:
                if prediction_type == "sorting":
                    # Usar predictor de ordenamiento
                    if sorting_predictor is None:
                        initialize_sorting_predictor()
                    
                    prediction = sorting_predictor.predict(code)
                    if prediction:
                        result = {
                            "type": "sorting",
                            "algorithm": prediction['algorithm'],
                            "confidence": f"{prediction['confidence']:.4f}",
                            "all_probabilities": prediction['all_probabilities']
                        }
                    else:
                        result = {"error": "No se pudo predecir el algoritmo de ordenamiento"}
                
                elif prediction_type == "complexity":
                    # Usar predictor de complejidad
                    if complexity_predictor is None:
                        initialize_complexity_predictor()
                    
                    prediction = complexity_predictor.predict(code)
                    if prediction:
                        result = {
                            "type": "complexity",
                            "O": prediction['O'],
                            "Ω": prediction['Ω'],
                            "Θ": prediction['Θ']
                        }
                    else:
                        result = {"error": "No se pudo predecir la complejidad"}
                
                elif prediction_type == "auto":
                    # Detección automática
                    if is_sorting_algorithm(code):
                        # Es algoritmo de ordenamiento
                        if sorting_predictor is None:
                            initialize_sorting_predictor()
                        
                        prediction = sorting_predictor.predict(code)
                        if prediction:
                            result = {
                                "type": "sorting",
                                "algorithm": prediction['algorithm'],
                                "confidence": f"{prediction['confidence']:.4f}",
                                "all_probabilities": prediction['all_probabilities'],
                                "auto_detected": True
                            }
                    else:
                        # Usar predictor de complejidad general
                        if complexity_predictor is None:
                            initialize_complexity_predictor()
                        
                        prediction = complexity_predictor.predict(code)
                        if prediction:
                            result = {
                                "type": "complexity",
                                "O": prediction['O'],
                                "Ω": prediction['Ω'],
                                "Θ": prediction['Θ'],
                                "auto_detected": True
                            }
                
            except Exception as e:
                result = {"error": f"Error durante la predicción: {str(e)}"}

    return render_template("index.html", result=result, code=code, prediction_type=prediction_type)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint para predicciones"""
    data = request.get_json()
    code = data.get("code", "")
    prediction_type = data.get("type", "auto")
    
    if not code:
        return jsonify({"error": "No code provided"}), 400
    
    try:
        if prediction_type == "sorting":
            if sorting_predictor is None:
                initialize_sorting_predictor()
            prediction = sorting_predictor.predict(code)
            return jsonify({"type": "sorting", "result": prediction})
        
        elif prediction_type == "complexity":
            if complexity_predictor is None:
                initialize_complexity_predictor()
            prediction = complexity_predictor.predict(code)
            return jsonify({"type": "complexity", "result": prediction})
        
        else:  # auto
            if is_sorting_algorithm(code):
                if sorting_predictor is None:
                    initialize_sorting_predictor()
                prediction = sorting_predictor.predict(code)
                return jsonify({"type": "sorting", "result": prediction, "auto_detected": True})
            else:
                if complexity_predictor is None:
                    initialize_complexity_predictor()
                prediction = complexity_predictor.predict(code)
                return jsonify({"type": "complexity", "result": prediction, "auto_detected": True})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ordenar", methods=["POST"])
def ordenar_array():
    data = request.get_json()
    array = data.get("array", [])
    algoritmo = data.get("algoritmo", "")

    try:
        if algoritmo == "bubble":
            from sorting_dataset_generator import generate_bubble_sort
            exec_code, _ = generate_bubble_sort()
        elif algoritmo == "selection":
            from sorting_dataset_generator import generate_selection_sort
            exec_code, _ = generate_selection_sort()
        elif algoritmo == "insertion":
            from sorting_dataset_generator import generate_insertion_sort
            exec_code, _ = generate_insertion_sort()
        elif algoritmo == "merge":
            from sorting_dataset_generator import generate_merge_sort
            exec_code, _ = generate_merge_sort()
        elif algoritmo == "quick":
            from sorting_dataset_generator import generate_quick_sort
            exec_code, _ = generate_quick_sort()
        else:
            return jsonify({"error": "Algoritmo no válido"}), 400

        # Inyectar el array en el código antes de ejecutarlo
        exec_globals = {}
        exec(exec_code, exec_globals)
        sorted_arr = exec_globals.get("sorted_arr", [])

        return jsonify({"ordenado": sorted_arr})
    except Exception as e:  
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)