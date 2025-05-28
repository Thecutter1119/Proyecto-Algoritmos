from flask import Flask, request, render_template
from complexity_predictor import ComplexityPredictor

app = Flask(__name__)

# Configuración del predictor
vector_size = 100
num_samples = 2000
epochs = 15
batch_size = 32

# Inicializar el predictor como variable global
predictor = None

def initialize_predictor():
    global predictor
    if predictor is None:
        print("Inicializando y entrenando el predictor de complejidad...")
        predictor = ComplexityPredictor(vector_size=vector_size)

        # Generar y entrenar con el dataset
        from dataset_generator import generate_dataset
        code_samples, o_labels, omega_labels, theta_labels = generate_dataset(num_samples=num_samples)

        # Dividir en conjunto de entrenamiento y prueba
        split_idx = int(0.8 * len(code_samples))
        train_samples = code_samples[:split_idx]
        train_o = o_labels[:split_idx]
        train_omega = omega_labels[:split_idx]
        train_theta = theta_labels[:split_idx]

        # Entrenar el modelo
        predictor.train(
            train_samples,
            train_o,
            train_omega,
            train_theta,
            epochs=epochs,
            batch_size=batch_size
        )
        print("Entrenamiento completado.")

# Inicializar el predictor al inicio
initialize_predictor()


def complexity_to_notation(value):
    try:
        if not isinstance(value, (int, str)):
            return "Desconocido"
        if isinstance(value, str):
            return value  # Si ya es una notación, la devolvemos directamente
        notations = [
            "O(1)",
            "O(log n)",
            "O(n)",
            "O(n log n)",
            "O(n²)",
            "O(2^n)"
        ]
        return notations[value] if 0 <= value < len(notations) else "Desconocido"
    except Exception as e:
        print(f"Error en complexity_to_notation: {str(e)}")
        return "Desconocido"

@app.route("/", methods=["GET", "POST"])
def index():
    global predictor
    result = None
    code = ""

    if request.method == "POST":
        code = request.form.get("code")
        if code:
            try:
                # Asegurarse de que el predictor está inicializado
                if predictor is None:
                    print("Inicializando predictor...")
                    initialize_predictor()
                
                print("Realizando predicción...")
                prediction = predictor.predict(code)
                print(f"Predicción recibida: {prediction}")
                
                if prediction is None:
                    result = {"error": "Error al vectorizar el código. Asegúrese de que el código es válido."}
                elif all(k in prediction for k in ['O', 'Ω', 'Θ']):
                    result = {
                        'O': prediction['O'],  # Ya viene en formato string desde ComplexityPredictor
                        'Ω': prediction['Ω'],
                        'Θ': prediction['Θ']
                    }
                    print(f"Resultado formateado: {result}")
                else:
                    result = {"error": "La predicción no contiene todas las complejidades esperadas."}
            except Exception as e:
                result = {"error": f"Error al predecir la complejidad: {str(e)}"}
                print(f"Error en la predicción: {str(e)}")
                import traceback
                print(f"Traceback completo: {traceback.format_exc()}")
                # Reinicializar el predictor si hay un error
                predictor = None

    return render_template("index.html", result=result, code=code)

if __name__ == "__main__":
    app.run(debug=True)
