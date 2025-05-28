from flask import Flask, request, render_template
from complexity_predictor import ComplexityPredictor

app = Flask(__name__)

# Configuración del predictor
vector_size = 100
num_samples = 2000
epochs = 15
batch_size = 32

# Inicializar y entrenar el predictor
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


def complexity_to_notation(value):
    notations = {
        0: "O(1)",
        1: "O(log n)",
        2: "O(n)",
        3: "O(n log n)",
        4: "O(n²)"
    }
    return notations.get(value, "Desconocido")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    code = ""

    if request.method == "POST":
        code = request.form.get("code")
        if code:
            try:
                prediction = predictor.predict(code)
                if prediction:
                    result = {
                        'O': complexity_to_notation(prediction['O']),
                        'Ω': complexity_to_notation(prediction['Ω']),
                        'Θ': complexity_to_notation(prediction['Θ'])
                    }
            except Exception as e:
                result = {"error": str(e)}

    return render_template("index.html", result=result, code=code)

if __name__ == "__main__":
    app.run(debug=True)
