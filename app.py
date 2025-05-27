from flask import Flask, request, render_template
from complexity_predictor import ComplexityPredictor

app = Flask(__name__)
predictor = ComplexityPredictor(vector_size=100)

# Opcional: cargar pesos si ya entrenaste y los guardaste
# predictor.o_model.load_weights("modelos_guardados/o_model.h5")
# predictor.omega_model.load_weights("modelos_guardados/omega_model.h5")
# predictor.theta_model.load_weights("modelos_guardados/theta_model.h5")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    code = ""

    if request.method == "POST":
        code = request.form.get("code")
        if code:
            prediction = predictor.predict(code)
            result = prediction

    return render_template("index.html", result=result, code=code)

if __name__ == "__main__":
    app.run(debug=True)
