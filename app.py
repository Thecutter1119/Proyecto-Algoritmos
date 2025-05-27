from flask import Flask, request, render_template
from complexity_predictor import ComplexityPredictor

app = Flask(__name__)
predictor = ComplexityPredictor(vector_size=100)

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
