
from flask import Flask, render_template, request
import joblib
import math

app = Flask(__name__)

# Load the trained Non-Linear SVM model
model = joblib.load("iris_svm.pkl")

flower_names = [
    "Setosa",
    "Versicolor",
    "Virginica"
]


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    values = {
        "sepal_length": "",
        "sepal_width": "",
        "petal_length": "",
        "petal_width": ""
    }

    if request.method == "POST":
        try:
            for feature in values:
                values[feature] = request.form.get(
                    feature, ""
                ).strip()

                if not values[feature]:
                    raise ValueError(
                        "Please enter all four measurements."
                    )

            measurements = [
                float(values["sepal_length"]),
                float(values["sepal_width"]),
                float(values["petal_length"]),
                float(values["petal_width"])
            ]

            if not all(math.isfinite(x) for x in measurements):
                raise ValueError(
                    "Please enter valid numerical values."
                )

            # Predict flower species
            result = int(model.predict([measurements])[0])
            prediction = flower_names[result]

        except ValueError as e:
            error = str(e) or "Please enter valid numbers."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        values=values
    )


if __name__ == "__main__":
    app.run(debug=True)
