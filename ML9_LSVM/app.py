from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained Linear SVM model
model = joblib.load("linear_svm_model.pkl")

# Input feature names in the correct order
FEATURES = [
    "variance",
    "skewness",
    "curtosis",
    "entropy"
]


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    error = None
    input_values = {}

    if request.method == "POST":

        try:
            # Read input values from HTML form
            values = [
                float(request.form["variance"]),
                float(request.form["skewness"]),
                float(request.form["curtosis"]),
                float(request.form["entropy"])
            ]

            input_values = dict(zip(FEATURES, values))

            # Convert input into a DataFrame
            input_data = pd.DataFrame(
                [values],
                columns=FEATURES
            )

            # Predict banknote class
            result = int(model.predict(input_data)[0])

            # Display the result
            if result == 0:
                prediction = "Genuine Banknote"
            else:
                prediction = "Counterfeit Banknote"

        except (ValueError, TypeError, KeyError):
            error = "Please enter valid numerical values in all fields."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        input_values=input_values
    )


if __name__ == "__main__":
    app.run(debug=True)