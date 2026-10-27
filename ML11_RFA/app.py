
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained models
classifier_bundle = joblib.load("diabetes_classifier.joblib")
regressor_bundle = joblib.load("glucose_regressor.joblib")

classifier = classifier_bundle["pipeline"]
regressor = regressor_bundle["pipeline"]

classification_features = classifier_bundle["features"]
regression_features = regressor_bundle["features"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get values submitted through the HTML form
        pregnancies = int(request.form["Pregnancies"])
        glucose = float(request.form["Glucose"])
        blood_pressure = float(request.form["BloodPressure"])
        skin_thickness = float(request.form["SkinThickness"])
        insulin = float(request.form["Insulin"])
        bmi = float(request.form["BMI"])
        dpf = float(request.form["DiabetesPedigreeFunction"])
        age = int(request.form["Age"])

        values = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age
        }

        # Reject invalid values before making predictions
        if any(not np.isfinite(float(v)) for v in values.values()):
            raise ValueError("Please enter valid numeric values.")

        if any(float(v) < 0 for v in values.values()):
            raise ValueError("Measurements cannot be negative.")

        # Build input rows in the exact feature order used in training
        classification_input = pd.DataFrame(
            [[values[col] for col in classification_features]],
            columns=classification_features
        )

        regression_input = pd.DataFrame(
            [[values[col] for col in regression_features]],
            columns=regression_features
        )

        # Apply the same missing-value assumption used during training
        zero_as_missing = [
            "Glucose", "BloodPressure",
            "SkinThickness", "Insulin", "BMI"
        ]

        for col in zero_as_missing:
            if col in classification_input.columns:
                classification_input.loc[
                    classification_input[col] == 0, col
                ] = np.nan

            if col in regression_input.columns:
                regression_input.loc[
                    regression_input[col] == 0, col
                ] = np.nan

        # Classification prediction
        outcome = int(classifier.predict(classification_input)[0])
        probability = float(
            classifier.predict_proba(classification_input)[0, 1]
        )

        # Regression prediction
        glucose_estimate = float(
            regressor.predict(regression_input)[0]
        )

        if outcome == 1:
            result = "Diabetes-positive class (1)"
        else:
            result = "Diabetes-negative class (0)"

        return render_template(
            "index.html",
            result=result,
            probability=f"{probability:.1%}",
            glucose_estimate=f"{glucose_estimate:.1f}",
            submitted=True
        )

    except (ValueError, KeyError, TypeError):
        return render_template(
            "index.html",
            error="Please enter valid numeric values in all fields.",
            submitted=False
        )


if __name__ == "__main__":
    app.run(debug=True)
