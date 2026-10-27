from flask import Flask, render_template, request
from joblib import load
import numpy as np

app = Flask(__name__)

# Load complete KNN pipeline
# Pipeline contains StandardScaler + KNN
model = load("KNN_Diabetes_Model.joblib")


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    # -----------------------------
    # Get input values
    # -----------------------------
    pregnancies = float(request.form['pregnancies'])
    glucose = float(request.form['glucose'])
    blood_pressure = float(request.form['blood_pressure'])
    skin_thickness = float(request.form['skin_thickness'])
    insulin = float(request.form['insulin'])
    bmi = float(request.form['bmi'])
    diabetes_pedigree = float(request.form['diabetes_pedigree'])
    age = float(request.form['age'])


    # -----------------------------
    # Create input array
    # IMPORTANT:
    # Feature order must match
    # training data
    # -----------------------------
    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]])


    # -----------------------------
    # Prediction
    # Pipeline automatically:
    # 1. Scales the input
    # 2. Applies KNN
    # -----------------------------
    prediction = model.predict(input_data)


    # -----------------------------
    # Convert prediction to result
    # -----------------------------
    if prediction[0] == 1:
        result = "Diabetes"
    else:
        result = "No Diabetes"


    # -----------------------------
    # Send result to HTML
    # -----------------------------
    return render_template(
        "index.html",
        prediction_text=f"Prediction : {result}"
    )


if __name__ == "__main__":
    app.run(debug=True)