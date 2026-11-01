"""
Algorithm 01: Simple Linear Regression
Title: Student Package Predictor
Input: Student CGPA
Prediction: Predicted package in LPA
"""

import os
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from utils.preprocessing import load_dataset, split_data
from utils.evaluation import evaluate_regression
from utils.visualization import plot_slr

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "slr_model.joblib")


class SimpleLinearRegressionLab:
    def __init__(self):
        self.model = None
        self.metrics = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.slope = None
        self.intercept = None
        self.train_or_load()

    def train_or_load(self, force_retrain=False):
        os.makedirs(MODEL_DIR, exist_ok=True)
        df = load_dataset("student_package.csv")
        X = df[['CGPA']].values
        y = df['Package_LPA'].values

        self.X_train, self.X_test, self.y_train, self.y_test = split_data(X, y, test_size=0.2, random_state=42)

        if not force_retrain and os.path.exists(MODEL_PATH):
            try:
                data = joblib.load(MODEL_PATH)
                self.model = data['model']
                self.metrics = data['metrics']
                self.slope = float(self.model.coef_[0])
                self.intercept = float(self.model.intercept_)
                return
            except Exception:
                pass

        # Train Scikit-learn model
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)

        y_pred = self.model.predict(self.X_test)
        self.metrics = evaluate_regression(self.y_test, y_pred)
        self.slope = float(self.model.coef_[0])
        self.intercept = float(self.model.intercept_)

        joblib.dump({'model': self.model, 'metrics': self.metrics}, MODEL_PATH)

    def predict(self, cgpa):
        cgpa = float(cgpa)
        if cgpa < 0 or cgpa > 10:
            raise ValueError("CGPA must be between 0.0 and 10.0")

        pred_val = float(self.model.predict([[cgpa]])[0])
        pred_val = round(max(0.0, pred_val), 2)

        plot_img = plot_slr(self.X_train, self.y_train, self.X_test, self.y_test, self.model, user_input=cgpa, user_pred=pred_val)

        equation_str = f"Package (LPA) = {self.slope:.3f} × CGPA + ({self.intercept:.3f})"
        calc_breakdown = f"{self.slope:.3f} × {cgpa} + ({self.intercept:.3f}) = ₹ {pred_val} LPA"

        return {
            'prediction': pred_val,
            'prediction_formatted': f"₹ {pred_val} LPA",
            'cgpa': cgpa,
            'equation': equation_str,
            'calculation': calc_breakdown,
            'plot': plot_img,
            'metrics': self.metrics,
            'slope': round(self.slope, 4),
            'intercept': round(self.intercept, 4)
        }

    def get_details(self):
        default_plot = plot_slr(self.X_train, self.y_train, self.X_test, self.y_test, self.model)
        equation_str = f"Package (LPA) = {self.slope:.3f} × CGPA + ({self.intercept:.3f})"
        return {
            'id': 'slr',
            'name': 'Simple Linear Regression',
            'title': 'Student Package Predictor',
            'category': 'Regression',
            'description': 'Models the linear relationship between a single independent variable (CGPA) and a continuous dependent variable (Package in LPA).',
            'equation': equation_str,
            'slope': round(self.slope, 4),
            'intercept': round(self.intercept, 4),
            'metrics': self.metrics,
            'plot': default_plot
        }


# Singleton instance
slr_lab = SimpleLinearRegressionLab()
