"""
Algorithm 02: Multiple Linear Regression
Title: Student Package Prediction System
Inputs: Student CGPA, IQ Score
Prediction: Expected package in LPA
"""

import os
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from utils.preprocessing import load_dataset, split_data
from utils.evaluation import evaluate_regression
from utils.visualization import plot_mlr_performance

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "mlr_model.joblib")


class MultipleLinearRegressionLab:
    def __init__(self):
        self.model = None
        self.metrics = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.feature_names = ['CGPA', 'IQ']
        self.train_or_load()

    def train_or_load(self, force_retrain=False):
        os.makedirs(MODEL_DIR, exist_ok=True)
        df = load_dataset("student_placement.csv")
        X = df[self.feature_names].values
        y = df['Package_LPA'].values

        self.X_train, self.X_test, self.y_train, self.y_test = split_data(X, y, test_size=0.2, random_state=42)

        if not force_retrain and os.path.exists(MODEL_PATH):
            try:
                data = joblib.load(MODEL_PATH)
                self.model = data['model']
                self.metrics = data['metrics']
                self.y_pred = self.model.predict(self.X_test)
                return
            except Exception:
                pass

        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)

        self.y_pred = self.model.predict(self.X_test)
        self.metrics = evaluate_regression(self.y_test, self.y_pred)

        joblib.dump({'model': self.model, 'metrics': self.metrics}, MODEL_PATH)

    def predict(self, cgpa, iq):
        cgpa = float(cgpa)
        iq = float(iq)

        if cgpa < 0 or cgpa > 10:
            raise ValueError("CGPA must be between 0.0 and 10.0")
        if iq < 50 or iq > 180:
            raise ValueError("IQ Score must be between 50 and 180")

        pred_val = float(self.model.predict([[cgpa, iq]])[0])
        pred_val = round(max(0.0, pred_val), 2)

        coef_cgpa = self.model.coef_[0]
        coef_iq = self.model.coef_[1]
        intercept = self.model.intercept_

        calc_str = f"({coef_cgpa:.3f} × {cgpa}) + ({coef_iq:.3f} × {iq}) + ({intercept:.3f}) = ₹ {pred_val} LPA"
        plot_img = plot_mlr_performance(self.y_test, self.y_pred, self.feature_names, self.model.coef_)

        return {
            'prediction': pred_val,
            'prediction_formatted': f"₹ {pred_val} LPA",
            'cgpa': cgpa,
            'iq': iq,
            'calculation': calc_str,
            'plot': plot_img,
            'metrics': self.metrics,
            'coefficients': {
                'CGPA': round(float(coef_cgpa), 4),
                'IQ': round(float(coef_iq), 4),
                'Intercept': round(float(intercept), 4)
            }
        }

    def get_details(self):
        coef_cgpa = self.model.coef_[0]
        coef_iq = self.model.coef_[1]
        intercept = self.model.intercept_
        eq_str = f"Package = ({coef_cgpa:.3f} × CGPA) + ({coef_iq:.3f} × IQ) + ({intercept:.3f})"
        plot_img = plot_mlr_performance(self.y_test, self.y_pred, self.feature_names, self.model.coef_)

        return {
            'id': 'mlr',
            'name': 'Multiple Linear Regression',
            'title': 'Student Package Prediction System',
            'category': 'Regression',
            'description': 'Predicts a continuous outcome by fitting a multi-dimensional linear hyperplane over multiple explanatory features (CGPA & IQ Score).',
            'equation': eq_str,
            'coefficients': {
                'CGPA': round(float(coef_cgpa), 4),
                'IQ': round(float(coef_iq), 4),
                'Intercept': round(float(intercept), 4)
            },
            'metrics': self.metrics,
            'plot': plot_img
        }


mlr_lab = MultipleLinearRegressionLab()
