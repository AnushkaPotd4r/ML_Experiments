"""
Algorithm 03: Binary Logistic Regression
Title: Student Placement Prediction
Inputs: CGPA, IQ Score
Prediction: Placed / Not Placed
"""

import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from utils.preprocessing import load_dataset, split_data
from utils.evaluation import evaluate_classification
from utils.visualization import plot_binary_decision_boundary, plot_confusion_matrix

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "logistic_binary_model.joblib")


class BinaryLogisticRegressionLab:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.metrics = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.feature_names = ['CGPA', 'IQ Score']
        self.classes = ['Not Placed', 'Placed']
        self.train_or_load()

    def train_or_load(self, force_retrain=False):
        os.makedirs(MODEL_DIR, exist_ok=True)
        df = load_dataset("student_placement.csv")
        X = df[['CGPA', 'IQ']].values
        y = (df['Placement_Status'] == 'Placed').astype(int).values

        self.X_train, self.X_test, self.y_train, self.y_test = split_data(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)

        if not force_retrain and os.path.exists(MODEL_PATH):
            try:
                data = joblib.load(MODEL_PATH)
                self.model = data['model']
                self.scaler = data['scaler']
                self.metrics = data['metrics']
                return
            except Exception:
                pass

        self.model = LogisticRegression(random_state=42)
        self.model.fit(X_train_scaled, self.y_train)

        y_pred = self.model.predict(X_test_scaled)
        self.metrics = evaluate_classification(self.y_test, y_pred, labels=[0, 1])

        joblib.dump({'model': self.model, 'scaler': self.scaler, 'metrics': self.metrics}, MODEL_PATH)

    def predict(self, cgpa, iq):
        cgpa = float(cgpa)
        iq = float(iq)

        if cgpa < 0 or cgpa > 10:
            raise ValueError("CGPA must be between 0.0 and 10.0")
        if iq < 50 or iq > 180:
            raise ValueError("IQ Score must be between 50 and 180")

        query_unscaled = np.array([[cgpa, iq]])
        query_scaled = self.scaler.transform(query_unscaled)

        prob_placed = float(self.model.predict_proba(query_scaled)[0, 1])
        prob_not_placed = 1.0 - prob_placed
        pred_label_idx = 1 if prob_placed >= 0.5 else 0
        predicted_status = self.classes[pred_label_idx]

        confidence = round(max(prob_placed, prob_not_placed) * 100, 2)

        plot_boundary = plot_binary_decision_boundary(
            self.model, self.X_train, self.y_train, self.scaler, self.feature_names,
            title='Placement Decision Boundary (Logistic Regression)',
            query_point=[cgpa, iq]
        )
        cm_plot = plot_confusion_matrix(self.metrics['confusion_matrix'], self.classes, 'Binary Confusion Matrix')

        return {
            'prediction': predicted_status,
            'is_placed': (pred_label_idx == 1),
            'probability_placed': round(prob_placed * 100, 2),
            'probability_not_placed': round(prob_not_placed * 100, 2),
            'confidence': confidence,
            'cgpa': cgpa,
            'iq': iq,
            'plot_boundary': plot_boundary,
            'cm_plot': cm_plot,
            'metrics': self.metrics
        }

    def get_details(self):
        plot_boundary = plot_binary_decision_boundary(
            self.model, self.X_train, self.y_train, self.scaler, self.feature_names,
            title='Placement Decision Boundary (Logistic Regression)'
        )
        cm_plot = plot_confusion_matrix(self.metrics['confusion_matrix'], self.classes, 'Binary Confusion Matrix')

        return {
            'id': 'logistic_binary',
            'name': 'Binary Logistic Regression',
            'title': 'Student Placement Prediction',
            'category': 'Classification',
            'description': 'Predicts a binary categorical outcome (Placed vs. Not Placed) by applying the sigmoid activation function to a linear combination of features.',
            'classes': self.classes,
            'metrics': self.metrics,
            'plot_boundary': plot_boundary,
            'cm_plot': cm_plot
        }


logistic_binary_lab = BinaryLogisticRegressionLab()
