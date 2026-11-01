"""
Algorithm 04: Multiclass Logistic Regression
Title: Student Performance Category Predictor
Inputs: CGPA, IQ Score
Prediction categories: Low Performance, Average Performance, High Performance
"""

import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from utils.preprocessing import load_dataset, split_data
from utils.evaluation import evaluate_classification
from utils.visualization import plot_confusion_matrix, plot_probabilities_bar

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "logistic_multiclass_model.joblib")


class MulticlassLogisticRegressionLab:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.metrics = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.classes = ['Low Performance', 'Average Performance', 'High Performance']
        self.train_or_load()

    def train_or_load(self, force_retrain=False):
        os.makedirs(MODEL_DIR, exist_ok=True)
        df = load_dataset("student_placement.csv")
        X = df[['CGPA', 'IQ']].values
        y = df['Performance_Category'].values

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

        self.model = LogisticRegression(solver='lbfgs', max_iter=500, random_state=42)
        self.model.fit(X_train_scaled, self.y_train)

        y_pred = self.model.predict(X_test_scaled)
        self.metrics = evaluate_classification(self.y_test, y_pred, labels=self.classes)

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

        probabilities = self.model.predict_proba(query_scaled)[0]
        pred_idx = np.argmax(probabilities)
        pred_label = self.model.classes_[pred_idx]

        # Map probabilities aligned with self.classes
        prob_dict = {}
        for c, p in zip(self.model.classes_, probabilities):
            prob_dict[c] = round(float(p) * 100, 2)

        prob_plot = plot_probabilities_bar(list(prob_dict.keys()), np.array(list(prob_dict.values())) / 100.0,
                                           title=f'Softmax Probability Distribution (Query: CGPA={cgpa}, IQ={iq})')
        cm_plot = plot_confusion_matrix(self.metrics['confusion_matrix'], self.classes, 'Multiclass Confusion Matrix')

        return {
            'prediction': pred_label,
            'probabilities': prob_dict,
            'confidence': round(float(probabilities[pred_idx]) * 100, 2),
            'cgpa': cgpa,
            'iq': iq,
            'prob_plot': prob_plot,
            'cm_plot': cm_plot,
            'metrics': self.metrics
        }

    def get_details(self):
        cm_plot = plot_confusion_matrix(self.metrics['confusion_matrix'], self.classes, 'Multiclass Confusion Matrix')
        return {
            'id': 'logistic_multiclass',
            'name': 'Multiclass Logistic Regression',
            'title': 'Student Performance Category Predictor',
            'category': 'Classification',
            'description': 'Generalizes binary logistic regression to multiple discrete categories using the multinomial softmax loss function.',
            'classes': self.classes,
            'metrics': self.metrics,
            'cm_plot': cm_plot
        }


logistic_multiclass_lab = MulticlassLogisticRegressionLab()
