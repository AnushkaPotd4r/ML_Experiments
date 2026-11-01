"""
Algorithm 08: Linear Support Vector Machine (Linear SVM)
Title: Linear Support Vector Machine
Inputs: Aptitude Score, Technical Score
Additional functionality: Adjustable regularization parameter C
Visualizations: Decision boundary, Support vectors, Classification regions
Display: Accuracy, Precision, Recall, F1 Score, Support Vectors Count
"""

import os
import joblib
import numpy as np
from sklearn.svm import SVC
from utils.preprocessing import load_dataset, split_data
from utils.evaluation import evaluate_classification
from utils.visualization import plot_svm_decision_boundary, plot_confusion_matrix

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "linear_svm_model.joblib")


class LinearSVMLab:
    def __init__(self):
        self.model = None
        self.metrics = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.feature_names = ['Aptitude_Score', 'Technical_Score']
        self.classes = ['Rejected', 'Admitted']
        self.C = 1.0
        self.train_or_load()

    def train_or_load(self, C=1.0, force_retrain=False):
        os.makedirs(MODEL_DIR, exist_ok=True)
        df = load_dataset("svm_twoclass.csv")
        X = df[self.feature_names].values
        y = (df['Admitted'] == 'Admitted').astype(int).values

        self.C = float(C)

        self.X_train, self.X_test, self.y_train, self.y_test = split_data(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        if not force_retrain and os.path.exists(MODEL_PATH) and self.C == 1.0:
            try:
                data = joblib.load(MODEL_PATH)
                self.model = data['model']
                self.metrics = data['metrics']
                return
            except Exception:
                pass

        self.model = SVC(kernel='linear', C=self.C, random_state=42)
        self.model.fit(self.X_train, self.y_train)

        y_pred = self.model.predict(self.X_test)
        self.metrics = evaluate_classification(self.y_test, y_pred, labels=[0, 1])

        joblib.dump({'model': self.model, 'metrics': self.metrics, 'C': self.C}, MODEL_PATH)

    def predict(self, aptitude, technical, C=None):
        if C is not None and float(C) != self.C:
            self.train_or_load(C=float(C), force_retrain=True)

        aptitude = float(aptitude)
        technical = float(technical)

        if aptitude < 0 or aptitude > 100:
            raise ValueError("Aptitude Score must be between 0 and 100")
        if technical < 0 or technical > 100:
            raise ValueError("Technical Score must be between 0 and 100")

        query = np.array([[aptitude, technical]])
        pred_idx = int(self.model.predict(query)[0])
        pred_label = self.classes[pred_idx]
        dist_to_hyperplane = float(self.model.decision_function(query)[0])

        boundary_plot = plot_svm_decision_boundary(
            self.model, self.X_train, self.y_train, self.feature_names,
            title=f'Linear SVM Hyperplane & Margins (C={self.C})',
            is_linear=True,
            query_point=[aptitude, technical]
        )
        cm_plot = plot_confusion_matrix(self.metrics['confusion_matrix'], self.classes, 'Linear SVM Confusion Matrix')

        weights = self.model.coef_[0]
        intercept = self.model.intercept_[0]
        hyperplane_eq = f"{weights[0]:.3f} × Aptitude + {weights[1]:.3f} × Technical + ({intercept:.3f}) = 0"

        return {
            'prediction': pred_label,
            'is_admitted': (pred_idx == 1),
            'distance_to_hyperplane': round(dist_to_hyperplane, 3),
            'inputs': {'Aptitude': aptitude, 'Technical': technical},
            'C': self.C,
            'n_support_vectors': int(len(self.model.support_vectors_)),
            'weights': [round(float(w), 4) for w in weights],
            'intercept': round(float(intercept), 4),
            'equation': hyperplane_eq,
            'boundary_plot': boundary_plot,
            'cm_plot': cm_plot,
            'metrics': self.metrics
        }

    def get_details(self):
        boundary_plot = plot_svm_decision_boundary(
            self.model, self.X_train, self.y_train, self.feature_names,
            title=f'Linear SVM Hyperplane & Margins (C={self.C})',
            is_linear=True
        )
        cm_plot = plot_confusion_matrix(self.metrics['confusion_matrix'], self.classes, 'Linear SVM Confusion Matrix')
        weights = self.model.coef_[0]
        intercept = self.model.intercept_[0]
        hyperplane_eq = f"{weights[0]:.3f} × Aptitude + {weights[1]:.3f} × Technical + ({intercept:.3f}) = 0"

        return {
            'id': 'linear_svm',
            'name': 'Linear Support Vector Machine',
            'title': 'Linear Support Vector Machine',
            'category': 'Classification',
            'description': 'Finds the optimal separating linear hyperplane with the maximum geometric margin between two classes using convex quadratic optimization.',
            'classes': self.classes,
            'C': self.C,
            'n_support_vectors': int(len(self.model.support_vectors_)),
            'weights': [round(float(w), 4) for w in weights],
            'intercept': round(float(intercept), 4),
            'equation': hyperplane_eq,
            'boundary_plot': boundary_plot,
            'cm_plot': cm_plot,
            'metrics': self.metrics
        }


linear_svm_lab = LinearSVMLab()
