"""
Algorithm 09: Non-Linear Support Vector Machine (Kernel SVM)
Title: Non-Linear Support Vector Machine
Inputs: Skill_X, Skill_Y (Moons Non-Linear Dataset)
Additional functionality: Kernel selection (RBF, Polynomial), C, Gamma
Visualizations: Non-linear decision boundary contour, Support vectors, Classification regions
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
MODEL_PATH = os.path.join(MODEL_DIR, "nonlinear_svm_model.joblib")


class NonLinearSVMLab:
    def __init__(self):
        self.model = None
        self.metrics = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.feature_names = ['Skill_X', 'Skill_Y']
        self.classes = ['Class-B', 'Class-A']
        self.kernel = 'rbf'
        self.C = 1.0
        self.gamma = 'scale'
        self.degree = 3
        self.train_or_load()

    def train_or_load(self, kernel='rbf', C=1.0, gamma='scale', degree=3, force_retrain=False):
        os.makedirs(MODEL_DIR, exist_ok=True)
        df = load_dataset("svm_nonlinear.csv")
        X = df[self.feature_names].values
        y = (df['Category'] == 'Class-A').astype(int).values

        self.kernel = kernel
        self.C = float(C)
        self.gamma = float(gamma) if (isinstance(gamma, (int, float)) or (isinstance(gamma, str) and gamma.replace('.', '', 1).isdigit())) else gamma
        self.degree = int(degree)

        self.X_train, self.X_test, self.y_train, self.y_test = split_data(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        if not force_retrain and os.path.exists(MODEL_PATH) and self.kernel == 'rbf' and self.C == 1.0 and self.gamma == 'scale':
            try:
                data = joblib.load(MODEL_PATH)
                self.model = data['model']
                self.metrics = data['metrics']
                return
            except Exception:
                pass

        self.model = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma, degree=self.degree, random_state=42)
        self.model.fit(self.X_train, self.y_train)

        y_pred = self.model.predict(self.X_test)
        self.metrics = evaluate_classification(self.y_test, y_pred, labels=[0, 1])

        joblib.dump({
            'model': self.model,
            'metrics': self.metrics,
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma,
            'degree': self.degree
        }, MODEL_PATH)

    def predict(self, skill_x, skill_y, kernel=None, C=None, gamma=None, degree=None):
        retrain_needed = False
        k = kernel if kernel is not None else self.kernel
        c_val = float(C) if C is not None else self.C
        g_val = gamma if gamma is not None else self.gamma
        d_val = int(degree) if degree is not None else self.degree

        if k != self.kernel or c_val != self.C or g_val != self.gamma or d_val != self.degree:
            retrain_needed = True

        if retrain_needed:
            self.train_or_load(kernel=k, C=c_val, gamma=g_val, degree=d_val, force_retrain=True)

        skill_x = float(skill_x)
        skill_y = float(skill_y)

        query = np.array([[skill_x, skill_y]])
        pred_idx = int(self.model.predict(query)[0])
        pred_label = self.classes[pred_idx]
        decision_val = float(self.model.decision_function(query)[0])

        boundary_plot = plot_svm_decision_boundary(
            self.model, self.X_train, self.y_train, self.feature_names,
            title=f'Non-Linear SVM Decision Boundary ({self.kernel.upper()} Kernel, C={self.C})',
            is_linear=False,
            query_point=[skill_x, skill_y]
        )
        cm_plot = plot_confusion_matrix(self.metrics['confusion_matrix'], self.classes, 'Non-Linear SVM Confusion Matrix')

        return {
            'prediction': pred_label,
            'is_class_a': (pred_idx == 1),
            'decision_value': round(decision_val, 3),
            'inputs': {'Skill_X': skill_x, 'Skill_Y': skill_y},
            'hyperparams': {'kernel': self.kernel, 'C': self.C, 'gamma': self.gamma, 'degree': self.degree},
            'n_support_vectors': int(len(self.model.support_vectors_)),
            'boundary_plot': boundary_plot,
            'cm_plot': cm_plot,
            'metrics': self.metrics
        }

    def get_details(self):
        boundary_plot = plot_svm_decision_boundary(
            self.model, self.X_train, self.y_train, self.feature_names,
            title=f'Non-Linear SVM Decision Boundary ({self.kernel.upper()} Kernel, C={self.C})',
            is_linear=False
        )
        cm_plot = plot_confusion_matrix(self.metrics['confusion_matrix'], self.classes, 'Non-Linear SVM Confusion Matrix')

        return {
            'id': 'nonlinear_svm',
            'name': 'Non-Linear Support Vector Machine',
            'title': 'Non-Linear Support Vector Machine',
            'category': 'Classification',
            'description': 'Projects non-linearly separable data into higher-dimensional Hilbert spaces using the Kernel Trick (RBF or Polynomial) to construct non-linear decision boundaries.',
            'classes': self.classes,
            'hyperparams': {'kernel': self.kernel, 'C': self.C, 'gamma': self.gamma, 'degree': self.degree},
            'n_support_vectors': int(len(self.model.support_vectors_)),
            'boundary_plot': boundary_plot,
            'cm_plot': cm_plot,
            'metrics': self.metrics
        }


nonlinear_svm_lab = NonLinearSVMLab()
