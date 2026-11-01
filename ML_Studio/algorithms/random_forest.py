"""
Algorithm 10: Random Forest Classifier
Title: Random Forest Classification System
Inputs: CGPA, IQ, Communication Rating, Projects Count
Additional functionality: Number of trees (n_estimators), Maximum tree depth (max_depth)
Prediction: Student Placement Status (Placed / Not Placed)
Visualizations: Feature importance chart, Confusion matrix, Estimator performance comparison
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from utils.preprocessing import load_dataset, split_data
from utils.evaluation import evaluate_classification
from utils.visualization import plot_feature_importance, plot_confusion_matrix

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.joblib")


class RandomForestLab:
    def __init__(self):
        self.model = None
        self.metrics = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.feature_names = ['CGPA', 'IQ', 'Communication_Rating', 'Projects_Count']
        self.classes = ['Not Placed', 'Placed']
        self.n_estimators = 100
        self.max_depth = 5
        self.train_or_load()

    def train_or_load(self, n_estimators=100, max_depth=5, force_retrain=False):
        os.makedirs(MODEL_DIR, exist_ok=True)
        df = load_dataset("student_placement.csv")
        X = df[self.feature_names].values
        y = (df['Placement_Status'] == 'Placed').astype(int).values

        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth) if max_depth is not None and str(max_depth).isdigit() else None

        self.X_train, self.X_test, self.y_train, self.y_test = split_data(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        if not force_retrain and os.path.exists(MODEL_PATH) and self.n_estimators == 100 and self.max_depth == 5:
            try:
                data = joblib.load(MODEL_PATH)
                self.model = data['model']
                self.metrics = data['metrics']
                return
            except Exception:
                pass

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42
        )
        self.model.fit(self.X_train, self.y_train)

        y_pred = self.model.predict(self.X_test)
        self.metrics = evaluate_classification(self.y_test, y_pred, labels=[0, 1])

        joblib.dump({
            'model': self.model,
            'metrics': self.metrics,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth
        }, MODEL_PATH)

    def predict(self, cgpa, iq, comm, projects, n_estimators=None, max_depth=None):
        retrain = False
        n_est = int(n_estimators) if n_estimators else self.n_estimators
        m_dep = int(max_depth) if (max_depth is not None and str(max_depth).isdigit()) else None

        if n_est != self.n_estimators or m_dep != self.max_depth:
            retrain = True

        if retrain:
            self.train_or_load(n_estimators=n_est, max_depth=m_dep, force_retrain=True)

        cgpa = float(cgpa)
        iq = float(iq)
        comm = float(comm)
        projects = int(projects)

        query = np.array([[cgpa, iq, comm, projects]])
        pred_idx = int(self.model.predict(query)[0])
        pred_label = self.classes[pred_idx]
        probs = self.model.predict_proba(query)[0]

        feat_plot = plot_feature_importance(self.feature_names, self.model.feature_importances_,
                                            title='Random Forest MDI Feature Importance')
        cm_plot = plot_confusion_matrix(self.metrics['confusion_matrix'], self.classes,
                                        f'Random Forest (Trees={self.n_estimators}) Confusion Matrix')

        return {
            'prediction': pred_label,
            'is_placed': (pred_idx == 1),
            'confidence': round(float(probs[pred_idx]) * 100, 2),
            'probabilities': {
                'Not Placed': round(float(probs[0]) * 100, 2),
                'Placed': round(float(probs[1]) * 100, 2)
            },
            'inputs': {'CGPA': cgpa, 'IQ': iq, 'Communication_Rating': comm, 'Projects_Count': projects},
            'hyperparams': {'n_estimators': self.n_estimators, 'max_depth': self.max_depth},
            'feat_plot': feat_plot,
            'cm_plot': cm_plot,
            'metrics': self.metrics
        }

    def get_details(self):
        feat_plot = plot_feature_importance(self.feature_names, self.model.feature_importances_,
                                            title='Random Forest MDI Feature Importance')
        cm_plot = plot_confusion_matrix(self.metrics['confusion_matrix'], self.classes,
                                        f'Random Forest (Trees={self.n_estimators}) Confusion Matrix')

        return {
            'id': 'random_forest',
            'name': 'Random Forest Classifier',
            'title': 'Random Forest Classification System',
            'category': 'Classification',
            'description': 'An ensemble bagging algorithm constructing a multitude of uncorrelated decision trees during training and aggregating their individual predictions by majority voting.',
            'classes': self.classes,
            'hyperparams': {'n_estimators': self.n_estimators, 'max_depth': self.max_depth},
            'feat_plot': feat_plot,
            'cm_plot': cm_plot,
            'metrics': self.metrics
        }


random_forest_lab = RandomForestLab()
