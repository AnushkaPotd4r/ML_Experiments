"""
Algorithm 05: Decision Tree
Title: Decision Tree Classification System
Inputs: CGPA, IQ Score, Communication Rating
Prediction: Student placement status (Placed / Not Placed)
Visualizations: Decision tree structure diagram, Feature importance, Confusion matrix
"""

import os
import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from utils.preprocessing import load_dataset, split_data
from utils.evaluation import evaluate_classification
from utils.visualization import plot_decision_tree_structure, plot_feature_importance, plot_confusion_matrix

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "decision_tree_model.joblib")


class DecisionTreeLab:
    def __init__(self):
        self.model = None
        self.metrics = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.feature_names = ['CGPA', 'IQ', 'Communication_Rating']
        self.classes = ['Not Placed', 'Placed']
        self.criterion = 'gini'
        self.max_depth = 4
        self.train_or_load()

    def train_or_load(self, criterion='gini', max_depth=4, force_retrain=False):
        os.makedirs(MODEL_DIR, exist_ok=True)
        df = load_dataset("student_placement.csv")
        X = df[self.feature_names].values
        y = (df['Placement_Status'] == 'Placed').astype(int).values

        self.X_train, self.X_test, self.y_train, self.y_test = split_data(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.criterion = criterion
        self.max_depth = int(max_depth)

        if not force_retrain and os.path.exists(MODEL_PATH) and criterion == 'gini' and max_depth == 4:
            try:
                data = joblib.load(MODEL_PATH)
                self.model = data['model']
                self.metrics = data['metrics']
                return
            except Exception:
                pass

        self.model = DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            random_state=42
        )
        self.model.fit(self.X_train, self.y_train)

        y_pred = self.model.predict(self.X_test)
        self.metrics = evaluate_classification(self.y_test, y_pred, labels=[0, 1])

        joblib.dump({'model': self.model, 'metrics': self.metrics}, MODEL_PATH)

    def predict(self, cgpa, iq, comm):
        cgpa = float(cgpa)
        iq = float(iq)
        comm = float(comm)

        if cgpa < 0 or cgpa > 10:
            raise ValueError("CGPA must be between 0.0 and 10.0")
        if iq < 50 or iq > 180:
            raise ValueError("IQ Score must be between 50 and 180")
        if comm < 1.0 or comm > 5.0:
            raise ValueError("Communication Rating must be between 1.0 and 5.0")

        pred_idx = int(self.model.predict([[cgpa, iq, comm]])[0])
        pred_label = self.classes[pred_idx]
        probs = self.model.predict_proba([[cgpa, iq, comm]])[0]
        confidence = round(float(probs[pred_idx]) * 100, 2)

        tree_plot = plot_decision_tree_structure(self.model, self.feature_names, self.classes)
        feat_plot = plot_feature_importance(self.feature_names, self.model.feature_importances_)
        cm_plot = plot_confusion_matrix(self.metrics['confusion_matrix'], self.classes, 'Decision Tree Confusion Matrix')

        return {
            'prediction': pred_label,
            'is_placed': (pred_idx == 1),
            'confidence': confidence,
            'probabilities': {
                'Not Placed': round(float(probs[0]) * 100, 2),
                'Placed': round(float(probs[1]) * 100, 2)
            },
            'inputs': {'CGPA': cgpa, 'IQ': iq, 'Communication_Rating': comm},
            'tree_plot': tree_plot,
            'feat_plot': feat_plot,
            'cm_plot': cm_plot,
            'metrics': self.metrics,
            'hyperparams': {'criterion': self.criterion, 'max_depth': self.max_depth}
        }

    def get_details(self):
        tree_plot = plot_decision_tree_structure(self.model, self.feature_names, self.classes)
        feat_plot = plot_feature_importance(self.feature_names, self.model.feature_importances_)
        cm_plot = plot_confusion_matrix(self.metrics['confusion_matrix'], self.classes, 'Decision Tree Confusion Matrix')

        return {
            'id': 'decision_tree',
            'name': 'Decision Tree Classifier',
            'title': 'Decision Tree Classification System',
            'category': 'Classification',
            'description': 'A non-parametric supervised learning algorithm that splits data into branch-like decision rules based on information gain (Entropy) or impurity reduction (Gini).',
            'classes': self.classes,
            'metrics': self.metrics,
            'tree_plot': tree_plot,
            'feat_plot': feat_plot,
            'cm_plot': cm_plot,
            'hyperparams': {'criterion': self.criterion, 'max_depth': self.max_depth}
        }


decision_tree_lab = DecisionTreeLab()
