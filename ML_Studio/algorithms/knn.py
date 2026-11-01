"""
Algorithm 06: K-Nearest Neighbors (KNN)
Title: KNN Prediction System
Inputs: Sepal Length, Sepal Width, Petal Length, Petal Width (Iris Dataset)
Adjustable functionality: K value (1 to 15), Distance Metric (euclidean, manhattan, minkowski)
Prediction: Iris Species
"""

import os
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from utils.preprocessing import load_dataset, split_data
from utils.evaluation import evaluate_classification
from utils.visualization import plot_knn_k_analysis, plot_confusion_matrix

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "knn_model.joblib")


class KNNLab:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.metrics = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.feature_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        self.classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        self.k_value = 5
        self.metric = 'euclidean'
        self.k_analysis = {}
        self.train_or_load()

    def _compute_k_analysis(self, X_train_scaled, X_test_scaled):
        k_range = list(range(1, 16))
        accs = []
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k, metric=self.metric)
            knn.fit(X_train_scaled, self.y_train)
            accs.append(round(float(knn.score(X_test_scaled, self.y_test)), 4))
        return {'k_range': k_range, 'accuracies': accs}

    def train_or_load(self, k_value=5, metric='euclidean', force_retrain=False):
        os.makedirs(MODEL_DIR, exist_ok=True)
        df = load_dataset("iris.csv")
        X = df[self.feature_names].values
        y = df['Species'].values

        self.k_value = int(k_value)
        self.metric = metric

        self.X_train, self.X_test, self.y_train, self.y_test = split_data(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)

        self.k_analysis = self._compute_k_analysis(X_train_scaled, X_test_scaled)

        self.model = KNeighborsClassifier(n_neighbors=self.k_value, metric=self.metric)
        self.model.fit(X_train_scaled, self.y_train)

        y_pred = self.model.predict(X_test_scaled)
        self.metrics = evaluate_classification(self.y_test, y_pred, labels=self.classes)

        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'metrics': self.metrics,
            'k_analysis': self.k_analysis
        }, MODEL_PATH)

    def predict(self, sepal_len, sepal_wid, petal_len, petal_wid, k_value=None, metric=None):
        if k_value is not None and (int(k_value) != self.k_value or (metric is not None and metric != self.metric)):
            self.train_or_load(k_value=int(k_value), metric=metric or self.metric, force_retrain=True)

        sepal_len = float(sepal_len)
        sepal_wid = float(sepal_wid)
        petal_len = float(petal_len)
        petal_wid = float(petal_wid)

        query_unscaled = np.array([[sepal_len, sepal_wid, petal_len, petal_wid]])
        query_scaled = self.scaler.transform(query_unscaled)

        pred_label = self.model.predict(query_scaled)[0]
        probs = self.model.predict_proba(query_scaled)[0]
        distances, indices = self.model.kneighbors(query_scaled)

        # Neighbor summary
        neighbor_classes = [self.y_train[i] for i in indices[0]]
        neighbor_dists = [round(float(d), 3) for d in distances[0]]

        k_plot = plot_knn_k_analysis(self.k_analysis['k_range'], self.k_analysis['accuracies'], self.k_value)
        cm_plot = plot_confusion_matrix(self.metrics['confusion_matrix'], self.classes, f'KNN (K={self.k_value}) Confusion Matrix')

        prob_dict = {c: round(float(p) * 100, 2) for c, p in zip(self.model.classes_, probs)}

        return {
            'prediction': pred_label,
            'confidence': round(float(max(probs)) * 100, 2),
            'probabilities': prob_dict,
            'inputs': {
                'SepalLength': sepal_len,
                'SepalWidth': sepal_wid,
                'PetalLength': petal_len,
                'PetalWidth': petal_wid
            },
            'k_value': self.k_value,
            'metric': self.metric,
            'neighbors': list(zip(neighbor_classes, neighbor_dists)),
            'k_plot': k_plot,
            'cm_plot': cm_plot,
            'metrics': self.metrics
        }

    def get_details(self):
        k_plot = plot_knn_k_analysis(self.k_analysis['k_range'], self.k_analysis['accuracies'], self.k_value)
        cm_plot = plot_confusion_matrix(self.metrics['confusion_matrix'], self.classes, f'KNN (K={self.k_value}) Confusion Matrix')

        return {
            'id': 'knn',
            'name': 'K-Nearest Neighbors (KNN)',
            'title': 'KNN Prediction System',
            'category': 'Classification',
            'description': 'An instance-based, non-parametric algorithm that classifies a query instance based on the majority vote of its K closest training neighbors in feature space.',
            'classes': self.classes,
            'k_value': self.k_value,
            'metric': self.metric,
            'k_plot': k_plot,
            'cm_plot': cm_plot,
            'metrics': self.metrics
        }


knn_lab = KNNLab()
