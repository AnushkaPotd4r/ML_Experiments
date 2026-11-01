"""
Algorithm 07: Naive Bayes
Title: Naive Bayes Classification System
Inputs: Sepal Length, Sepal Width, Petal Length, Petal Width (Gaussian Naive Bayes)
Prediction: Iris Species
Visualizations: Class Probability Chart, Confusion Matrix
Display: Accuracy, Precision, Recall, F1 Score
"""

import os
import joblib
import numpy as np
from sklearn.naive_bayes import GaussianNB
from utils.preprocessing import load_dataset, split_data
from utils.evaluation import evaluate_classification
from utils.visualization import plot_probabilities_bar, plot_confusion_matrix

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "naive_bayes_model.joblib")


class NaiveBayesLab:
    def __init__(self):
        self.model = None
        self.metrics = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.feature_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        self.classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        self.train_or_load()

    def train_or_load(self, force_retrain=False):
        os.makedirs(MODEL_DIR, exist_ok=True)
        df = load_dataset("iris.csv")
        X = df[self.feature_names].values
        y = df['Species'].values

        self.X_train, self.X_test, self.y_train, self.y_test = split_data(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        if not force_retrain and os.path.exists(MODEL_PATH):
            try:
                data = joblib.load(MODEL_PATH)
                self.model = data['model']
                self.metrics = data['metrics']
                return
            except Exception:
                pass

        self.model = GaussianNB()
        self.model.fit(self.X_train, self.y_train)

        y_pred = self.model.predict(self.X_test)
        self.metrics = evaluate_classification(self.y_test, y_pred, labels=self.classes)

        joblib.dump({'model': self.model, 'metrics': self.metrics}, MODEL_PATH)

    def predict(self, sepal_len, sepal_wid, petal_len, petal_wid):
        sepal_len = float(sepal_len)
        sepal_wid = float(sepal_wid)
        petal_len = float(petal_len)
        petal_wid = float(petal_wid)

        query = np.array([[sepal_len, sepal_wid, petal_len, petal_wid]])
        pred_label = self.model.predict(query)[0]
        probs = self.model.predict_proba(query)[0]

        prob_dict = {c: round(float(p) * 100, 2) for c, p in zip(self.model.classes_, probs)}
        confidence = round(float(max(probs)) * 100, 2)

        prob_plot = plot_probabilities_bar(list(prob_dict.keys()), np.array(list(prob_dict.values())) / 100.0,
                                           title=f'Posterior Probabilities P(Species | Features)')
        cm_plot = plot_confusion_matrix(self.metrics['confusion_matrix'], self.classes, 'Gaussian Naive Bayes Confusion Matrix')

        class_priors = {c: round(float(p), 4) for c, p in zip(self.model.classes_, self.model.class_prior_)}

        return {
            'prediction': pred_label,
            'confidence': confidence,
            'probabilities': prob_dict,
            'priors': class_priors,
            'inputs': {
                'SepalLength': sepal_len,
                'SepalWidth': sepal_wid,
                'PetalLength': petal_len,
                'PetalWidth': petal_wid
            },
            'prob_plot': prob_plot,
            'cm_plot': cm_plot,
            'metrics': self.metrics
        }

    def get_details(self):
        cm_plot = plot_confusion_matrix(self.metrics['confusion_matrix'], self.classes, 'Gaussian Naive Bayes Confusion Matrix')
        class_priors = {c: round(float(p), 4) for c, p in zip(self.model.classes_, self.model.class_prior_)}

        return {
            'id': 'naive_bayes',
            'name': 'Gaussian Naive Bayes',
            'title': 'Naive Bayes Classification System',
            'category': 'Classification',
            'description': 'A probabilistic classifier applying Bayes Theorem under the naive assumption of conditional feature independence and Gaussian normal likelihoods.',
            'classes': self.classes,
            'priors': class_priors,
            'cm_plot': cm_plot,
            'metrics': self.metrics
        }


naive_bayes_lab = NaiveBayesLab()
