"""
End-to-End Test Suite for ML Studio
Validates that:
1. All routes load with HTTP 200
2. All 10 algorithm prediction forms accept POST inputs and return valid predictions
3. Dataset Explorer loads datasets and performs custom model training
4. Performance page renders regression and classification metrics
5. SQLite database properly logs experiments and can clear history
"""

import unittest
from app import app
from utils.database import init_db, get_recent_experiments, clear_history


class MLStudioTestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()
        init_db()

    def test_01_dashboard(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'ML STUDIO', response.data)
        self.assertIn(b'Simple Linear Regression', response.data)
        self.assertIn(b'Random Forest Classifier', response.data)

    def test_02_slr(self):
        # GET
        res = self.client.get('/experiments/slr')
        self.assertEqual(res.status_code, 200)
        # POST
        res_post = self.client.post('/experiments/slr', data={'cgpa': '8.75'})
        self.assertEqual(res_post.status_code, 200)
        self.assertIn(b'Predicted Package', res_post.data)
        self.assertIn(b'LPA', res_post.data)

    def test_03_mlr(self):
        res = self.client.get('/experiments/mlr')
        self.assertEqual(res.status_code, 200)
        res_post = self.client.post('/experiments/mlr', data={'cgpa': '8.9', 'iq': '125'})
        self.assertEqual(res_post.status_code, 200)
        self.assertIn(b'Expected Package', res_post.data)

    def test_04_logistic_binary(self):
        res = self.client.get('/experiments/logistic_binary')
        self.assertEqual(res.status_code, 200)
        res_post = self.client.post('/experiments/logistic_binary', data={'cgpa': '8.5', 'iq': '115'})
        self.assertEqual(res_post.status_code, 200)
        self.assertIn(b'Prediction Status', res_post.data)

    def test_05_logistic_multiclass(self):
        res = self.client.get('/experiments/logistic_multiclass')
        self.assertEqual(res.status_code, 200)
        res_post = self.client.post('/experiments/logistic_multiclass', data={'cgpa': '9.1', 'iq': '130'})
        self.assertEqual(res_post.status_code, 200)
        self.assertIn(b'Predicted Performance Tier', res_post.data)

    def test_06_decision_tree(self):
        res = self.client.get('/experiments/decision_tree')
        self.assertEqual(res.status_code, 200)
        res_post = self.client.post('/experiments/decision_tree', data={
            'cgpa': '8.0', 'iq': '110', 'comm': '3.5', 'criterion': 'gini', 'max_depth': '3'
        })
        self.assertEqual(res_post.status_code, 200)
        self.assertIn(b'Placement Decision', res_post.data)

    def test_07_knn(self):
        res = self.client.get('/experiments/knn')
        self.assertEqual(res.status_code, 200)
        res_post = self.client.post('/experiments/knn', data={
            'sepal_len': '5.1', 'sepal_wid': '3.5', 'petal_len': '1.4', 'petal_wid': '0.2',
            'k_value': '3', 'metric': 'euclidean'
        })
        self.assertEqual(res_post.status_code, 200)
        self.assertIn(b'Iris-setosa', res_post.data)

    def test_08_naive_bayes(self):
        res = self.client.get('/experiments/naive_bayes')
        self.assertEqual(res.status_code, 200)
        res_post = self.client.post('/experiments/naive_bayes', data={
            'sepal_len': '6.2', 'sepal_wid': '3.4', 'petal_len': '5.4', 'petal_wid': '2.3'
        })
        self.assertEqual(res_post.status_code, 200)
        self.assertIn(b'Predicted Iris Class', res_post.data)

    def test_09_linear_svm(self):
        res = self.client.get('/experiments/linear_svm')
        self.assertEqual(res.status_code, 200)
        res_post = self.client.post('/experiments/linear_svm', data={
            'aptitude': '82.0', 'technical': '76.0', 'c_param': '1.0'
        })
        self.assertEqual(res_post.status_code, 200)
        self.assertIn(b'Admission Decision', res_post.data)

    def test_10_nonlinear_svm(self):
        res = self.client.get('/experiments/nonlinear_svm')
        self.assertEqual(res.status_code, 200)
        res_post = self.client.post('/experiments/nonlinear_svm', data={
            'skill_x': '9.0', 'skill_y': '11.5', 'kernel': 'rbf', 'c_param': '1.0', 'gamma': 'scale'
        })
        self.assertEqual(res_post.status_code, 200)
        self.assertIn(b'Predicted Category', res_post.data)

    def test_11_random_forest(self):
        res = self.client.get('/experiments/random_forest')
        self.assertEqual(res.status_code, 200)
        res_post = self.client.post('/experiments/random_forest', data={
            'cgpa': '8.5', 'iq': '118', 'comm': '4.2', 'projects': '4',
            'n_estimators': '50', 'max_depth': '4'
        })
        self.assertEqual(res_post.status_code, 200)
        self.assertIn(b'Ensemble Placement Verdict', res_post.data)

    def test_12_dataset_explorer(self):
        res = self.client.get('/dataset_explorer?file=student_package.csv')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Package_LPA', res.data)

    def test_13_dynamic_model_training(self):
        res = self.client.post('/train_custom_dataset', data={
            'dataset_file': 'student_package.csv',
            'target_col': 'Package_LPA',
            'feature_cols': ['CGPA'],
            'algo_choice': 'linear_regression'
        })
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Training Completed: Linear Regression', res.data)

    def test_14_performance_page(self):
        res = self.client.get('/performance')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Regression Models Performance', res.data)
        self.assertIn(b'Classification Models Performance', res.data)

    def test_15_about_page(self):
        res = self.client.get('/about')
        self.assertEqual(res.status_code, 200)
        self.assertIn(b'Viva Voce', res.data)

    def test_16_history_and_clear(self):
        history = get_recent_experiments()
        self.assertTrue(len(history) > 0)
        res = self.client.post('/clear-history', follow_redirects=True)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(get_recent_experiments()), 0)


if __name__ == '__main__':
    unittest.main()
