"""
ML STUDIO – INTERACTIVE MACHINE LEARNING ALGORITHMS LAB
Main Flask Application Server

A centralized, responsive educational dashboard uniting 10 foundational Machine Learning algorithms.
"""

import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename

# Import Algorithm Labs
from algorithms.simple_linear_regression import slr_lab
from algorithms.multiple_linear_regression import mlr_lab
from algorithms.logistic_binary import logistic_binary_lab
from algorithms.logistic_multiclass import logistic_multiclass_lab
from algorithms.decision_tree import decision_tree_lab
from algorithms.knn import knn_lab
from algorithms.naive_bayes import naive_bayes_lab
from algorithms.linear_svm import linear_svm_lab
from algorithms.nonlinear_svm import nonlinear_svm_lab
from algorithms.random_forest import random_forest_lab

# Import Utilities
from utils.database import init_db, log_experiment, get_recent_experiments, clear_history
from utils.preprocessing import split_data, scale_features, encode_labels
from utils.evaluation import evaluate_regression, evaluate_classification

app = Flask(__name__)
app.secret_key = 'ml-studio-academic-laboratory-secret-key-2026'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Initialize SQLite database
init_db()


# -------------------------------------------------------------
# 1. MAIN DASHBOARD ROUTE
# -------------------------------------------------------------
@app.route('/')
@app.route('/dashboard')
def dashboard():
    recent_experiments = get_recent_experiments(limit=10)
    return render_template('dashboard.html',
                           active_page='dashboard',
                           recent_experiments=recent_experiments)


@app.route('/clear-history', methods=['POST'])
def clear_experiment_history():
    clear_history()
    flash('Experiment history cleared successfully.', 'info')
    return redirect(url_for('dashboard'))


# -------------------------------------------------------------
# 2. INDIVIDUAL ALGORITHM EXPERIMENTS (01 TO 10)
# -------------------------------------------------------------

# ALGORITHM 01: Simple Linear Regression
@app.route('/experiments/slr', methods=['GET', 'POST'])
def exp_slr():
    details = slr_lab.get_details()
    result = None
    query_cgpa = None

    if request.method == 'POST':
        try:
            query_cgpa = float(request.form.get('cgpa', 8.2))
            result = slr_lab.predict(query_cgpa)
            log_experiment('slr', 'Simple Linear Regression', 'Regression',
                           {'CGPA': query_cgpa}, result['prediction_formatted'],
                           result['metrics'])
        except Exception as e:
            flash(f"Input Error: {str(e)}", 'error')

    return render_template('experiments/slr.html',
                           active_page='slr',
                           details=details,
                           result=result,
                           query_cgpa=query_cgpa)


# ALGORITHM 02: Multiple Linear Regression
@app.route('/experiments/mlr', methods=['GET', 'POST'])
def exp_mlr():
    details = mlr_lab.get_details()
    result = None
    query_cgpa = None
    query_iq = None

    if request.method == 'POST':
        try:
            query_cgpa = float(request.form.get('cgpa', 8.5))
            query_iq = float(request.form.get('iq', 115))
            result = mlr_lab.predict(query_cgpa, query_iq)
            log_experiment('mlr', 'Multiple Linear Regression', 'Regression',
                           {'CGPA': query_cgpa, 'IQ': query_iq},
                           result['prediction_formatted'], result['metrics'])
        except Exception as e:
            flash(f"Input Error: {str(e)}", 'error')

    return render_template('experiments/mlr.html',
                           active_page='mlr',
                           details=details,
                           result=result,
                           query_cgpa=query_cgpa,
                           query_iq=query_iq)


# ALGORITHM 03: Binary Logistic Regression
@app.route('/experiments/logistic_binary', methods=['GET', 'POST'])
def exp_logistic_binary():
    details = logistic_binary_lab.get_details()
    result = None
    query_cgpa = None
    query_iq = None

    if request.method == 'POST':
        try:
            query_cgpa = float(request.form.get('cgpa', 7.8))
            query_iq = float(request.form.get('iq', 105))
            result = logistic_binary_lab.predict(query_cgpa, query_iq)
            log_experiment('logistic_binary', 'Binary Logistic Regression', 'Classification',
                           {'CGPA': query_cgpa, 'IQ': query_iq},
                           result['prediction'], result['metrics'])
        except Exception as e:
            flash(f"Input Error: {str(e)}", 'error')

    return render_template('experiments/logistic_binary.html',
                           active_page='logistic_binary',
                           details=details,
                           result=result,
                           query_cgpa=query_cgpa,
                           query_iq=query_iq)


# ALGORITHM 04: Multiclass Logistic Regression
@app.route('/experiments/logistic_multiclass', methods=['GET', 'POST'])
def exp_logistic_multiclass():
    details = logistic_multiclass_lab.get_details()
    result = None
    query_cgpa = None
    query_iq = None

    if request.method == 'POST':
        try:
            query_cgpa = float(request.form.get('cgpa', 7.8))
            query_iq = float(request.form.get('iq', 110))
            result = logistic_multiclass_lab.predict(query_cgpa, query_iq)
            log_experiment('logistic_multiclass', 'Multiclass Logistic Regression', 'Classification',
                           {'CGPA': query_cgpa, 'IQ': query_iq},
                           result['prediction'], result['metrics'])
        except Exception as e:
            flash(f"Input Error: {str(e)}", 'error')

    return render_template('experiments/logistic_multiclass.html',
                           active_page='logistic_multiclass',
                           details=details,
                           result=result,
                           query_cgpa=query_cgpa,
                           query_iq=query_iq)


# ALGORITHM 05: Decision Tree Classifier
@app.route('/experiments/decision_tree', methods=['GET', 'POST'])
def exp_decision_tree():
    details = decision_tree_lab.get_details()
    result = None
    query_cgpa = None
    query_iq = None
    query_comm = None
    selected_criterion = 'gini'
    selected_depth = 4

    if request.method == 'POST':
        try:
            query_cgpa = float(request.form.get('cgpa', 8.1))
            query_iq = float(request.form.get('iq', 112))
            query_comm = float(request.form.get('comm', 3.8))
            selected_criterion = request.form.get('criterion', 'gini')
            selected_depth = int(request.form.get('max_depth', 4))

            # Retrain if hyperparameter changed
            if selected_criterion != decision_tree_lab.criterion or selected_depth != decision_tree_lab.max_depth:
                decision_tree_lab.train_or_load(criterion=selected_criterion, max_depth=selected_depth, force_retrain=True)

            result = decision_tree_lab.predict(query_cgpa, query_iq, comm=query_comm)
            log_experiment('decision_tree', 'Decision Tree Classifier', 'Classification',
                           {'CGPA': query_cgpa, 'IQ': query_iq, 'Communication': query_comm, 'Depth': selected_depth},
                           result['prediction'], result['metrics'])
        except Exception as e:
            flash(f"Execution Error: {str(e)}", 'error')

    return render_template('experiments/decision_tree.html',
                           active_page='decision_tree',
                           details=details,
                           result=result,
                           query_cgpa=query_cgpa,
                           query_iq=query_iq,
                           query_comm=query_comm,
                           selected_criterion=selected_criterion,
                           selected_depth=selected_depth)


# ALGORITHM 06: K-Nearest Neighbors (KNN)
@app.route('/experiments/knn', methods=['GET', 'POST'])
def exp_knn():
    details = knn_lab.get_details()
    result = None
    query_sepal_len = None
    query_sepal_wid = None
    query_petal_len = None
    query_petal_wid = None
    selected_k = knn_lab.k_value
    selected_metric = knn_lab.metric

    if request.method == 'POST':
        try:
            query_sepal_len = float(request.form.get('sepal_len', 5.8))
            query_sepal_wid = float(request.form.get('sepal_wid', 3.0))
            query_petal_len = float(request.form.get('petal_len', 4.2))
            query_petal_wid = float(request.form.get('petal_wid', 1.3))
            selected_k = int(request.form.get('k_value', 5))
            selected_metric = request.form.get('metric', 'euclidean')

            result = knn_lab.predict(query_sepal_len, query_sepal_wid, query_petal_len, query_petal_wid,
                                    k_value=selected_k, metric=selected_metric)
            log_experiment('knn', 'K-Nearest Neighbors', 'Classification',
                           {'SepalLen': query_sepal_len, 'PetalLen': query_petal_len, 'K': selected_k, 'Metric': selected_metric},
                           result['prediction'], result['metrics'])
        except Exception as e:
            flash(f"Execution Error: {str(e)}", 'error')

    return render_template('experiments/knn.html',
                           active_page='knn',
                           details=details,
                           result=result,
                           query_sepal_len=query_sepal_len,
                           query_sepal_wid=query_sepal_wid,
                           query_petal_len=query_petal_len,
                           query_petal_wid=query_petal_wid,
                           selected_k=selected_k,
                           selected_metric=selected_metric)


# ALGORITHM 07: Gaussian Naive Bayes
@app.route('/experiments/naive_bayes', methods=['GET', 'POST'])
def exp_naive_bayes():
    details = naive_bayes_lab.get_details()
    result = None
    query_sepal_len = None
    query_sepal_wid = None
    query_petal_len = None
    query_petal_wid = None

    if request.method == 'POST':
        try:
            query_sepal_len = float(request.form.get('sepal_len', 5.5))
            query_sepal_wid = float(request.form.get('sepal_wid', 3.2))
            query_petal_len = float(request.form.get('petal_len', 3.8))
            query_petal_wid = float(request.form.get('petal_wid', 1.1))

            result = naive_bayes_lab.predict(query_sepal_len, query_sepal_wid, query_petal_len, query_petal_wid)
            log_experiment('naive_bayes', 'Gaussian Naive Bayes', 'Classification',
                           {'SepalLen': query_sepal_len, 'PetalLen': query_petal_len},
                           result['prediction'], result['metrics'])
        except Exception as e:
            flash(f"Execution Error: {str(e)}", 'error')

    return render_template('experiments/naive_bayes.html',
                           active_page='naive_bayes',
                           details=details,
                           result=result,
                           query_sepal_len=query_sepal_len,
                           query_sepal_wid=query_sepal_wid,
                           query_petal_len=query_petal_len,
                           query_petal_wid=query_petal_wid)


# ALGORITHM 08: Linear SVM
@app.route('/experiments/linear_svm', methods=['GET', 'POST'])
def exp_linear_svm():
    details = linear_svm_lab.get_details()
    result = None
    query_aptitude = None
    query_technical = None
    selected_c = linear_svm_lab.C

    if request.method == 'POST':
        try:
            query_aptitude = float(request.form.get('aptitude', 75.0))
            query_technical = float(request.form.get('technical', 68.0))
            selected_c = float(request.form.get('c_param', 1.0))

            result = linear_svm_lab.predict(query_aptitude, query_technical, C=selected_c)
            log_experiment('linear_svm', 'Linear SVM', 'Classification',
                           {'Aptitude': query_aptitude, 'Technical': query_technical, 'C': selected_c},
                           result['prediction'], result['metrics'])
        except Exception as e:
            flash(f"Execution Error: {str(e)}", 'error')

    return render_template('experiments/linear_svm.html',
                           active_page='linear_svm',
                           details=details,
                           result=result,
                           query_aptitude=query_aptitude,
                           query_technical=query_technical,
                           selected_c=selected_c)


# ALGORITHM 09: Non-Linear SVM
@app.route('/experiments/nonlinear_svm', methods=['GET', 'POST'])
def exp_nonlinear_svm():
    details = nonlinear_svm_lab.get_details()
    result = None
    query_skill_x = None
    query_skill_y = None
    selected_kernel = nonlinear_svm_lab.kernel
    selected_c = nonlinear_svm_lab.C
    selected_gamma = nonlinear_svm_lab.gamma

    if request.method == 'POST':
        try:
            query_skill_x = float(request.form.get('skill_x', 10.0))
            query_skill_y = float(request.form.get('skill_y', 10.0))
            selected_kernel = request.form.get('kernel', 'rbf')
            selected_c = float(request.form.get('c_param', 1.0))
            selected_gamma = request.form.get('gamma', 'scale')

            result = nonlinear_svm_lab.predict(query_skill_x, query_skill_y,
                                              kernel=selected_kernel, C=selected_c, gamma=selected_gamma)
            log_experiment('nonlinear_svm', 'Non-Linear SVM', 'Classification',
                           {'Skill_X': query_skill_x, 'Skill_Y': query_skill_y, 'Kernel': selected_kernel, 'C': selected_c},
                           result['prediction'], result['metrics'])
        except Exception as e:
            flash(f"Execution Error: {str(e)}", 'error')

    return render_template('experiments/nonlinear_svm.html',
                           active_page='nonlinear_svm',
                           details=details,
                           result=result,
                           query_skill_x=query_skill_x,
                           query_skill_y=query_skill_y,
                           selected_kernel=selected_kernel,
                           selected_c=selected_c,
                           selected_gamma=selected_gamma)


# ALGORITHM 10: Random Forest Classifier
@app.route('/experiments/random_forest', methods=['GET', 'POST'])
def exp_random_forest():
    details = random_forest_lab.get_details()
    result = None
    query_cgpa = None
    query_iq = None
    query_comm = None
    query_projects = None
    selected_estimators = random_forest_lab.n_estimators
    selected_depth = random_forest_lab.max_depth

    if request.method == 'POST':
        try:
            query_cgpa = float(request.form.get('cgpa', 8.2))
            query_iq = float(request.form.get('iq', 112))
            query_comm = float(request.form.get('comm', 3.7))
            query_projects = int(request.form.get('projects', 3))
            selected_estimators = int(request.form.get('n_estimators', 100))
            depth_input = request.form.get('max_depth', '5')
            selected_depth = int(depth_input) if depth_input != 'None' else None

            result = random_forest_lab.predict(query_cgpa, query_iq, query_comm, query_projects,
                                              n_estimators=selected_estimators, max_depth=selected_depth)
            log_experiment('random_forest', 'Random Forest Classifier', 'Classification',
                           {'CGPA': query_cgpa, 'IQ': query_iq, 'Trees': selected_estimators, 'MaxDepth': selected_depth},
                           result['prediction'], result['metrics'])
        except Exception as e:
            flash(f"Execution Error: {str(e)}", 'error')

    return render_template('experiments/random_forest.html',
                           active_page='random_forest',
                           details=details,
                           result=result,
                           query_cgpa=query_cgpa,
                           query_iq=query_iq,
                           query_comm=query_comm,
                           query_projects=query_projects,
                           selected_estimators=selected_estimators,
                           selected_depth=selected_depth)


# -------------------------------------------------------------
# 3. DATASET EXPLORER & DYNAMIC TRAINER
# -------------------------------------------------------------
def get_available_datasets():
    files = [f for f in os.listdir(DATASETS_DIR) if f.endswith('.csv')]
    if os.path.exists(UPLOADS_DIR):
        uploaded = [f"uploads/{f}" for f in os.listdir(UPLOADS_DIR) if f.endswith('.csv')]
        files.extend(uploaded)
    return files


def load_dataset_by_name(filename):
    if filename.startswith("uploads/"):
        real_path = os.path.join(UPLOADS_DIR, filename.replace("uploads/", ""))
    else:
        real_path = os.path.join(DATASETS_DIR, filename)
    if not os.path.exists(real_path):
        raise FileNotFoundError(f"Dataset {filename} does not exist.")
    return pd.read_csv(real_path)


@app.route('/dataset_explorer')
def dataset_explorer():
    available_datasets = get_available_datasets()
    selected_filename = request.args.get('file', 'student_placement.csv')
    if selected_filename not in available_datasets and available_datasets:
        selected_filename = available_datasets[0]

    try:
        df = load_dataset_by_name(selected_filename)
        preview_rows = df.head(15).to_dict(orient='records')
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        missing_counts = df.isnull().sum().to_dict()
        total_missing = sum(missing_counts.values())

        # Descriptive statistics
        stats_dict = {}
        if numeric_columns:
            stats_df = df[numeric_columns].describe().to_dict()
            stats_dict = stats_df

        dataset_info = {
            'rows': len(df),
            'columns_count': len(df.columns),
            'columns': df.columns.tolist(),
            'dtypes': dtypes,
            'numeric_columns': numeric_columns,
            'missing_counts': missing_counts,
            'total_missing': total_missing,
            'preview': preview_rows,
            'statistics': stats_dict
        }
    except Exception as e:
        flash(f"Error loading dataset: {str(e)}", 'error')
        dataset_info = {
            'rows': 0, 'columns_count': 0, 'columns': [], 'dtypes': {},
            'numeric_columns': [], 'missing_counts': {}, 'total_missing': 0,
            'preview': [], 'statistics': {}
        }

    return render_template('dataset_explorer.html',
                           active_page='dataset_explorer',
                           available_datasets=available_datasets,
                           selected_filename=selected_filename,
                           dataset_info=dataset_info,
                           train_result=None)


@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    if 'csv_file' not in request.files:
        flash('No file part provided.', 'error')
        return redirect(url_for('dataset_explorer'))
    file = request.files['csv_file']
    if file.filename == '':
        flash('No file selected.', 'error')
        return redirect(url_for('dataset_explorer'))
    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOADS_DIR, filename)
        file.save(save_path)
        flash(f'Dataset "{filename}" uploaded and ready for analysis!', 'info')
        return redirect(url_for('dataset_explorer', file=f"uploads/{filename}"))
    else:
        flash('Only standard .csv files are supported.', 'error')
        return redirect(url_for('dataset_explorer'))


@app.route('/train_custom_dataset', methods=['POST'])
def train_custom_dataset():
    dataset_file = request.form.get('dataset_file')
    target_col = request.form.get('target_col')
    feature_cols = request.form.getlist('feature_cols')
    algo_choice = request.form.get('algo_choice')

    if not feature_cols:
        flash("Please select at least one numeric input feature.", "error")
        return redirect(url_for('dataset_explorer', file=dataset_file))

    try:
        df = load_dataset_by_name(dataset_file)
        # Drop rows with NaN in features or target
        df_clean = df.dropna(subset=feature_cols + [target_col])

        X = df_clean[feature_cols].values
        y_raw = df_clean[target_col].values

        is_regression = (algo_choice == 'linear_regression')
        from sklearn.model_selection import train_test_split

        if is_regression:
            y = y_raw.astype(float)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = evaluate_regression(y_test, y_pred)
            algo_name = "Linear Regression (OLS)"
            task_type = "Regression"
        else:
            # Classification
            y_encoded, le = encode_labels(y_raw)
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
            X_train_s, X_test_s, scaler = scale_features(X_train, X_test)

            if algo_choice == 'logistic_regression':
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(max_iter=500, random_state=42)
                model.fit(X_train_s, y_train)
                y_pred = model.predict(X_test_s)
                algo_name = "Logistic Regression"
            elif algo_choice == 'decision_tree':
                from sklearn.tree import DecisionTreeClassifier
                model = DecisionTreeClassifier(max_depth=5, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                algo_name = "Decision Tree (max_depth=5)"
            elif algo_choice == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                algo_name = "Random Forest (50 Trees)"
            elif algo_choice == 'knn':
                from sklearn.neighbors import KNeighborsClassifier
                model = KNeighborsClassifier(n_neighbors=5)
                model.fit(X_train_s, y_train)
                y_pred = model.predict(X_test_s)
                algo_name = "K-Nearest Neighbors (K=5)"
            else:
                raise ValueError(f"Unknown algorithm: {algo_choice}")

            metrics = evaluate_classification(y_test, y_pred)
            task_type = "Classification"

        train_result = {
            'algorithm_name': algo_name,
            'task_type': task_type,
            'metrics': metrics,
            'features': feature_cols,
            'target': target_col
        }

        flash(f"Successfully trained {algo_name} on target '{target_col}'!", "info")

        # Reload page data with train_result populated
        available_datasets = get_available_datasets()
        preview_rows = df.head(15).to_dict(orient='records')
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        missing_counts = df.isnull().sum().to_dict()
        total_missing = sum(missing_counts.values())
        stats_dict = df[numeric_columns].describe().to_dict() if numeric_columns else {}

        dataset_info = {
            'rows': len(df),
            'columns_count': len(df.columns),
            'columns': df.columns.tolist(),
            'dtypes': dtypes,
            'numeric_columns': numeric_columns,
            'missing_counts': missing_counts,
            'total_missing': total_missing,
            'preview': preview_rows,
            'statistics': stats_dict
        }

        return render_template('dataset_explorer.html',
                               active_page='dataset_explorer',
                               available_datasets=available_datasets,
                               selected_filename=dataset_file,
                               dataset_info=dataset_info,
                               train_result=train_result)

    except Exception as e:
        flash(f"Model Training Error: {str(e)}", 'error')
        return redirect(url_for('dataset_explorer', file=dataset_file))


# -------------------------------------------------------------
# 4. MODEL PERFORMANCE PAGE
# -------------------------------------------------------------
@app.route('/performance')
def performance():
    # 1. Regression Models (R², MAE, MSE, RMSE)
    slr_m = slr_lab.metrics
    mlr_m = mlr_lab.metrics

    regression_models = [
        {
            'name': 'Simple Linear Regression',
            'dataset': 'student_package.csv',
            'features': 'CGPA (1 Feature)',
            'metrics': slr_m,
            'link': url_for('exp_slr')
        },
        {
            'name': 'Multiple Linear Regression',
            'dataset': 'student_placement.csv',
            'features': 'CGPA, IQ (2 Features)',
            'metrics': mlr_m,
            'link': url_for('exp_mlr')
        }
    ]

    reg_labels = [m['name'] for m in regression_models]
    reg_r2 = [m['metrics']['r2_score'] for m in regression_models]
    reg_mae = [m['metrics']['mae'] for m in regression_models]
    reg_rmse = [m['metrics']['rmse'] for m in regression_models]

    # 2. Classification Models (Accuracy, Precision, Recall, F1)
    classification_models = [
        {
            'name': 'Binary Logistic Regression',
            'dataset': 'student_placement.csv',
            'metrics': logistic_binary_lab.metrics,
            'notes': 'StandardScaler, C=1.0',
            'link': url_for('exp_logistic_binary')
        },
        {
            'name': 'Multiclass Logistic Regression',
            'dataset': 'student_placement.csv',
            'metrics': logistic_multiclass_lab.metrics,
            'notes': 'Softmax multinomial, 3 Tiers',
            'link': url_for('exp_logistic_multiclass')
        },
        {
            'name': 'Decision Tree Classifier',
            'dataset': 'student_placement.csv',
            'metrics': decision_tree_lab.metrics,
            'notes': f'Criterion={decision_tree_lab.criterion}, Depth={decision_tree_lab.max_depth}',
            'link': url_for('exp_decision_tree')
        },
        {
            'name': 'K-Nearest Neighbors (KNN)',
            'dataset': 'iris.csv',
            'metrics': knn_lab.metrics,
            'notes': f'K={knn_lab.k_value}, Metric={knn_lab.metric}',
            'link': url_for('exp_knn')
        },
        {
            'name': 'Gaussian Naive Bayes',
            'dataset': 'iris.csv',
            'metrics': naive_bayes_lab.metrics,
            'notes': 'Normal Likelihoods',
            'link': url_for('exp_naive_bayes')
        },
        {
            'name': 'Linear Support Vector Machine',
            'dataset': 'svm_twoclass.csv',
            'metrics': linear_svm_lab.metrics,
            'notes': f'Linear Kernel, C={linear_svm_lab.C}',
            'link': url_for('exp_linear_svm')
        },
        {
            'name': 'Non-Linear Support Vector Machine',
            'dataset': 'svm_nonlinear.csv',
            'metrics': nonlinear_svm_lab.metrics,
            'notes': f'Kernel={nonlinear_svm_lab.kernel.upper()}, C={nonlinear_svm_lab.C}, Gamma={nonlinear_svm_lab.gamma}',
            'link': url_for('exp_nonlinear_svm')
        },
        {
            'name': 'Random Forest Classifier',
            'dataset': 'student_placement.csv',
            'metrics': random_forest_lab.metrics,
            'notes': f'Trees={random_forest_lab.n_estimators}, Depth={random_forest_lab.max_depth}',
            'link': url_for('exp_random_forest')
        }
    ]

    cls_labels = [
        'Binary Logistic', 'Multiclass Logistic', 'Decision Tree',
        'KNN (K=5)', 'Naive Bayes', 'Linear SVM', 'Non-Linear SVM', 'Random Forest'
    ]
    cls_acc = [round(m['metrics']['accuracy'] * 100, 1) for m in classification_models]
    cls_prec = [round(m['metrics']['precision'] * 100, 1) for m in classification_models]
    cls_rec = [round(m['metrics']['recall'] * 100, 1) for m in classification_models]
    cls_f1 = [round(m['metrics']['f1_score'] * 100, 1) for m in classification_models]

    return render_template('performance.html',
                           active_page='performance',
                           regression_models=regression_models,
                           reg_labels=reg_labels,
                           reg_r2=reg_r2,
                           reg_mae=reg_mae,
                           reg_rmse=reg_rmse,
                           classification_models=classification_models,
                           cls_labels=cls_labels,
                           cls_acc=cls_acc,
                           cls_prec=cls_prec,
                           cls_rec=cls_rec,
                           cls_f1=cls_f1)


# -------------------------------------------------------------
# 5. ABOUT PROJECT ROUTE
# -------------------------------------------------------------
@app.route('/about')
def about():
    return render_template('about.html', active_page='about')


if __name__ == '__main__':
    print("=" * 65)
    print("  ML STUDIO: INTERACTIVE MACHINE LEARNING ALGORITHMS LAB")
    print("  Server launching on http://127.0.0.1:5000")
    print("=" * 65)
    app.run(host='127.0.0.1', port=5000, debug=True)
