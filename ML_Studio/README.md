# ML STUDIO – INTERACTIVE MACHINE LEARNING ALGORITHMS LAB

> **College-Level Machine Learning Mini-Project & Interactive Lab Dashboard**  
> Developed using **Python, Flask, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, and Chart.js**.

---

## 📌 Executive Summary

**ML Studio** is a centralized, educational Machine Learning laboratory uniting **10 foundational supervised Machine Learning algorithms** (2 Regression and 8 Classification models) into a unified, interactive web dashboard.

Unlike standard static demos or dummy prediction mockups, **every model in ML Studio is backed by an authentic, trained Scikit-learn pipeline** with:
- 80/20 or 75/25 train-test splitting
- Standard feature scaling (`StandardScaler`) to prevent data leakage
- Comprehensive evaluation metrics computed on unseen test sets ($R^2$, MAE, MSE, RMSE, Accuracy, Precision, Recall, F1-Score, and Confusion Matrices)
- Headless high-resolution visual plots (scatter lines, 2D decision boundaries, margins, support vectors, decision trees, and feature importances)
- Dynamic interactive Chart.js graphs
- Persistent experiment run logging via SQLite database (`database.db`)
- Dataset Explorer with custom CSV upload and on-demand model training

---

## 🏗️ Project Architecture & Directory Structure

```text
ML_Studio/
│
├── app.py                      # Primary Flask application and route orchestrator
├── test_app.py                 # Automated unit and integration test suite (16 tests)
├── generate_datasets.py        # Scientific dataset generator script
├── requirements.txt            # Project dependencies
├── database.db                 # SQLite experiment history database
├── README.md                   # Complete academic documentation & viva guide
│
├── datasets/                   # Benchmark CSV datasets
│   ├── student_package.csv     # SLR: CGPA vs Package (LPA)
│   ├── student_placement.csv   # MLR, Logistic, Trees, Ensembles: multi-feature profiles
│   ├── iris.csv                # KNN, Naive Bayes: 4 botanical features, 3 species
│   ├── svm_twoclass.csv        # Linear SVM: 2D Aptitude & Technical scores
│   └── svm_nonlinear.csv       # Non-Linear SVM: 2D non-linear moon manifolds
│
├── models/                     # Serialized trained Scikit-learn models (.joblib)
│
├── algorithms/                 # Encapsulated algorithm modules
│   ├── simple_linear_regression.py     # Algorithm 01: SLR
│   ├── multiple_linear_regression.py   # Algorithm 02: MLR
│   ├── logistic_binary.py              # Algorithm 03: Binary Logistic
│   ├── logistic_multiclass.py          # Algorithm 04: Multiclass Logistic
│   ├── decision_tree.py                # Algorithm 05: Decision Tree
│   ├── knn.py                          # Algorithm 06: K-Nearest Neighbors
│   ├── naive_bayes.py                  # Algorithm 07: Gaussian Naive Bayes
│   ├── linear_svm.py                   # Algorithm 08: Linear SVM
│   ├── nonlinear_svm.py                # Algorithm 09: Non-Linear SVM
│   └── random_forest.py                # Algorithm 10: Random Forest
│
├── templates/                  # Jinja2 HTML templates
│   ├── base.html               # Responsive master layout with academic sidebar
│   ├── dashboard.html          # Main overview dashboard with search & filter
│   ├── dataset_explorer.html   # Dataset inspector & custom model trainer
│   ├── performance.html        # Unified model performance matrix & Chart.js plots
│   ├── about.html              # Project architecture & viva voce preparation
│   │
│   └── experiments/            # Dedicated experiment interfaces
│       ├── slr.html                    # 01. SLR Interface
│       ├── mlr.html                    # 02. MLR Interface
│       ├── logistic_binary.html        # 03. Binary Logistic Interface
│       ├── logistic_multiclass.html    # 04. Multiclass Logistic Interface
│       ├── decision_tree.html          # 05. Decision Tree Interface
│       ├── knn.html                    # 06. KNN Interface
│       ├── naive_bayes.html            # 07. Naive Bayes Interface
│       ├── linear_svm.html             # 08. Linear SVM Interface
│       ├── nonlinear_svm.html          # 09. Non-Linear SVM Interface
│       └── random_forest.html          # 10. Random Forest Interface
│
├── static/                     # Web assets
│   ├── css/
│   │   └── style.css           # Clean academic navy blue & slate styling
│   └── js/
│       └── main.js             # Client-side filtering, presets & sidebar interactions
│
├── uploads/                    # Directory for user-uploaded CSV datasets
└── utils/                      # Reusable utilities
    ├── preprocessing.py        # Loading, splitting, scaling, encoding
    ├── evaluation.py           # Regression & classification metric calculations
    ├── visualization.py        # Matplotlib/Seaborn base64 plot generators
    └── database.py             # SQLite helper for experiment logging
```

---

## 🔬 Suite of 10 Machine Learning Algorithms

| # | Algorithm Name | Task Type | Scikit-learn Class | Features & Input | Target Output | Key Visualizations |
|---|----------------|-----------|--------------------|------------------|---------------|-------------------|
| **01** | **Simple Linear Regression** | Regression | `LinearRegression()` | Student CGPA | Annual Package in ₹ LPA | Scatter plot, fitted regression line, residual drop-lines |
| **02** | **Multiple Linear Regression** | Regression | `LinearRegression()` | CGPA, IQ Score | Expected Package in ₹ LPA | Actual vs. Predicted parity line ($y=x$), feature coefficient weights ($\beta$) |
| **03** | **Binary Logistic Regression** | Classification | `LogisticRegression()` | CGPA, IQ Score | Placement Status (Placed / Not Placed) | 2D Probability contour, decision boundary ($P=0.5$), confusion matrix |
| **04** | **Multiclass Logistic Regression** | Classification | `LogisticRegression(solver='lbfgs')` | CGPA, IQ Score | Performance Tier (Low / Average / High) | Softmax probability distribution bar chart, multiclass confusion matrix |
| **05** | **Decision Tree Classifier** | Classification | `DecisionTreeClassifier()` | CGPA, IQ, Communication Rating | Placement Status | High-resolution tree architecture via `plot_tree()`, Gini feature importance |
| **06** | **K-Nearest Neighbors (KNN)** | Classification | `KNeighborsClassifier()` | Sepal & Petal Dimensions (Iris) | Iris Species | $K$-value vs. Accuracy elbow curve ($K=1\dots15$), confusion matrix |
| **07** | **Gaussian Naive Bayes** | Classification | `GaussianNB()` | Sepal & Petal Dimensions (Iris) | Iris Species | Class posterior probability distribution, confusion matrix |
| **08** | **Linear SVM** | Classification | `SVC(kernel='linear')` | Aptitude & Technical Scores | Admission Decision (Admitted / Rejected) | Separating hyperplane ($w\cdot x + b = 0$), margin lines ($\pm 1$), circled support vectors |
| **09** | **Non-Linear SVM** | Classification | `SVC(kernel='rbf'/'poly')` | 2D Skill Manifolds (Moons) | Category Class-A / Class-B | Curved non-linear boundary contours, circled support vectors |
| **10** | **Random Forest Classifier** | Classification | `RandomForestClassifier()` | CGPA, IQ, Communication, Projects | Placement Status | Mean Decrease in Impurity (MDI) feature ranking, confusion matrix |

---

## 📊 Dataset Explorer & Custom Model Trainer

The **Dataset Explorer** tab provides:
1. **Preloaded Demonstration Datasets**:
   - `student_package.csv`: 250 records for Simple Linear Regression
   - `student_placement.csv`: 350 records across 7 features
   - `iris.csv`: 150 records across 4 botanical dimensions
   - `svm_twoclass.csv`: 200 linearly separable admissions profiles
   - `svm_nonlinear.csv`: 250 non-linear moon coordinates
2. **Custom CSV Upload**: Upload any `.csv` file with automatic validation.
3. **Data Inspection**: Shape, column names, inferred datatypes, missing values count.
4. **Descriptive Statistics**: Five-number summary (count, mean, std, min, 25%, 50%, 75%, max).
5. **On-Demand Dynamic Training**: Select any numeric/categorical target column and feature subset; train Linear Regression, Logistic Regression, Decision Tree, Random Forest, or KNN directly and receive instant test-set metrics!

---

## 📈 Model Performance Matrix

The **Model Performance** section segregates Regression and Classification metrics:
- **Regression Evaluation**:
  - $R^2$ Score (Coefficient of Determination)
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Side-by-side Chart.js visual comparison
- **Classification Evaluation**:
  - Accuracy Score
  - Precision (Weighted / Binary)
  - Recall (Sensitivity)
  - F1-Score (Harmonic Mean)
  - Grouped multi-bar Chart.js comparison across all 8 classification models

---

## 🚀 Quickstart & Setup Instructions

### 1. Prerequisites
Ensure Python 3.10+ (tested on Python 3.13) is installed.

### 2. Installation
Open a terminal in the project directory:
```bash
cd ML_Studio
pip install -r requirements.txt
```

### 3. Generate Datasets (if setting up fresh)
```bash
python generate_datasets.py
```

### 4. Run the Automated Test Suite
```bash
python -m unittest test_app.py
```
*Expected Output:*
```text
Ran 16 tests in ~11s
OK
```

### 5. Launch the Web Application
```bash
python app.py
```
Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## 🎓 College Viva Voce / Examination Preparation

### Q1: What is the primary difference between Regression and Classification?
- **Regression** predicts a continuous numeric value (e.g., salary package in ₹ LPA). Evaluation relies on error metrics like $R^2$, MAE, and MSE.
- **Classification** predicts a discrete qualitative category or label (e.g., Placed vs. Not Placed). Evaluation relies on threshold/confusion metrics like Accuracy, Precision, Recall, and F1-Score.

### Q2: Why is Ordinary Least Squares (OLS) used in Linear Regression?
OLS minimizes the sum of squared differences (residuals) between observed values $y_i$ and model predictions $\hat{y}_i$:
$$\text{RSS} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
It provides an optimal, unbiased closed-form analytical solution: $\beta = (X^T X)^{-1} X^T y$.

### Q3: Why does Logistic Regression use the Sigmoid function?
Standard linear regression yields unbounded predictions $(-\infty, +\infty)$, which are invalid for probabilities. The sigmoid activation function:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$
maps any real number $z \in \mathbb{R}$ into the strict probability interval $[0, 1]$.

### Q4: What is the "Kernel Trick" in Non-Linear SVM?
The Kernel Trick allows SVMs to find linear decision boundaries in an implicit, high-dimensional Hilbert space $\Phi(x)$ without calculating coordinates in that space. It evaluates inner products directly:
$$K(x, x') = \exp(-\gamma \|x - x'\|^2)$$
This enables efficient separation of complex non-linear manifolds with low computational overhead.

### Q5: How does Random Forest prevent overfitting compared to a single Decision Tree?
A single decision tree has low bias but high variance (overfits easily to training noise). Random Forest reduces variance via **Bagging (Bootstrap Aggregation)**:
1. Training $B$ individual trees on bootstrapped random subsets of the data.
2. Restricting candidate split features to a random subset (typically $\sqrt{p}$) at each node.
3. Aggregating all individual tree predictions by majority voting.

---

## 📜 Technology Stack

- **Backend Framework**: Flask 3.1
- **Machine Learning**: Scikit-learn 1.7, NumPy 2.3, Pandas 2.3, Joblib 1.5
- **Visualizations**: Matplotlib 3.10 (Agg Headless Backend), Seaborn 0.13, Chart.js
- **Database**: SQLite3
- **Frontend**: HTML5, Modern Semantic CSS3, Vanilla JavaScript (ES6)

---

## 🏆 Project Highlights for Evaluators

1. **Zero Hardcoded Metrics**: All numbers displayed (slopes, intercepts, accuracies, $R^2$, confusion matrices) are derived from genuine Scikit-learn models evaluated on real test splits.
2. **Interactive Hyperparameters**: Real-time experimentation with $K$ in KNN, regularization $C$ in SVM, Kernel types (RBF vs. Poly), Tree Depth in Decision Trees, and number of estimators in Random Forest.
3. **Publication-Quality Academic Visualizations**: Clean plots generated headlessly and rendered natively as base64 images alongside responsive Chart.js canvases.
4. **Clean Code & Robust Testing**: Fully unit-tested with 16 automated tests covering every endpoint, pipeline, and form.
