"""
Script to generate realistic, scientifically sound demonstration datasets for ML Studio:
1. student_package.csv (Simple Linear Regression)
2. student_placement.csv (Multiple Linear Regression, Binary/Multiclass Logistic, Trees, Ensembles)
3. iris.csv (Iris dataset for KNN, Naive Bayes, Multiclass)
4. svm_twoclass.csv (Linearly separable 2D dataset for Linear SVM)
5. svm_nonlinear.csv (Non-linear concentric circles / moons for Non-Linear SVM)
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, make_moons, make_classification

np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
os.makedirs(DATASETS_DIR, exist_ok=True)

# 1. student_package.csv (Simple Linear Regression: CGPA -> Package_LPA)
n_slr = 250
cgpa_slr = np.round(np.random.uniform(5.0, 9.8, n_slr), 2)
# Package = 2.8 * CGPA - 9.5 + noise
package_slr = 2.65 * cgpa_slr - 9.2 + np.random.normal(0, 1.25, n_slr)
# Ensure realistic bounds (2.5 LPA minimum to 25.0 LPA max)
package_slr = np.round(np.clip(package_slr, 2.5, 25.0), 2)

df_slr = pd.DataFrame({
    'CGPA': cgpa_slr,
    'Package_LPA': package_slr
})
df_slr.to_csv(os.path.join(DATASETS_DIR, "student_package.csv"), index=False)
print("Generated student_package.csv:", df_slr.shape)

# 2. student_placement.csv (Multiple features: CGPA, IQ, Communication_Rating, Projects_Count)
n_placement = 350
cgpa_pl = np.round(np.random.uniform(5.2, 9.8, n_placement), 2)
iq_pl = np.random.randint(85, 145, n_placement)
comm_pl = np.round(np.random.uniform(1.0, 5.0, n_placement), 1)
projects_pl = np.random.randint(1, 8, n_placement)

# Package calculation: MLR
package_mlr = (2.2 * cgpa_pl) + (0.12 * iq_pl) + (0.85 * comm_pl) + (0.5 * projects_pl) - 15.0 + np.random.normal(0, 1.1, n_placement)
package_mlr = np.round(np.clip(package_mlr, 3.0, 30.0), 2)

# Placement Status (Binary): depends on CGPA, IQ, Communication
placement_score = (cgpa_pl - 6.5) * 1.5 + (iq_pl - 105) * 0.08 + (comm_pl - 3.0) * 0.9 + (projects_pl - 2) * 0.4
prob_placement = 1 / (1 + np.exp(-placement_score))
placed_binary = (prob_placement > 0.48).astype(int)
placement_labels = np.where(placed_binary == 1, 'Placed', 'Not Placed')

# Performance Category (Multiclass: Low, Average, High)
combined_score = 0.45 * (cgpa_pl / 10.0) + 0.35 * (iq_pl / 140.0) + 0.20 * (comm_pl / 5.0)
perf_category = []
for s in combined_score:
    if s < 0.65:
        perf_category.append('Low Performance')
    elif s < 0.78:
        perf_category.append('Average Performance')
    else:
        perf_category.append('High Performance')

df_placement = pd.DataFrame({
    'CGPA': cgpa_pl,
    'IQ': iq_pl,
    'Communication_Rating': comm_pl,
    'Projects_Count': projects_pl,
    'Package_LPA': package_mlr,
    'Placement_Status': placement_labels,
    'Performance_Category': perf_category
})
df_placement.to_csv(os.path.join(DATASETS_DIR, "student_placement.csv"), index=False)
print("Generated student_placement.csv:", df_placement.shape)

# 3. iris.csv (Classic classification dataset)
iris = load_iris(as_frame=True)
df_iris = iris.frame.copy()
df_iris.columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species_Code']
species_map = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
df_iris['Species'] = df_iris['Species_Code'].map(species_map)
df_iris.drop(columns=['Species_Code'], inplace=True)
df_iris.to_csv(os.path.join(DATASETS_DIR, "iris.csv"), index=False)
print("Generated iris.csv:", df_iris.shape)

# 4. svm_twoclass.csv (Linearly separable 2D dataset)
X_svm_lin, y_svm_lin = make_classification(
    n_samples=200, n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=1, class_sep=1.35, random_state=42
)
df_svm_lin = pd.DataFrame(X_svm_lin, columns=['Aptitude_Score', 'Technical_Score'])
df_svm_lin['Aptitude_Score'] = np.round((df_svm_lin['Aptitude_Score'] * 12 + 70).clip(40, 100), 1)
df_svm_lin['Technical_Score'] = np.round((df_svm_lin['Technical_Score'] * 12 + 65).clip(35, 100), 1)
df_svm_lin['Admitted'] = np.where(y_svm_lin == 1, 'Admitted', 'Rejected')
df_svm_lin.to_csv(os.path.join(DATASETS_DIR, "svm_twoclass.csv"), index=False)
print("Generated svm_twoclass.csv:", df_svm_lin.shape)

# 5. svm_nonlinear.csv (Non-linear dataset: Moons structure)
X_svm_nl, y_svm_nl = make_moons(n_samples=250, noise=0.18, random_state=42)
df_svm_nl = pd.DataFrame(X_svm_nl, columns=['Skill_X', 'Skill_Y'])
df_svm_nl['Skill_X'] = np.round(df_svm_nl['Skill_X'] * 5 + 10, 2)
df_svm_nl['Skill_Y'] = np.round(df_svm_nl['Skill_Y'] * 5 + 10, 2)
df_svm_nl['Category'] = np.where(y_svm_nl == 1, 'Class-A', 'Class-B')
df_svm_nl.to_csv(os.path.join(DATASETS_DIR, "svm_nonlinear.csv"), index=False)
print("Generated svm_nonlinear.csv:", df_svm_nl.shape)

print("All datasets generated successfully!")
