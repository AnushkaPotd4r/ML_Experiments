
import pandas as pd
import numpy as np
import joblib
import sklearn

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

print("Scikit-learn version:", sklearn.__version__)

# Load dataset
df = pd.read_csv("diabetes.csv")

# Remove duplicate rows
df = df.drop_duplicates().copy()

# Replace invalid zero measurements with missing values
zero_as_missing = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for col in zero_as_missing:
    df[col] = df[col].replace(0, np.nan)

# Classification: predict Outcome
classification_features = [
    col for col in df.columns if col != "Outcome"
]

X_classification = df[classification_features]
y_classification = df["Outcome"]

classifier_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("selector", SelectKBest(score_func=f_classif, k="all")),
    ("model", RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

classifier_pipeline.fit(X_classification, y_classification)

# Save classifier bundle for Flask
joblib.dump({
    "pipeline": classifier_pipeline,
    "features": classification_features
}, "diabetes_classifier.joblib")

# Regression: predict Glucose using other columns
regression_features = [
    col for col in df.columns
    if col not in ["Glucose", "Outcome"]
]

X_regression = df[regression_features]
y_regression = df["Glucose"]

# Rows with missing target values cannot be used for training
valid_rows = y_regression.notna()

regressor_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("selector", SelectKBest(score_func=f_regression, k="all")),
    ("model", RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    ))
])

regressor_pipeline.fit(
    X_regression.loc[valid_rows],
    y_regression.loc[valid_rows]
)

# Save regressor bundle for Flask
joblib.dump({
    "pipeline": regressor_pipeline,
    "features": regression_features
}, "glucose_regressor.joblib")

print("\nBoth models trained and saved successfully!")
print("Created: diabetes_classifier.joblib")
print("Created: glucose_regressor.joblib")
