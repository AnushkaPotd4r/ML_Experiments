"""
Preprocessing utilities for ML Studio.
Provides helpers for dataset loading, train-test splitting, feature scaling, and categorical encoding.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")


def load_dataset(filename):
    """Load a CSV dataset from the datasets directory."""
    path = os.path.join(DATASETS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file '{filename}' not found at {path}")
    return pd.read_csv(path)


def split_data(X, y, test_size=0.2, random_state=42, stratify=None):
    """
    Split data into training and testing sets cleanly with optional stratification.
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )


def scale_features(X_train, X_test=None, scaler_type='standard'):
    """
    Fit scaler on training data and transform train (and optionally test/single inputs).
    Prevents data leakage.
    """
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")

    X_train_scaled = scaler.fit_transform(X_train)
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    return X_train_scaled, scaler


def encode_labels(y):
    """
    Encode string labels into integer indices and return encoder for inverse mapping.
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le
