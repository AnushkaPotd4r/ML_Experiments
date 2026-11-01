"""
Evaluation utilities for ML Studio.
Computes standard Scikit-learn performance metrics for regression and classification models.
"""

import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_regression(y_true, y_pred):
    """
    Compute standard regression evaluation metrics.
    Returns:
        dict: r2, mae, mse, rmse
    """
    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))

    return {
        'r2_score': round(r2, 4),
        'mae': round(mae, 4),
        'mse': round(mse, 4),
        'rmse': round(rmse, 4)
    }


def evaluate_classification(y_true, y_pred, labels=None):
    """
    Compute standard classification evaluation metrics.
    Returns:
        dict: accuracy, precision, recall, f1, confusion_matrix, report
    """
    unique_labels = np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)]))
    is_binary = len(unique_labels) == 2
    avg_mode = 'binary' if is_binary else 'weighted'

    pos_label = 1 if is_binary and (1 in unique_labels) else (unique_labels[1] if is_binary else 1)

    try:
        acc = float(accuracy_score(y_true, y_pred))
        if is_binary:
            prec = float(precision_score(y_true, y_pred, average='binary', pos_label=pos_label, zero_division=0))
            rec = float(recall_score(y_true, y_pred, average='binary', pos_label=pos_label, zero_division=0))
            f1 = float(f1_score(y_true, y_pred, average='binary', pos_label=pos_label, zero_division=0))
        else:
            prec = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
            rec = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
            f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    except Exception:
        # Fallback to weighted
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
        rec = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    report_dict = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    return {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'confusion_matrix': cm,
        'classification_report': report_dict,
        'labels': [str(l) for l in (labels if labels is not None else unique_labels)]
    }
