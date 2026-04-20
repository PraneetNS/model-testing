import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score
from .exceptions import MetricComputationError

def compute_accuracy(y_true, y_pred, y_prob=None):
    try:
        return float(accuracy_score(y_true, y_pred))
    except Exception as e:
        raise MetricComputationError(f"Failed to compute accuracy: {str(e)}")

def compute_f1(y_true, y_pred, y_prob=None):
    try:
        is_binary = len(np.unique(y_true)) <= 2
        avg_mode = 'binary' if is_binary else 'weighted'
        return float(f1_score(y_true, y_pred, average=avg_mode))
    except Exception as e:
        raise MetricComputationError(f"Failed to compute F1 score: {str(e)}")

def compute_roc_auc(y_true, y_pred, y_prob=None):
    if y_prob is None:
        raise MetricComputationError("ROC-AUC requires prediction probabilities (y_prob).")
    try:
        if len(y_prob.shape) == 1 or y_prob.shape[1] == 1:
            return float(roc_auc_score(y_true, y_prob))
        else:
            return float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
    except Exception as e:
        raise MetricComputationError(f"Failed to compute ROC-AUC: {str(e)}")

def compute_mse(y_true, y_pred):
    try:
        return float(mean_squared_error(y_true, y_pred))
    except Exception as e:
        raise MetricComputationError(f"Failed to compute MSE: {str(e)}")

def compute_rmse(y_true, y_pred):
    try:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    except Exception as e:
        raise MetricComputationError(f"Failed to compute RMSE: {str(e)}")

def compute_r2(y_true, y_pred):
    try:
        return float(r2_score(y_true, y_pred))
    except Exception as e:
        raise MetricComputationError(f"Failed to compute R2: {str(e)}")

def detect_task_type(y_true):
    """Simple heuristic to detect classification vs regression."""
    y = np.array(y_true)
    unique_vals = np.unique(y[~np.isnan(y)])
    if len(unique_vals) <= 20 and np.all(unique_vals.astype(float) == unique_vals.astype(int)):
        return "classification"
    return "regression"
