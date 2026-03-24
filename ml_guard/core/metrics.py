import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
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
