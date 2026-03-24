"""
Feature Leakage Detection using Mutual Information.
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from .exceptions import MetricComputationError


def detect_leakage(X: pd.DataFrame, y: np.ndarray, task: str = "classification", threshold: float = 0.85) -> dict:
    """
    Compute Mutual Information for each feature with the target.
    If any feature has MI > threshold * max(MI), flag as potential leakage.

    Returns:
        mi_scores: MI per feature (normalized to [0, 1])
        leakage_suspects: features exceeding the threshold
        risk_level: 'NONE', 'MODERATE', 'HIGH'
    """
    try:
        X_numeric = X.select_dtypes(include=[np.number])
        if X_numeric.shape[1] == 0:
            raise MetricComputationError("No numeric features available for leakage detection.")

        X_filled = X_numeric.fillna(X_numeric.median())
        y_arr = np.array(y)

        if task == "classification":
            mi_raw = mutual_info_classif(X_filled, y_arr, random_state=42)
        else:
            mi_raw = mutual_info_regression(X_filled, y_arr, random_state=42)

        max_mi = mi_raw.max() if mi_raw.max() > 0 else 1.0
        mi_normalized = mi_raw / max_mi

        mi_scores = {col: round(float(mi), 4) for col, mi in zip(X_numeric.columns, mi_normalized)}

        # Flag features with very high MI relative to max
        suspects = {col: score for col, score in mi_scores.items() if score >= threshold}

        risk_level = "NONE"
        if len(suspects) > 0:
            max_suspect_score = max(suspects.values())
            risk_level = "HIGH" if max_suspect_score >= 0.95 else "MODERATE"

        return {
            "mi_scores": dict(sorted(mi_scores.items(), key=lambda x: -x[1])),
            "leakage_suspects": suspects,
            "risk_level": risk_level,
            "threshold_used": threshold,
        }
    except MetricComputationError:
        raise
    except Exception as e:
        raise MetricComputationError(f"Leakage detection failed: {e}")
