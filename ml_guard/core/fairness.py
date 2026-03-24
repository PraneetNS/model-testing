import numpy as np
from typing import Dict, Any, List, Optional
try:
    from fairlearn.metrics import (
        demographic_parity_difference,
        equal_opportunity_difference,
    )
    _has_fairlearn = True
except ImportError:
    _has_fairlearn = False


def _group_split(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Split y_true/y_pred arrays by unique values of the sensitive feature."""
    groups: Dict[str, Dict[str, np.ndarray]] = {}
    for val in np.unique(sensitive):
        mask = sensitive == val
        groups[str(val)] = {
            "y_true": y_true[mask],
            "y_pred": y_pred[mask],
            "count":  int(np.sum(mask)),
        }
    return groups


def statistical_parity_difference(
    y_pred: np.ndarray,
    sensitive: np.ndarray,
) -> float:
    """
    SPD = P(ŷ=1 | S=privileged) − P(ŷ=1 | S=unprivileged)
    Range: [-1, 1].  |SPD| close to 0 is fair.
    """
    if _has_fairlearn:
        return float(demographic_parity_difference(None, y_pred, sensitive_features=sensitive))
    
    groups = np.unique(sensitive)
    if len(groups) < 2:
        return 0.0
    rates = []
    for g in groups:
        mask = sensitive == g
        rates.append(float(np.mean(y_pred[mask])))
    return float(max(rates) - min(rates))


def equal_opportunity_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
) -> float:
    """
    EOD = TPR(privileged) − TPR(unprivileged)
    Only considers positive class (y_true == 1).
    """
    if _has_fairlearn:
        return float(equal_opportunity_difference(y_true, y_pred, sensitive_features=sensitive))

    groups = np.unique(sensitive)
    if len(groups) < 2:
        return 0.0
    tprs = []
    for g in groups:
        mask = sensitive == g
        positives = y_true[mask] == 1
        if np.sum(positives) == 0:
            tprs.append(0.0)
            continue
        tpr = float(np.mean(y_pred[mask][positives] == 1))
        tprs.append(tpr)
    return float(max(tprs) - min(tprs))


def disparate_impact_ratio(
    y_pred: np.ndarray,
    sensitive: np.ndarray,
) -> float:
    """
    DIR = P(ŷ=1 | S=unprivileged) / P(ŷ=1 | S=privileged)
    DIR ∈ [0.8, 1.25] is generally considered fair (80% rule).
    Returns 1.0 if no predictions are positive.
    """
    groups = np.unique(sensitive)
    if len(groups) < 2:
        return 1.0
    rates = []
    for g in groups:
        mask = sensitive == g
        rate = float(np.mean(y_pred[mask]))
        rates.append(rate)
    max_rate = max(rates)
    min_rate = min(rates)
    if max_rate == 0:
        return 1.0
    return float(min_rate / max_rate)


def group_performance_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Per-group accuracy, precision, recall, and F1.
    """
    breakdown = {}
    for val in np.unique(sensitive):
        mask = sensitive == val
        yt = y_true[mask]
        yp = y_pred[mask]
        n = int(np.sum(mask))

        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        tn = int(np.sum((yt == 0) & (yp == 0)))

        accuracy  = (tp + tn) / n if n > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        breakdown[str(val)] = {
            "count":     n,
            "accuracy":  round(accuracy, 4),
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "positive_rate": round(float(np.mean(yp)), 4),
        }
    return breakdown


def compute_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Full fairness analysis.

    Parameters:
        y_true:      Ground truth labels (0/1).
        y_pred:      Predicted labels (0/1).
        sensitive:   Sensitive feature values per sample.
        thresholds:  Optional policy thresholds {max_spd, min_dir, max_eod}.

    Returns:
        Complete fairness report with flag and group breakdown.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    sensitive = np.asarray(sensitive).ravel()

    if thresholds is None:
        thresholds = {"max_spd": 0.1, "min_dir": 0.8, "max_eod": 0.1}

    spd = statistical_parity_difference(y_pred, sensitive)
    eod = equal_opportunity_difference(y_true, y_pred, sensitive)
    dir_val = disparate_impact_ratio(y_pred, sensitive)
    group_metrics = group_performance_breakdown(y_true, y_pred, sensitive)

    # Fairness flag: True if ALL within thresholds
    spd_ok = abs(spd) <= thresholds.get("max_spd", 0.1)
    dir_ok = dir_val >= thresholds.get("min_dir", 0.8)
    eod_ok = abs(eod) <= thresholds.get("max_eod", 0.1)
    fairness_flag = spd_ok and dir_ok and eod_ok

    # Fairness subscore for governance integration [0, 1]
    # Penalize deviations from ideal
    spd_penalty = min(abs(spd) / 0.3, 1.0)  # normalized penalty
    eod_penalty = min(abs(eod) / 0.3, 1.0)
    dir_penalty = max(0, 1.0 - dir_val) / 0.5  # how far below 1.0
    dir_penalty = min(dir_penalty, 1.0)

    fairness_subscore = max(0.0, 1.0 - (spd_penalty * 0.4 + eod_penalty * 0.3 + dir_penalty * 0.3))

    return {
        "statistical_parity_diff": round(spd, 4),
        "equal_opportunity_diff":  round(eod, 4),
        "disparate_impact_ratio":  round(dir_val, 4),
        "group_metrics":           group_metrics,
        "fairness_flag":           fairness_flag,
        "fairness_subscore":       round(fairness_subscore, 4),
        "thresholds_used":         thresholds,
        "violations": {
            "spd_violated": not spd_ok,
            "dir_violated": not dir_ok,
            "eod_violated": not eod_ok,
        },
    }
