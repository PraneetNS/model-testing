import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon
from typing import Literal

from ..models import DriftReport, FeatureDrift

def _calculate_psi(expected, actual, buckets=10):
    def scale_range(input_val, min_val, max_val):
        input_val += -(np.min(input_val))
        input_val /= np.max(input_val) / (max_val - min_val)
        input_val += min_val
        return input_val

    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100
    try:
        breakpoints = scale_range(breakpoints, np.min(expected), np.max(expected))
    except Exception:
        pass
        
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)

    def sub_psi(e_perc, a_perc):
        if a_perc == 0:
            a_perc = 0.0001
        if e_perc == 0:
            e_perc = 0.0001
        value = (e_perc - a_perc) * np.log(e_perc / a_perc)
        return value

    psi_value = np.sum(sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents)))
    return psi_value


def detect_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, method: Literal["psi", "ks", "js"] = "psi") -> DriftReport:
    """
    Computes the selected drift metric for each column.
    """
    per_feature = []
    overall_drift_detected = False

    thresholds = {
        "psi": 0.2, # > 0.2 indicates significant drift
        "ks": 0.1,  # typical p-value or statistic threshold (we use statistic here for simplicity: >0.1 is drift)
        "js": 0.1   # distance > 0.1
    }
    threshold = thresholds.get(method, 0.1)

    common_cols = set(reference_df.columns).intersection(current_df.columns)
    
    for col in common_cols:
        ref_col = reference_df[col].dropna()
        cur_col = current_df[col].dropna()

        if not pd.api.types.is_numeric_dtype(ref_col) or not pd.api.types.is_numeric_dtype(cur_col):
            continue

        statistic = 0.0
        try:
            if method == "psi":
                statistic = _calculate_psi(ref_col.values, cur_col.values)
            elif method == "ks":
                stat, p_value = ks_2samp(ref_col.values, cur_col.values)
                statistic = stat
            elif method == "js":
                # Need probability distributions, compute histograms first
                hist_ref, bin_edges = np.histogram(ref_col.values, bins='auto', density=True)
                hist_cur, _ = np.histogram(cur_col.values, bins=bin_edges, density=True)
                statistic = jensenshannon(hist_ref, hist_cur)
        except Exception:
            statistic = 0.0
            
        is_drifted = bool(statistic > threshold)
        if is_drifted:
            overall_drift_detected = True
            
        per_feature.append(FeatureDrift(
            feature=col,
            statistic=float(statistic),
            threshold=threshold,
            drifted=is_drifted
        ))

    return DriftReport(
        overall_drift_detected=overall_drift_detected,
        per_feature=per_feature,
        method=method,
        reference_rows=len(reference_df),
        current_rows=len(current_df)
    )
