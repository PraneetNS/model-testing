import numpy as np
import pandas as pd
from scipy import stats
from .exceptions import MetricComputationError

def compute_psi(expected, actual, buckets=10):
    """PSI = Σ (actual_i - expected_i) * ln(actual_i / expected_i)"""
    try:
        expected = np.array(expected, dtype=float)
        actual = np.array(actual, dtype=float)
        expected = expected[~np.isnan(expected)]
        actual = actual[~np.isnan(actual)]
        if len(expected) == 0 or len(actual) == 0:
            raise ValueError("Arrays must contain non-NaN values.")
        breakpoints = np.unique(np.percentile(expected, np.linspace(0, 100, buckets + 1)))
        if len(breakpoints) < 2:
            breakpoints = np.unique(expected)
            if len(breakpoints) < 2:
                return 0.0
            breakpoints = np.append(breakpoints, breakpoints[-1] + 1)
        breakpoints[0] = min(breakpoints[0], np.min(actual))
        breakpoints[-1] = max(breakpoints[-1], np.max(actual)) + 1e-5
        exp_f = np.histogram(expected, breakpoints)[0] / len(expected)
        act_f = np.histogram(actual, breakpoints)[0] / len(actual)
        exp_f = np.where(exp_f == 0, 1e-4, exp_f)
        act_f = np.where(act_f == 0, 1e-4, act_f)
        return float(np.sum((act_f - exp_f) * np.log(act_f / exp_f)))
    except Exception as e:
        raise MetricComputationError(f"Failed to compute PSI: {e}")

def compute_ks(expected, actual):
    """Kolmogorov-Smirnov two-sample test via scipy."""
    try:
        expected = np.array(expected, dtype=float)
        actual = np.array(actual, dtype=float)
        expected = expected[~np.isnan(expected)]
        actual = actual[~np.isnan(actual)]
        stat, pval = stats.ks_2samp(expected, actual)
        return float(stat), float(pval)
    except Exception as e:
        raise MetricComputationError(f"Failed to compute KS statistic: {e}")

def compute_jsd(expected, actual, bins=20):
    """
    Jensen-Shannon Divergence.
    JSD(P||Q) = 0.5*KL(P||M) + 0.5*KL(Q||M), where M = 0.5*(P+Q)
    Bounded [0, 1] when using log base 2.
    """
    try:
        expected = np.array(expected, dtype=float)
        actual = np.array(actual, dtype=float)
        expected = expected[~np.isnan(expected)]
        actual = actual[~np.isnan(actual)]
        if len(expected) == 0 or len(actual) == 0:
            raise ValueError("Empty arrays.")
        all_data = np.concatenate([expected, actual])
        edges = np.histogram_bin_edges(all_data, bins=bins)
        P, _ = np.histogram(expected, bins=edges, density=True)
        Q, _ = np.histogram(actual, bins=edges, density=True)
        # Normalize to probability
        P = P + 1e-10
        Q = Q + 1e-10
        P = P / P.sum()
        Q = Q / Q.sum()
        M = 0.5 * (P + Q)

        def kl(a, b):
            return np.sum(a * np.log2(a / b))

        jsd = 0.5 * kl(P, M) + 0.5 * kl(Q, M)
        return float(np.clip(jsd, 0.0, 1.0))
    except Exception as e:
        raise MetricComputationError(f"Failed to compute JSD: {e}")

def compute_target_drift(y_train, y_val, task="classification"):
    """
    Check P(y_train) vs P(y_val).
    - Classification: Chi-square test on class frequencies.
    - Regression: KS test.
    """
    try:
        y_train = np.array(y_train)
        y_val = np.array(y_val)
        if task in ("classification", "classifier"):
            classes = np.union1d(np.unique(y_train), np.unique(y_val))
            obs_train = np.array([np.sum(y_train == c) for c in classes]) + 1  # Laplace smooth
            obs_val   = np.array([np.sum(y_val   == c) for c in classes]) + 1
            # Scale val to same total as train
            obs_val_scaled = obs_val / obs_val.sum() * obs_train.sum()
            stat, pval = stats.chisquare(obs_train, f_exp=obs_val_scaled)
            return {
                "test": "chi_square",
                "statistic": float(stat),
                "p_value": float(pval),
                "drifted": bool(pval < 0.05),
                "class_distribution_train": {str(c): int(np.sum(y_train == c)) for c in classes},
                "class_distribution_val": {str(c): int(np.sum(y_val == c)) for c in classes},
            }
        else:
            stat, pval = stats.ks_2samp(y_train.astype(float), y_val.astype(float))
            return {
                "test": "ks",
                "statistic": float(stat),
                "p_value": float(pval),
                "drifted": bool(pval < 0.05),
            }
    except Exception as e:
        raise MetricComputationError(f"Target drift computation failed: {e}")

def compute_feature_drift_report(X_train: pd.DataFrame, X_val: pd.DataFrame, psi_threshold=0.2, jsd_threshold=0.1):
    """Full per-feature drift report: PSI, KS, and JSD."""
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    report = {}
    for col in numeric_cols:
        if col not in X_val.columns:
            continue
        psi = compute_psi(X_train[col].values, X_val[col].values)
        ks_stat, ks_pval = compute_ks(X_train[col].values, X_val[col].values)
        jsd = compute_jsd(X_train[col].values, X_val[col].values)
        report[col] = {
            "PSI": round(psi, 6),
            "KS_Stat": round(ks_stat, 6),
            "KS_pval": round(ks_pval, 6),
            "JSD": round(jsd, 6),
            "drift_flag": psi > psi_threshold or jsd > jsd_threshold,
        }
    # Top 5 drifted features by JSD
    sorted_by_jsd = sorted(report.items(), key=lambda x: x[1]["JSD"], reverse=True)
    top5 = [col for col, _ in sorted_by_jsd[:5]]
    return report, top5
