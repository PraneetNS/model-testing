"""
Calibration analysis:
- Brier Score
- Reliability diagram data (calibration curve)
- Overconfidence detection
"""
import numpy as np
from sklearn.calibration import calibration_curve
from .exceptions import MetricComputationError

def compute_brier_score(y_true, y_prob):
    """
    Brier Score = (1/N) * Σ (p_i - y_i)^2
    Lower is better. 0.0 is perfect. 0.25 is naive (always 0.5).
    """
    try:
        y_true = np.array(y_true, dtype=float)
        y_prob = np.array(y_prob, dtype=float)
        if y_prob.ndim == 2:
            y_prob = y_prob[:, 1]  # Binary: use positive class prob
        return float(np.mean((y_prob - y_true) ** 2))
    except Exception as e:
        raise MetricComputationError(f"Brier score failed: {e}")

def compute_calibration(y_true, y_prob, n_bins=10):
    """
    Compute calibration curve data and ECE (Expected Calibration Error).
    Returns fraction_of_positives vs mean_predicted_value per bin.
    """
    try:
        y_true = np.array(y_true, dtype=float)
        y_prob = np.array(y_prob, dtype=float)
        if y_prob.ndim == 2:
            y_prob = y_prob[:, 1]

        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy='uniform'
        )

        # Expected Calibration Error
        # ECE = Σ |accuracy_bin - confidence_bin| * (n_bin / N)
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob, bins[1:-1])
        ece = 0.0
        for b in range(n_bins):
            mask = bin_indices == b
            if mask.sum() == 0:
                continue
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += np.abs(bin_acc - bin_conf) * mask.sum() / len(y_true)

        brier = compute_brier_score(y_true, y_prob)
        overconfident = brier > 0.15 and float(mean_predicted_value.mean()) > 0.8

        return {
            "brier_score": round(brier, 6),
            "ece": round(ece, 6),
            "overconfident_flag": overconfident,
            "calibration_curve": {
                "fraction_of_positives": fraction_of_positives.tolist(),
                "mean_predicted_value": mean_predicted_value.tolist(),
            },
        }
    except Exception as e:
        raise MetricComputationError(f"Calibration computation failed: {e}")
