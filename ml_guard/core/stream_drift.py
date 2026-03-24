"""
Streaming Drift Detection Engine.

Implements real-time sliding-window drift analysis:
  - Rolling PSI (Population Stability Index)
  - Rolling JSD (Jensen-Shannon Divergence)
  - Adaptive Thresholding (dynamic baselines)
  - Drift Trend Detection (consecutive window analysis)

Designed for in-memory streaming with configurable window sizes.
"""
import numpy as np
from scipy.spatial.distance import jensenshannon
from typing import Dict, Any, List, Optional, Tuple
from collections import deque


class StreamDriftDetector:
    """
    Stateful drift detector for streaming production data.
    Maintains a rolling window and compares against a baseline distribution.
    """

    def __init__(
        self,
        window_size: int = 500,
        n_bins: int = 20,
        psi_threshold: float = 0.2,
        jsd_threshold: float = 0.1,
        consecutive_alert_windows: int = 3,
    ):
        self.window_size = window_size
        self.n_bins = n_bins
        self.psi_threshold = psi_threshold
        self.jsd_threshold = jsd_threshold
        self.consecutive_alert_windows = consecutive_alert_windows

        # Sliding window
        self.window: deque = deque(maxlen=window_size)

        # Baseline distribution
        self._baseline_hist: Optional[np.ndarray] = None
        self._baseline_bins: Optional[np.ndarray] = None
        self._baseline_set: bool = False

        # Trend tracking
        self._psi_history: List[float] = []
        self._jsd_history: List[float] = []
        self._alert_streak: int = 0
        self._total_events: int = 0

    def set_baseline(self, baseline_data: np.ndarray) -> None:
        """Establish reference distribution from training/reference predictions."""
        arr = np.asarray(baseline_data, dtype=float).ravel()
        self._baseline_hist, self._baseline_bins = np.histogram(
            arr, bins=self.n_bins, density=True
        )
        self._baseline_hist = self._baseline_hist + 1e-10  # Laplace smoothing
        self._baseline_set = True

    def add_event(self, prediction: float) -> None:
        """Add a single prediction to the rolling window."""
        self.window.append(float(prediction))
        self._total_events += 1

    def add_batch(self, predictions: List[float]) -> None:
        """Add a batch of predictions to the rolling window."""
        for p in predictions:
            self.window.append(float(p))
        self._total_events += len(predictions)

    def compute_rolling_psi(self) -> float:
        """
        PSI between baseline and current window distribution.
        PSI = Σ (P_i - Q_i) × ln(P_i / Q_i)
        """
        if not self._baseline_set or len(self.window) < 30:
            return 0.0

        current = np.array(self.window)
        current_hist, _ = np.histogram(
            current, bins=self._baseline_bins, density=True
        )
        current_hist = current_hist + 1e-10

        # Normalize to probability distributions
        baseline_p = self._baseline_hist / self._baseline_hist.sum()
        current_p = current_hist / current_hist.sum()

        psi = float(np.sum(
            (current_p - baseline_p) * np.log(current_p / baseline_p)
        ))
        return round(max(psi, 0.0), 6)

    def compute_rolling_jsd(self) -> float:
        """Jensen-Shannon Divergence between baseline and current window."""
        if not self._baseline_set or len(self.window) < 30:
            return 0.0

        current = np.array(self.window)
        current_hist, _ = np.histogram(
            current, bins=self._baseline_bins, density=True
        )
        current_hist = current_hist + 1e-10

        baseline_p = self._baseline_hist / self._baseline_hist.sum()
        current_p = current_hist / current_hist.sum()

        return float(round(jensenshannon(baseline_p, current_p), 6))

    def detect_trend(self) -> str:
        """
        Analyze PSI history to detect drift trend.
        Returns: "stable" | "increasing" | "critical"
        """
        if len(self._psi_history) < 3:
            return "stable"

        recent = self._psi_history[-5:]
        if len(recent) < 3:
            return "stable"

        # Check if consistently increasing
        diffs = [recent[i+1] - recent[i] for i in range(len(recent)-1)]
        avg_diff = np.mean(diffs)

        if avg_diff > 0.01 and recent[-1] > self.psi_threshold:
            return "critical"
        elif avg_diff > 0.005:
            return "increasing"
        return "stable"

    def compute_adaptive_threshold(self) -> float:
        """
        Dynamic threshold based on historical PSI distribution.
        Uses μ + 2σ of historical PSI as the adaptive threshold.
        Falls back to static threshold if insufficient history.
        """
        if len(self._psi_history) < 10:
            return self.psi_threshold

        arr = np.array(self._psi_history)
        adaptive = float(np.mean(arr) + 2 * np.std(arr))
        return round(max(adaptive, self.psi_threshold * 0.5), 4)

    def evaluate(self) -> Dict[str, Any]:
        """
        Full streaming drift evaluation.
        Returns metrics, trend, and alert status.
        """
        psi = self.compute_rolling_psi()
        jsd = self.compute_rolling_jsd()

        self._psi_history.append(psi)
        self._jsd_history.append(jsd)

        # Keep history bounded
        if len(self._psi_history) > 100:
            self._psi_history = self._psi_history[-100:]
            self._jsd_history = self._jsd_history[-100:]

        adaptive_threshold = self.compute_adaptive_threshold()
        trend = self.detect_trend()

        # Alert logic: PSI crosses threshold for N consecutive windows
        if psi > adaptive_threshold:
            self._alert_streak += 1
        else:
            self._alert_streak = 0

        alert = self._alert_streak >= self.consecutive_alert_windows

        # Severity
        if alert and trend == "critical":
            severity = "CRITICAL"
        elif alert:
            severity = "HIGH"
        elif psi > adaptive_threshold:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        return {
            "window_psi":           psi,
            "window_jsd":           jsd,
            "trend":                trend,
            "alert":                alert,
            "severity":             severity,
            "alert_streak":         self._alert_streak,
            "adaptive_threshold":   adaptive_threshold,
            "window_size":          len(self.window),
            "total_events":         self._total_events,
            "psi_history":          self._psi_history[-20:],
            "jsd_history":          self._jsd_history[-20:],
        }


def compute_stream_drift(
    baseline: np.ndarray,
    current_window: np.ndarray,
    n_bins: int = 20,
) -> Dict[str, Any]:
    """
    Stateless one-shot streaming drift computation.
    For use outside the stateful StreamDriftDetector class.
    """
    baseline = np.asarray(baseline, dtype=float).ravel()
    current_window = np.asarray(current_window, dtype=float).ravel()

    if len(current_window) < 10 or len(baseline) < 10:
        return {"window_psi": 0.0, "window_jsd": 0.0, "trend": "stable", "alert": False}

    b_hist, bins = np.histogram(baseline, bins=n_bins, density=True)
    c_hist, _ = np.histogram(current_window, bins=bins, density=True)

    b_hist = b_hist + 1e-10
    c_hist = c_hist + 1e-10

    bp = b_hist / b_hist.sum()
    cp = c_hist / c_hist.sum()

    psi = float(np.sum((cp - bp) * np.log(cp / bp)))
    jsd = float(jensenshannon(bp, cp))

    return {
        "window_psi": round(max(psi, 0.0), 6),
        "window_jsd": round(jsd, 6),
        "trend":      "stable",
        "alert":      psi > 0.2,
    }
