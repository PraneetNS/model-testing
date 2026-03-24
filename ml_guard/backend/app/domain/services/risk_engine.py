"""
ML Guard — Model Risk Score Engine (v2)
========================================
Pure-logic service (no DB access).
Computes a weighted, deterministic risk score (0–100) from audit metrics.

Weights:
  Accuracy Delta:        25%
  PSI (drift):           30%
  Brier Score:           15%
  Drifted Feature Count: 15%
  Calibration Issues:    15%

Risk Levels:
  0–30  = LOW
  31–60 = MEDIUM
  61–80 = HIGH
  81–100 = CRITICAL
"""

from typing import Dict, Any, List


class RiskEngine:
    """
    Deterministic, weighted Model Risk Score Engine.
    No DB access. Accepts a flat metrics dict and returns
    a structured risk report.
    """

    # ── Weights (must sum to 1.0) ──────────────────────────────────────────────
    WEIGHTS = {
        "accuracy_delta":        0.25,
        "psi":                   0.30,
        "brier_score":           0.15,
        "drifted_features_count": 0.15,
        "calibration":           0.15,
    }

    # ── Thresholds for normalisation ─────────────────────────────────────────
    # Each value: the metric value that represents 100% risk (saturates at 1.0)
    SATURATES_AT = {
        "accuracy_delta":        0.20,   # ≥20 pp drop → full risk
        "psi":                   0.50,   # PSI ≥ 0.5 → full risk
        "brier_score":           0.40,   # Brier ≥ 0.40 → full risk
        "drifted_features_count": 10,    # ≥10 drifted features → full risk
        "calibration":           1.0,    # binary (0 or 1)
    }

    # ── Risk level bands ───────────────────────────────────────────────────────
    @staticmethod
    def _risk_level(score: float) -> str:
        if score <= 30:
            return "LOW"
        elif score <= 60:
            return "MEDIUM"
        elif score <= 80:
            return "HIGH"
        else:
            return "CRITICAL"

    # ── Normalise a single raw metric to [0, 1] risk ──────────────────────────
    @staticmethod
    def _normalise(value: float, saturate_at: float) -> float:
        if saturate_at == 0:
            return 0.0
        return min(1.0, max(0.0, value / saturate_at))

    def calculate_risk_score(self, metrics: dict) -> dict:
        """
        Compute risk score from an audit metrics dict.

        Accepted keys (all optional, defaults to 0):
          accuracy_delta          – train_accuracy − val_accuracy (gap)
          psi                     – max PSI across all features
          brier_score             – calibration Brier score
          drifted_features_count  – number of features with drift_flag=True
          calibration_flag        – bool/int (1 = overconfident)

        Returns:
          {
            "risk_score": int,
            "risk_level": str,
            "component_breakdown": {
              "accuracy_delta": {"raw": ..., "normalised": ..., "weighted_risk": ...},
              ...
            }
          }
        """
        # ── Extract metric values ─────────────────────────────────────────────
        raw = {
            "accuracy_delta":        abs(float(metrics.get("accuracy_delta", 0) or 0)),
            "psi":                   float(metrics.get("psi", 0) or 0),
            "brier_score":           float(metrics.get("brier_score", 0) or 0),
            "drifted_features_count": int(metrics.get("drifted_features_count", 0) or 0),
            "calibration":           float(bool(metrics.get("calibration_flag", False))),
        }

        # ── Normalise + weight ───────────────────────────────────────────────
        breakdown = {}
        total_weighted_risk = 0.0

        for key, weight in self.WEIGHTS.items():
            norm = self._normalise(raw[key], self.SATURATES_AT[key])
            weighted = round(norm * weight, 4)
            total_weighted_risk += weighted
            breakdown[key] = {
                "raw":           raw[key],
                "normalised":    round(norm, 4),
                "weight":        weight,
                "weighted_risk": weighted,
            }

        # ── Final risk score (0–100, higher = riskier) ───────────────────────
        risk_score = round(total_weighted_risk * 100)

        return {
            "risk_score":          risk_score,
            "risk_level":          self._risk_level(risk_score),
            "component_breakdown": breakdown,
        }

    # ── Legacy method kept for backward-compat with orchestrator ─────────────
    @staticmethod
    def calculate_score(test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Original penalty-based scoring used by TestOrchestrator.
        Kept intact — nothing removed.
        """
        categories = {
            "data_quality":         {"penalty": 0.0, "total": 0.0},
            "model_performance":    {"penalty": 0.0, "total": 0.0},
            "robustness":           {"penalty": 0.0, "total": 0.0},
            "bias_fairness":        {"penalty": 0.0, "total": 0.0},
            "statistical_stability": {"penalty": 0.0, "total": 0.0},
        }
        WEIGHTS = {"critical": 1.0, "high": 0.5, "medium": 0.2, "low": 0.05}

        failure_count = 0
        critical_count = 0

        for res in test_results:
            cat = res.get("category", "data_quality")
            if cat not in categories:
                cat = "data_quality"
            severity = res.get("severity", "medium").lower()
            weight = WEIGHTS.get(severity, 0.2)
            categories[cat]["total"] += weight
            if res.get("status") in ["failed", "fail"]:
                categories[cat]["penalty"] += weight
                failure_count += 1
                if severity == "critical":
                    critical_count += 1

        breakdown = {}
        weighted_score_sum = 0.0
        active_cat_count = 0

        for cat, stats in categories.items():
            if stats["total"] > 0:
                cat_score = max(0, 100 * (1 - (stats["penalty"] / stats["total"])))
                breakdown[cat] = round(cat_score, 2)
                weighted_score_sum += cat_score
                active_cat_count += 1
            else:
                breakdown[cat] = 100.0

        final_score = weighted_score_sum / active_cat_count if active_cat_count > 0 else 100.0
        deployment_allowed = critical_count == 0 and final_score >= 80.0

        risk_level = "Low"
        if critical_count > 0:  risk_level = "Critical"
        elif final_score < 60:  risk_level = "High"
        elif final_score < 85:  risk_level = "Medium"

        return {
            "score": round(final_score, 2),
            "risk_level": risk_level,
            "deployment_allowed": deployment_allowed,
            "breakdown": {
                "data_quality": breakdown.get("data_quality", 100),
                "performance":  breakdown.get("model_performance", 100),
                "robustness":   breakdown.get("robustness", 100),
                "bias":         breakdown.get("bias_fairness", 100),
                "drift":        breakdown.get("statistical_stability", 100),
            },
            "metrics": {
                "total_tests":      len(test_results),
                "failures":         failure_count,
                "critical_failures": critical_count,
            },
        }
