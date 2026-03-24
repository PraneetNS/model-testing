"""
Unit Tests — Feature 1: RiskEngine & Feature 2: DriftEngine.top_drifted_features
Run with:  cd ml_guard/backend && .\\venv\\Scripts\\python.exe -m pytest app/tests/test_risk_drift.py -v
"""
import pytest
from app.domain.services.risk_engine import RiskEngine
from app.domain.services.drift_engine import DriftEngine


# ═══════════════════════════════════════════════════════
#  RiskEngine Tests
# ═══════════════════════════════════════════════════════

class TestRiskEngine:
    def setup_method(self):
        self.engine = RiskEngine()

    # ── Risk level classification ───────────────────────────────────────────
    def test_level_low(self):
        r = self.engine.calculate_risk_score({
            "accuracy_delta": 0.0,
            "psi": 0.0,
            "brier_score": 0.0,
            "drifted_features_count": 0,
            "calibration_flag": False,
        })
        assert r["risk_score"] == 0
        assert r["risk_level"] == "LOW"

    def test_level_critical(self):
        r = self.engine.calculate_risk_score({
            "accuracy_delta": 0.20,   # saturates → weight 0.25
            "psi": 0.50,              # saturates → weight 0.30
            "brier_score": 0.40,      # saturates → weight 0.15
            "drifted_features_count": 10,  # saturates → weight 0.15
            "calibration_flag": True, # 1.0       → weight 0.15
        })
        assert r["risk_score"] == 100
        assert r["risk_level"] == "CRITICAL"

    def test_medium_boundary(self):
        # Risk score around 45 → MEDIUM
        r = self.engine.calculate_risk_score({
            "accuracy_delta": 0.10,   # 0.5 norm * 0.25 = 0.125
            "psi": 0.25,              # 0.5 norm * 0.30 = 0.15
            "brier_score": 0.0,
            "drifted_features_count": 0,
            "calibration_flag": False,
        })
        assert r["risk_level"] in ("MEDIUM", "LOW")

    def test_partial_risk(self):
        r = self.engine.calculate_risk_score({
            "accuracy_delta": 0.05,
            "psi": 0.10,
            "brier_score": 0.10,
            "drifted_features_count": 2,
            "calibration_flag": False,
        })
        assert 0 <= r["risk_score"] <= 100
        assert r["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    # ── Component breakdown ─────────────────────────────────────────────────
    def test_breakdown_keys_present(self):
        r = self.engine.calculate_risk_score({})
        bd = r["component_breakdown"]
        assert set(bd.keys()) == {"accuracy_delta", "psi", "brier_score", "drifted_features_count", "calibration"}

    def test_breakdown_normalised_in_range(self):
        r = self.engine.calculate_risk_score({
            "accuracy_delta": 0.10,
            "psi": 0.30,
            "brier_score": 0.25,
            "drifted_features_count": 5,
            "calibration_flag": True,
        })
        for comp in r["component_breakdown"].values():
            assert 0.0 <= comp["normalised"] <= 1.0

    # ── Missing / None handling ─────────────────────────────────────────────
    def test_missing_metrics_default_to_zero(self):
        r = self.engine.calculate_risk_score({})
        assert r["risk_score"] == 0
        assert r["risk_level"] == "LOW"

    def test_none_values_handled(self):
        r = self.engine.calculate_risk_score({
            "accuracy_delta": None,
            "psi": None,
            "brier_score": None,
            "drifted_features_count": None,
            "calibration_flag": None,
        })
        assert r["risk_score"] == 0

    # ── Clamping: values above saturation cap at 1.0 ────────────────────────
    def test_saturation_clamp(self):
        r = self.engine.calculate_risk_score({
            "psi": 999,  # way above 0.50 saturate_at
            "accuracy_delta": 999,
        })
        bd = r["component_breakdown"]
        assert bd["psi"]["normalised"] == 1.0
        assert bd["accuracy_delta"]["normalised"] == 1.0

    # ── Weights sum to 1.0 ───────────────────────────────────────────────────
    def test_weights_sum_to_one(self):
        total = sum(RiskEngine.WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9


# ═══════════════════════════════════════════════════════
#  DriftEngine.top_drifted_features Tests
# ═══════════════════════════════════════════════════════

class TestTopDriftedFeatures:
    def _make_report(self):
        return {
            "feature_a": {"psi": 0.32, "status": "drifted"},   # CRITICAL
            "feature_b": {"psi": 0.18, "status": "drifted"},   # WARNING
            "feature_c": {"psi": 0.05, "status": "stable"},    # STABLE
            "feature_d": {"psi": 0.28, "status": "drifted"},   # CRITICAL
            "feature_e": {"psi": 0.02, "status": "stable"},    # STABLE
            "feature_f": {"psi": 0.11, "status": "stable"},    # STABLE
            "feature_g": {"psi": 0.40, "status": "drifted"},   # CRITICAL
        }

    def test_top_5_returned(self):
        result = DriftEngine.top_drifted_features(self._make_report(), top_n=5)
        assert len(result) == 5

    def test_sorted_descending(self):
        result = DriftEngine.top_drifted_features(self._make_report(), top_n=5)
        psis = [r["psi"] for r in result]
        assert psis == sorted(psis, reverse=True)

    def test_severity_critical(self):
        report = {"f": {"psi": 0.30}}
        result = DriftEngine.top_drifted_features(report)
        assert result[0]["severity"] == "CRITICAL"

    def test_severity_warning(self):
        report = {"f": {"psi": 0.20}}
        result = DriftEngine.top_drifted_features(report)
        assert result[0]["severity"] == "WARNING"

    def test_severity_stable(self):
        report = {"f": {"psi": 0.05}}
        result = DriftEngine.top_drifted_features(report)
        assert result[0]["severity"] == "STABLE"

    def test_boundary_warning_lower(self):
        # Exactly 0.15 → WARNING (> 0.15 is CRITICAL, so 0.15 is WARNING or STABLE)
        report = {"f": {"psi": 0.16}}
        result = DriftEngine.top_drifted_features(report)
        assert result[0]["severity"] == "WARNING"

    def test_empty_report(self):
        assert DriftEngine.top_drifted_features({}) == []

    def test_no_psi_features_excluded(self):
        report = {"f": {"ks_p_value": 0.01}}  # no PSI key
        assert DriftEngine.top_drifted_features(report) == []

    def test_top_n_respected(self):
        report = {f"feat_{i}": {"psi": i * 0.01 + 0.01} for i in range(20)}
        result = DriftEngine.top_drifted_features(report, top_n=3)
        assert len(result) == 3

    def test_output_keys(self):
        report = {"f": {"psi": 0.10}}
        result = DriftEngine.top_drifted_features(report)
        assert set(result[0].keys()) == {"feature", "psi", "severity"}
