"""
governance_engine.py — ML Guard Governance Score Engine

The heart of ML Guard's unique value proposition.
Turns raw audit data into a weighted composite governance score.
Computes live decay from production monitoring signals.
This is what Evidently AI and Arize do NOT do.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# ─── Result Types ─────────────────────────────────────────────────────────────

@dataclass
class PolicyGateResult:
    metric: str
    value: float
    threshold: float
    operator: str     # "lt" | "gt" | "lte" | "gte"
    verdict: str      # PASS | WARN | FAIL
    message: str


@dataclass
class GovernanceScoreResult:
    model_id: str
    overall_score: float
    live_score: float
    verdict: str                             # CERTIFIED | CONDITIONAL | FAILED
    component_scores: Dict[str, float]       # per-module raw scores
    component_weights: Dict[str, float]      # weights applied
    gate_results: List[PolicyGateResult]
    computed_at: datetime
    drift_penalty: float
    perf_penalty: float
    data_freshness_hours: Optional[float]
    recommendations: List[str] = field(default_factory=list)


# ─── Governance Engine ────────────────────────────────────────────────────────

class GovernanceEngine:
    """
    Computes a weighted composite governance score from all available
    audit results for a model. The live score decays automatically
    as production drift and performance degradation are detected.
    """

    WEIGHTS: Dict[str, float] = {
        "performance":  0.25,
        "drift":        0.20,
        "fairness":     0.25,
        "llm_safety":   0.20,
        "robustness":   0.10,
    }

    def compute_score(
        self,
        model_id: str,
        db: Session,
    ) -> GovernanceScoreResult:
        """
        Pull latest result for each module from DB.
        Compute weighted score 0-100.
        Apply live decay from observability signals.
        Return full GovernanceScoreResult.
        """
        from app.db.models import (
            ScanRecord, PerformanceResult, DriftResult, FairnessResult,
            LLMResult, GovernanceResult, DriftReport, PerformanceSnapshot,
        )

        component_scores: Dict[str, float] = {}
        recommendations: List[str] = []

        # ── Pull latest scan record for this model ──────────────────────────
        last_scan: Optional[Any] = None
        try:
            last_scan = (
                db.query(ScanRecord)
                .filter(ScanRecord.model_id == model_id)
                .order_by(ScanRecord.created_at.desc())
                .first()
            )
        except Exception as e:
            logger.warning(f"scan_record_query_failed model_id={model_id} error={str(e)}")

        # ── Per-module scores from latest results ───────────────────────────

        # Performance
        try:
            perf = (
                db.query(PerformanceResult)
                .filter(PerformanceResult.model_id == model_id)
                .order_by(PerformanceResult.created_at.desc())
                .first()
            )
            metrics = getattr(perf, "computed_metrics_json", {}) or {}
            if metrics:
                acc = metrics.get("accuracy", metrics.get("accuracy_score", 0.75))
                component_scores["performance"] = float(acc) * 100
            else:
                base = getattr(last_scan, "governance_score", 0.0) or 70.0
                component_scores["performance"] = float(base) * 0.25 if last_scan else 70.0
                recommendations.append("No recent performance audit — score estimated.")
        except Exception:
            component_scores["performance"] = 70.0

        # Drift
        try:
            drift = (
                db.query(DriftResult)
                .filter(DriftResult.model_id == model_id)
                .order_by(DriftResult.created_at.desc())
                .first()
            )
            metrics = getattr(drift, "computed_metrics_json", {}) or {}
            if metrics:
                psi = metrics.get("psi", metrics.get("overall_psi", 0.0))
                component_scores["drift"] = max(0, 100 - float(psi) * 400)
            else:
                component_scores["drift"] = 80.0
                recommendations.append("No drift audit found.")
        except Exception:
            component_scores["drift"] = 80.0

        # Fairness
        try:
            fairness = (
                db.query(FairnessResult)
                .filter(FairnessResult.model_id == model_id)
                .order_by(FairnessResult.created_at.desc())
                .first()
            )
            if fairness and fairness.computed_metrics_json:
                m = fairness.computed_metrics_json
                # demographic_parity_diff: 0 is best, >0.1 is bad
                dpd = abs(m.get("demographic_parity_diff", m.get("dpd", 0.0)))
                component_scores["fairness"] = max(0, 100 - float(dpd) * 500)
            else:
                component_scores["fairness"] = 75.0
                recommendations.append("No fairness audit found — run a fairness check for compliance.")
        except Exception:
            component_scores["fairness"] = 75.0

        # LLM Safety
        try:
            llm = (
                db.query(LLMResult)
                .filter(LLMResult.model_id == model_id)
                .order_by(LLMResult.created_at.desc())
                .first()
            )
            if llm and llm.computed_metrics_json:
                m = llm.computed_metrics_json
                tox = m.get("toxicity_score", 0.0)
                hall = m.get("hallucination_score", m.get("faithfulness_score", 1.0))
                # Low toxicity + high faithfulness = high score
                component_scores["llm_safety"] = max(0, ((1 - float(tox)) + float(hall)) / 2 * 100)
            else:
                component_scores["llm_safety"] = 85.0
        except Exception:
            component_scores["llm_safety"] = 85.0

        # Robustness (from behavior tests in ScanRecord or GovernanceResult)
        try:
            gov = (
                db.query(GovernanceResult)
                .filter(GovernanceResult.model_id == model_id)
                .order_by(GovernanceResult.created_at.desc())
                .first()
            )
            if gov and gov.computed_metrics_json:
                m = gov.computed_metrics_json
                rob = m.get("robustness_score", m.get("stability_score", 0.75))
                component_scores["robustness"] = float(rob) * 100
            else:
                component_scores["robustness"] = 72.0
        except Exception:
            component_scores["robustness"] = 72.0

        # ── Weighted composite score ────────────────────────────────────────
        overall_score = sum(
            component_scores.get(k, 50.0) * w
            for k, w in self.WEIGHTS.items()
        )
        overall_score = round(min(100.0, max(0.0, overall_score)), 2)

        # ── Live decay from production observability ─────────────────────────
        try:
            last_drift_report = (
                db.query(DriftReport)
                .filter(DriftReport.model_id == model_id)
                .order_by(DriftReport.created_at.desc())
                .first()
            )
            last_perf_snapshot = (
                db.query(PerformanceSnapshot)
                .filter(PerformanceSnapshot.model_id == model_id)
                .order_by(PerformanceSnapshot.computed_at.desc())
                .first()
            )
        except Exception:
            last_drift_report = None
            last_perf_snapshot = None

        live_score, drift_penalty, perf_penalty = self._compute_live_decay(
            overall_score, last_drift_report, last_perf_snapshot
        )

        if drift_penalty > 0.1:
            recommendations.append(f"Production drift detected (penalty {drift_penalty:.1%}). Re-baseline or investigate feature shifts.")
        if perf_penalty > 0.05:
            recommendations.append("Performance degradation detected in production. Verify labels and retrain if needed.")

        verdict = self.get_verdict(live_score)

        # Data freshness
        data_freshness_hours = None
        if last_scan and hasattr(last_scan, "created_at") and last_scan.created_at:
            delta = datetime.utcnow() - last_scan.created_at
            data_freshness_hours = round(delta.total_seconds() / 3600, 1)
            if data_freshness_hours > 168:
                recommendations.append(f"Last audit was {data_freshness_hours:.0f}h ago. Consider re-auditing for fresh compliance.")

        # ── Contract breach penalty ──────────────────────────────────────────
        # Deduct governance points for unresolved behavioral contract breaches
        # in the last 24 hours. Capped at -20 pts total.
        try:
            from app.services.contract_engine import ContractEngine
            _ce = ContractEngine()
            breach_summary = _ce.get_breach_summary(db, model_id, hours=24)
            contract_penalty = breach_summary.get("governance_penalty", 0.0)
            if contract_penalty > 0:
                live_score = max(0.0, live_score - contract_penalty)
                n_breaches = breach_summary["total_breaches"]
                recommendations.append(
                    f"{n_breaches} contract breach(es) detected in last 24h "
                    f"(-{contract_penalty:.1f} pts governance penalty). "
                    f"Review /api/v1/contracts/{model_id}/breach-summary."
                )
        except Exception as _ce_err:
            logger.debug(f"contract_penalty_skipped model_id={model_id} error={_ce_err}")

        # Re-evaluate verdict after contract penalty (score may have dropped)
        verdict = self.get_verdict(live_score)

        return GovernanceScoreResult(
            model_id=model_id,
            overall_score=overall_score,
            live_score=round(live_score, 2),
            verdict=verdict,
            component_scores=component_scores,
            component_weights=self.WEIGHTS,
            gate_results=[],
            computed_at=datetime.utcnow(),
            drift_penalty=drift_penalty,
            perf_penalty=perf_penalty,
            data_freshness_hours=data_freshness_hours,
            recommendations=recommendations,
        )

    def _compute_live_decay(
        self,
        base_score: float,
        latest_drift_report: Optional[Any],
        latest_perf_snapshot: Optional[Any],
    ) -> tuple[float, float, float]:
        """
        Decay base_score based on live production monitoring signals.

        Formula:
            drift_penalty = min(0.30, overall_drift_score)
            perf_penalty  = max(0, baseline_acc - current_acc) × 2
            live_score    = base_score × (1 - drift_penalty) × (1 - perf_penalty)

        This is ML Guard's unique differentiator — nobody else does live score decay.
        """
        drift_penalty = 0.0
        perf_penalty = 0.0

        if latest_drift_report:
            raw = getattr(latest_drift_report, "overall_drift_score", 0.0) or 0.0
            drift_penalty = min(0.30, float(raw))

        if latest_perf_snapshot and latest_perf_snapshot.degradation_report:
            dr = latest_perf_snapshot.degradation_report
            acc_data = dr.get("accuracy", {})
            if acc_data:
                delta = acc_data.get("delta", 0) or 0
                perf_penalty = max(0.0, -float(delta) * 2)

        live_score = base_score * (1 - drift_penalty) * (1 - perf_penalty)
        live_score = round(min(100.0, max(0.0, live_score)), 2)
        return live_score, round(drift_penalty, 4), round(perf_penalty, 4)

    def compute_live_score(
        self,
        base_score: float,
        latest_drift_report: Optional[Any],
        latest_perf_snapshot: Optional[Any],
    ) -> float:
        """Public convenience wrapper for live score only."""
        live_score, _, _ = self._compute_live_decay(
            base_score, latest_drift_report, latest_perf_snapshot
        )
        return live_score

    def get_verdict(self, score: float) -> str:
        """Map numeric score to compliance verdict."""
        if score >= 80:
            return "CERTIFIED"
        if score >= 60:
            return "CONDITIONAL"
        return "FAILED"

    def check_policy_gates(
        self,
        results: Dict[str, Any],
        policy: Any,
    ) -> List[PolicyGateResult]:
        """
        Check each metric against policy thresholds.
        Used by CI/CD gate endpoint and certify flow.

        policy.config is expected as:
        {
            "max_psi": 0.2,
            "min_accuracy": 0.85,
            "bias_parity_threshold": 0.1,
            "max_hallucination_rate": 0.05,
        }
        """
        gates: List[PolicyGateResult] = []
        if not policy:
            return gates

        config: Dict[str, Any] = {}
        if hasattr(policy, "config") and policy.config:
            config = policy.config
        elif hasattr(policy, "rules") and policy.rules:
            config = policy.rules
        elif isinstance(policy, dict):
            config = policy

        threshold_map = {
            "max_psi":                   ("psi",                "lte", "PSI drift"),
            "min_accuracy":              ("accuracy",           "gte", "Accuracy"),
            "min_f1":                    ("f1",                 "gte", "F1 Score"),
            "bias_parity_threshold":     ("dpd",                "lte", "Fairness Parity Diff"),
            "max_hallucination_rate":    ("hallucination_rate", "lte", "Hallucination Rate"),
            "max_toxicity":              ("toxicity_score",     "lte", "Toxicity"),
            "min_roc_auc":               ("roc_auc",            "gte", "ROC-AUC"),
            "min_governance_score":      ("governance_score",   "gte", "Governance Score"),
        }

        for policy_key, (result_key, operator, label) in threshold_map.items():
            threshold = config.get(policy_key)
            if threshold is None:
                continue

            value = results.get(result_key)
            if value is None:
                gates.append(PolicyGateResult(
                    metric=label,
                    value=-1.0,
                    threshold=float(threshold),
                    operator=operator,
                    verdict="WARN",
                    message=f"{label}: No data available — metric not computed.",
                ))
                continue

            value = float(value)
            threshold = float(threshold)

            if operator == "lte":
                passed = value <= threshold
            elif operator == "gte":
                passed = value >= threshold
            elif operator == "lt":
                passed = value < threshold
            elif operator == "gt":
                passed = value > threshold
            else:
                passed = True

            if passed:
                verdict = "PASS"
                msg = f"{label}: {value:.4f} ≤ threshold {threshold:.4f} ✓"
            else:
                verdict = "FAIL"
                msg = f"{label}: {value:.4f} violates threshold {threshold:.4f} ✗"

            gates.append(PolicyGateResult(
                metric=label,
                value=value,
                threshold=threshold,
                operator=operator,
                verdict=verdict,
                message=msg,
            ))

        return gates
