"""
Rule-Based Advisory Engine.
Evaluates governance signals and produces actionable, deterministic advice.
No LLM. No interpretation.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Advisory:
    code: str
    severity: str  # CRITICAL | WARNING | INFO
    message: str
    recommendation: str


def generate_advisories(
    drift_report: dict = None,
    overfitting_gap: dict = None,
    calibration: dict = None,
    leakage: dict = None,
    robustness_score: float = None,
    governance_score: float = None,
) -> List[dict]:
    advisories = []

    # --- DRIFT ADVISORIES ---
    if drift_report:
        drifted_features = [f for f, m in drift_report.items() if isinstance(m, dict) and m.get("drift_flag")]
        drift_ratio = len(drifted_features) / max(len(drift_report), 1)
        if drift_ratio > 0.3:
            advisories.append(Advisory(
                code="DRIFT_WIDESPREAD",
                severity="CRITICAL",
                message=f"{len(drifted_features)} of {len(drift_report)} features showing significant drift (>{drift_ratio*100:.0f}%).",
                recommendation="Retrain model using a recent data window aligned to current distribution."
            ))
        elif drift_ratio > 0.0:
            advisories.append(Advisory(
                code="DRIFT_PARTIAL",
                severity="WARNING",
                message=f"Drift detected on {len(drifted_features)} feature(s): {', '.join(drifted_features[:5])}.",
                recommendation="Monitor affected features closely. Consider targeted feature re-engineering."
            ))

    # --- OVERFITTING ADVISORIES ---
    if overfitting_gap:
        for metric, gap in overfitting_gap.items():
            if gap > 0.10:
                advisories.append(Advisory(
                    code="OVERFIT_HIGH",
                    severity="CRITICAL",
                    message=f"Overfitting gap on {metric}: {gap:.4f} (>10% threshold).",
                    recommendation="Increase regularization (C, alpha, or max_depth). Consider pruning or dropout. Cross-validate with more folds."
                ))
            elif gap > 0.05:
                advisories.append(Advisory(
                    code="OVERFIT_MODERATE",
                    severity="WARNING",
                    message=f"Mild overfitting on {metric}: gap = {gap:.4f}.",
                    recommendation="Review training data size and apply early stopping or regularization."
                ))

    # --- CALIBRATION ADVISORIES ---
    if calibration:
        brier = calibration.get("brier_score", 0)
        ece = calibration.get("ece", 0)
        overconfident = calibration.get("overconfident_flag", False)
        if overconfident:
            advisories.append(Advisory(
                code="CALIBRATION_OVERCONFIDENT",
                severity="CRITICAL",
                message=f"Model is overconfident. Brier={brier:.4f}, ECE={ece:.4f}.",
                recommendation="Apply Platt scaling or isotonic regression post-hoc calibration. Do not deploy with raw probabilities."
            ))
        elif brier > 0.20:
            advisories.append(Advisory(
                code="CALIBRATION_POOR",
                severity="WARNING",
                message=f"High Brier Score: {brier:.4f}. Probability estimates are unreliable.",
                recommendation="Recalibrate using CalibratedClassifierCV or temperature scaling."
            ))

    # --- LEAKAGE ADVISORIES ---
    if leakage:
        risk = leakage.get("risk_level", "NONE")
        suspects = leakage.get("leakage_suspects", {})
        if risk == "HIGH":
            advisories.append(Advisory(
                code="LEAKAGE_HIGH",
                severity="CRITICAL",
                message=f"Potential target leakage detected in: {', '.join(suspects.keys())}.",
                recommendation="Audit feature engineering pipeline. Verify temporal isolation between train/test. Remove or investigate suspected columns."
            ))
        elif risk == "MODERATE":
            advisories.append(Advisory(
                code="LEAKAGE_MODERATE",
                severity="WARNING",
                message=f"Moderate leakage risk on: {', '.join(suspects.keys())}.",
                recommendation="Cross-check feature availability at inference time. Ensure no future data is used for training."
            ))

    # --- ROBUSTNESS ADVISORIES ---
    if robustness_score is not None:
        if robustness_score < 60:
            advisories.append(Advisory(
                code="ROBUSTNESS_FRAGILE",
                severity="CRITICAL",
                message=f"Robustness score critically low: {robustness_score}/100.",
                recommendation="Model is sensitive to input perturbations. Add input validation, hard constraints, and fallback rules at inference."
            ))
        elif robustness_score < 80:
            advisories.append(Advisory(
                code="ROBUSTNESS_MODERATE",
                severity="WARNING",
                message=f"Robustness score below optimal: {robustness_score}/100.",
                recommendation="Investigate high-sensitivity features. Consider ensemble methods to reduce variance."
            ))

    # --- FINAL GOVERNANCE ---
    if governance_score is not None:
        if governance_score < 50:
            advisories.append(Advisory(
                code="GOVERNANCE_CRITICAL",
                severity="CRITICAL",
                message=f"Governance score critically low: {governance_score:.1f}/100. Deployment BLOCKED.",
                recommendation="Do not deploy. Resolve critical violations before re-evaluation."
            ))
        elif governance_score < 70:
            advisories.append(Advisory(
                code="GOVERNANCE_AT_RISK",
                severity="WARNING",
                message=f"Governance score below deployment threshold: {governance_score:.1f}/100.",
                recommendation="Address warning-level issues before deploying to production."
            ))

    return [{"code": a.code, "severity": a.severity, "message": a.message, "recommendation": a.recommendation} for a in advisories]
