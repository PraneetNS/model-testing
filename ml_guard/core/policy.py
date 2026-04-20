"""
Governance Policy Engine.
Evaluates governance signals against a configurable policy and emits:
- PASSED (all checks green)
- WARNING (some checks near threshold)
- CRITICAL (one or more checks exceeded; deployment blocked)

CI/CD compatible — returns a machine-readable status with per-check breakdown.
"""
from dataclasses import dataclass
from typing import Optional, List


DEFAULT_POLICY = {
    "max_psi":              0.20,
    "max_jsd":              0.10,
    "min_accuracy":         0.80,
    "min_r2":               0.80,
    "max_overfit_gap":      0.08,
    "max_brier_score":      0.20,
    "min_stability_score":  0.90,
    "min_governance_score": 70.0,
    # Fairness thresholds
    "max_spd":              0.10,
    "min_dir":              0.80,
    "max_eod":              0.10,
    # LLM thresholds
    "max_llm_risk":         60.0,
    "max_toxicity":         0.30,
    "max_hallucination":    0.50,
}


@dataclass
class PolicyCheckResult:
    name: str
    policy_value: float
    actual_value: Optional[float]
    status: str          # PASSED | WARNING | CRITICAL
    message: str


def evaluate_policy(
    metrics: dict = None,
    drift_report: dict = None,
    overfitting_gap: dict = None,
    calibration: dict = None,
    stability_score: float = None,
    governance_score: float = None,
    fairness: dict = None,
    llm_evaluation: dict = None,
    security: dict = None,
    policy: dict = None,
) -> dict:
    """
    Evaluate all signals against a policy config.
    Returns structured per-check results and overall gate status.

    Supports classical ML checks, fairness checks, and LLM checks.
    """
    pol = {**DEFAULT_POLICY, **(policy or {})}
    checks: List[PolicyCheckResult] = []

    # --- Accuracy ---
    if metrics and "accuracy" in metrics:
        acc = float(metrics["accuracy"])
        threshold = pol["min_accuracy"]
        status = "PASSED" if acc >= threshold else "CRITICAL"
        checks.append(PolicyCheckResult(
            name="Accuracy",
            policy_value=threshold,
            actual_value=acc,
            status=status,
            message=f"Accuracy {acc:.4f} {'≥' if status == 'PASSED' else '<'} policy min {threshold:.2f}",
        ))

    # --- R2 (Regression) ---
    if metrics and "r2_score" in metrics:
        r2 = float(metrics["r2_score"])
        threshold = pol.get("min_r2", 0.80)
        status = "PASSED" if r2 >= threshold else "CRITICAL"
        checks.append(PolicyCheckResult(
            name="R2 Score",
            policy_value=threshold,
            actual_value=r2,
            status=status,
            message=f"R2 Score {r2:.4f} {'≥' if status == 'PASSED' else '<'} policy min {threshold:.2f}",
        ))

    # --- PSI Drift ---
    if drift_report:
        psi_vals = [v.get("PSI", 0) for v in drift_report.values() if isinstance(v, dict)]
        max_psi = max(psi_vals) if psi_vals else 0.0
        threshold = pol["max_psi"]
        status = "PASSED" if max_psi <= threshold else "CRITICAL"
        checks.append(PolicyCheckResult(
            name="Max PSI Drift",
            policy_value=threshold,
            actual_value=max_psi,
            status=status,
            message=f"Max PSI {max_psi:.4f} {'≤' if status == 'PASSED' else '>'} policy max {threshold:.2f}",
        ))

    # --- JSD Drift ---
    if drift_report:
        jsd_vals = [v.get("JSD", 0) for v in drift_report.values() if isinstance(v, dict)]
        max_jsd = max(jsd_vals) if jsd_vals else 0.0
        threshold = pol["max_jsd"]
        status = "PASSED" if max_jsd <= threshold else ("WARNING" if max_jsd <= threshold * 1.5 else "CRITICAL")
        checks.append(PolicyCheckResult(
            name="Max JSD Drift",
            policy_value=threshold,
            actual_value=max_jsd,
            status=status,
            message=f"Max JSD {max_jsd:.4f} vs policy max {threshold}",
        ))

    # --- Overfitting ---
    if overfitting_gap:
        max_gap = max(abs(float(v)) for v in overfitting_gap.values())
        threshold = pol["max_overfit_gap"]
        status = "PASSED" if max_gap <= threshold else ("WARNING" if max_gap <= threshold * 1.5 else "CRITICAL")
        checks.append(PolicyCheckResult(
            name="Overfitting Gap",
            policy_value=threshold,
            actual_value=max_gap,
            status=status,
            message=f"Max overfitting gap {max_gap:.4f} vs policy max {threshold}",
        ))

    # --- Brier Score ---
    if calibration:
        brier = float(calibration.get("brier_score", 0))
        threshold = pol["max_brier_score"]
        status = "PASSED" if brier <= threshold else "WARNING"
        checks.append(PolicyCheckResult(
            name="Brier Score (Calibration)",
            policy_value=threshold,
            actual_value=brier,
            status=status,
            message=f"Brier {brier:.4f} vs policy max {threshold}",
        ))

    # --- Stability ---
    if stability_score is not None:
        threshold = pol["min_stability_score"]
        status = "PASSED" if stability_score >= threshold else "CRITICAL"
        checks.append(PolicyCheckResult(
            name="Monte Carlo Stability",
            policy_value=threshold,
            actual_value=stability_score,
            status=status,
            message=f"Stability {stability_score:.4f} vs policy min {threshold}",
        ))

    # --- Governance Score ---
    if governance_score is not None:
        threshold = pol["min_governance_score"]
        status = "PASSED" if governance_score >= threshold else "CRITICAL"
        checks.append(PolicyCheckResult(
            name="Composite Governance Score",
            policy_value=threshold,
            actual_value=governance_score,
            status=status,
            message=f"Governance {governance_score:.2f} vs policy min {threshold}",
        ))

    # --- Fairness: SPD ---
    if fairness and "statistical_parity_diff" in fairness:
        spd = abs(float(fairness["statistical_parity_diff"]))
        threshold = pol.get("max_spd", 0.1)
        status = "PASSED" if spd <= threshold else ("WARNING" if spd <= threshold * 1.5 else "CRITICAL")
        checks.append(PolicyCheckResult(
            name="Statistical Parity Difference",
            policy_value=threshold,
            actual_value=spd,
            status=status,
            message=f"|SPD| {spd:.4f} vs policy max {threshold}",
        ))

    # --- Fairness: DIR ---
    if fairness and "disparate_impact_ratio" in fairness:
        dir_val = float(fairness["disparate_impact_ratio"])
        threshold = pol.get("min_dir", 0.8)
        status = "PASSED" if dir_val >= threshold else "CRITICAL"
        checks.append(PolicyCheckResult(
            name="Disparate Impact Ratio",
            policy_value=threshold,
            actual_value=dir_val,
            status=status,
            message=f"DIR {dir_val:.4f} {'≥' if status == 'PASSED' else '<'} policy min {threshold}",
        ))

    # --- Fairness: EOD ---
    if fairness and "equal_opportunity_diff" in fairness:
        eod = abs(float(fairness["equal_opportunity_diff"]))
        threshold = pol.get("max_eod", 0.1)
        status = "PASSED" if eod <= threshold else ("WARNING" if eod <= threshold * 1.5 else "CRITICAL")
        checks.append(PolicyCheckResult(
            name="Equal Opportunity Difference",
            policy_value=threshold,
            actual_value=eod,
            status=status,
            message=f"|EOD| {eod:.4f} vs policy max {threshold}",
        ))

    # --- LLM Risk Score ---
    if llm_evaluation and "llm_risk_score" in llm_evaluation:
        risk = float(llm_evaluation["llm_risk_score"])
        threshold = pol.get("max_llm_risk", 60.0)
        status = "PASSED" if risk <= threshold else "CRITICAL"
        checks.append(PolicyCheckResult(
            name="LLM Risk Score",
            policy_value=threshold,
            actual_value=risk,
            status=status,
            message=f"LLM risk {risk:.2f} vs policy max {threshold}",
        ))

    # --- LLM Toxicity ---
    if llm_evaluation and "toxicity_score" in llm_evaluation:
        tox = float(llm_evaluation["toxicity_score"])
        threshold = pol.get("max_toxicity", 0.30)
        status = "PASSED" if tox <= threshold else ("WARNING" if tox <= threshold * 1.5 else "CRITICAL")
        checks.append(PolicyCheckResult(
            name="LLM Toxicity",
            policy_value=threshold,
            actual_value=tox,
            status=status,
            message=f"Toxicity {tox:.4f} vs policy max {threshold}",
        ))

    # --- LLM Hallucination ---
    if llm_evaluation and "hallucination_risk" in llm_evaluation:
        hall = float(llm_evaluation["hallucination_risk"])
        threshold = pol.get("max_hallucination", 0.50)
        status = "PASSED" if hall <= threshold else "WARNING"
        checks.append(PolicyCheckResult(
            name="LLM Hallucination Risk",
            policy_value=threshold,
            actual_value=hall,
            status=status,
            message=f"Hallucination {hall:.4f} vs policy max {threshold}",
        ))

    # --- Security Risk ---
    if security:
        risk = security.get("overall_risk", "LOW")
        status = "PASSED" if risk in ["LOW", "MEDIUM"] else "CRITICAL"
        checks.append(PolicyCheckResult(
            name="Model Security Risk",
            policy_value=0.0, # N/A
            actual_value=None,
            status=status,
            message=f"Model security risk level is {risk}",
        ))

    # --- Overall Gate ---
    all_statuses = [c.status for c in checks]
    if "CRITICAL" in all_statuses:
        gate = "CRITICAL"
    elif "WARNING" in all_statuses:
        gate = "WARNING"
    else:
        gate = "PASSED"

    return {
        "gate_status": gate,
        "deployment_allowed": gate == "PASSED",
        "policy_used": pol,
        "checks": [
            {
                "name": c.name,
                "status": c.status,
                "actual_value": c.actual_value,
                "policy_value": c.policy_value,
                "message": c.message,
            }
            for c in checks
        ],
    }

