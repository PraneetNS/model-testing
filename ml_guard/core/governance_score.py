"""
Mathematically transparent governance scoring.
All subscores normalized to [0, 1], then combined with configurable weights.

Governance = w_d·Sd + w_o·So + w_c·Sc + w_r·Sr + w_f·Sf

Where:
  Sd       = drift score         = exp(-max_PSI)
  So       = overfitting score   = 1 - max_gap
  Sc       = calibration score   = 1 - brier_score
  Srobust  = robustness score    = stability_score (Monte Carlo)
  Sf       = fairness score      = from fairness engine subscore
"""
import numpy as np
import hashlib
import joblib
import os
import tempfile
from typing import Dict, List, Optional


def compute_drift_subscore(drift_report: dict) -> float:
    """Sd = exp(-max_PSI). Max PSI across features."""
    if not drift_report:
        return 1.0
    psi_values = [v.get("PSI", 0) for v in drift_report.values() if isinstance(v, dict)]
    max_psi = max(psi_values) if psi_values else 0.0
    return float(np.exp(-max_psi))


def compute_overfitting_subscore(overfitting_gap: dict) -> float:
    """So = 1 - max_gap, clipped to [0, 1]."""
    if not overfitting_gap:
        return 1.0
    max_gap = max(abs(float(v)) for v in overfitting_gap.values())
    return float(np.clip(1.0 - max_gap, 0.0, 1.0))


def compute_calibration_subscore(brier_score: Optional[float]) -> float:
    """Sc = 1 - brier_score, clipped to [0, 1]."""
    if brier_score is None:
        return 1.0
    return float(np.clip(1.0 - brier_score, 0.0, 1.0))


def compute_robustness_subscore(stability_score: Optional[float]) -> float:
    """Sr = stability_score (already in [0, 1] from Monte Carlo)."""
    if stability_score is None:
        return 1.0
    return float(np.clip(stability_score, 0.0, 1.0))


def compute_fairness_subscore(fairness_subscore: Optional[float] = None) -> float:
    """Sf = fairness_subscore (already in [0, 1] from fairness engine)."""
    if fairness_subscore is None:
        return 1.0
    return float(np.clip(fairness_subscore, 0.0, 1.0))


def compute_agent_behavior_subscore(agent_violations: Optional[List[Dict]] = None) -> float:
    """
    Sa = Agent Behavioral Risk subscore.
    Score = 100 - (CRITICAL_violations * 20) - (HIGH_violations * 8) - (LOW_violations * 2), floored at 0.
    Normalized to [0, 1].
    """
    if agent_violations is None:
        return 1.0
    
    penalty_sum = 0
    for v in agent_violations:
        sev = v.get("severity", "LOW") if isinstance(v, dict) else getattr(v, "severity", "LOW")
        if sev == "CRITICAL": penalty_sum += 20
        elif sev == "HIGH": penalty_sum += 8
        elif sev == "LOW": penalty_sum += 2
    
    raw_score = max(0, 100 - penalty_sum)
    return float(raw_score / 100.0)


def compute_governance_score(
    drift_report: dict = None,
    overfitting_gap: dict = None,
    brier_score: float = None,
    stability_score: float = None,
    fairness_subscore: float = None,
    agent_violations: list = None,
    compliance_score: float = None,
    weights: dict = None,
) -> dict:
    """
    Compute the normalized composite Governance Score.
    Returns score [0,100], per-component breakdown, and deployment decision.

    Weights default:
      drift: 0.20, overfitting: 0.20, calibration: 0.20,
      robustness: 0.20, fairness: 0.15  (remaining 0.05 distributed)

    If fairness data is not provided, the old 4-weight system is used
    for full backward compatibility.
    """
    if weights is None:
        if agent_violations is not None:
            # 6-weight system including Agent Behavioral Risk (15%)
            weights = {
                "drift": 0.17, "overfitting": 0.17, "calibration": 0.17,
                "robustness": 0.17, "fairness": 0.17, "agent": 0.15
            }
            # Adjust to exactly 1.0
            diff = 1.0 - sum(weights.values())
            weights["drift"] += diff
        elif fairness_subscore is not None:
            weights = {
                "drift": 0.20, "overfitting": 0.20, "calibration": 0.20,
                "robustness": 0.20, "fairness": 0.15,
            }
            # Distribute remaining 5% equally
            remainder = 1.0 - sum(weights.values())
            for k in weights:
                weights[k] += remainder / len(weights)
        else:
            weights = {"drift": 0.25, "overfitting": 0.25, "calibration": 0.25, "robustness": 0.25}

    Sd = compute_drift_subscore(drift_report or {})
    So = compute_overfitting_subscore(overfitting_gap or {})
    Sc = compute_calibration_subscore(brier_score)
    Sr = compute_robustness_subscore(stability_score)
    Sf = compute_fairness_subscore(fairness_subscore)
    Sa = compute_agent_behavior_subscore(agent_violations)

    gov_score = (
        weights.get("drift", 0)       * Sd +
        weights.get("overfitting", 0) * So +
        weights.get("calibration", 0) * Sc +
        weights.get("robustness", 0)  * Sr +
        weights.get("fairness", 0)    * Sf +
        weights.get("agent", 0)       * Sa
    ) * 100.0

    gov_score = float(np.clip(gov_score, 0, 100))
    deployment_allowed = gov_score >= 70.0

    component_scores = {
        "drift_score":       round(Sd * 100, 2),
        "overfitting_score": round(So * 100, 2),
        "calibration_score": round(Sc * 100, 2),
        "robustness_score":  round(Sr * 100, 2),
    }
    if fairness_subscore is not None:
        component_scores["fairness_score"] = round(Sf * 100, 2)
    if agent_violations is not None:
        component_scores["agent_score"] = round(Sa * 100, 2)
    if compliance_score is not None:
        component_scores["compliance_score"] = round(compliance_score, 2)

    return {
        "governance_score": round(gov_score, 2),
        "deployment_allowed": deployment_allowed,
        "component_scores": component_scores,
        "weights": weights,
    }


def compute_model_fingerprint(model_input) -> str:
    """SHA-256 hash of the serialized model. Supports bytes or file-like objects."""
    sha = hashlib.sha256()
    if isinstance(model_input, bytes):
        sha.update(model_input)
    else:
        # Assume file-like object
        while chunk := model_input.read(8192):
            sha.update(chunk)
        if hasattr(model_input, "seek"):
            model_input.seek(0)
    return sha.hexdigest()


def compute_model_complexity(model) -> dict:
    """
    Estimate model complexity from introspectable attributes.
    - Tree-based: n_estimators * max_depth
    - Linear: number of coefficients
    - Other: approximate parameter count
    """
    complexity = {"type": type(model).__name__, "proxy_score": None, "notes": []}

    if hasattr(model, "n_estimators") and hasattr(model, "estimators_"):
        depths = [est.get_depth() for est in model.estimators_ if hasattr(est, "get_depth")]
        avg_depth = float(np.mean(depths)) if depths else 0
        complexity["n_estimators"] = model.n_estimators
        complexity["avg_tree_depth"] = round(avg_depth, 2)
        complexity["proxy_score"] = int(model.n_estimators * avg_depth)
        complexity["notes"].append("Tree ensemble: complexity ∝ n_estimators × avg_depth")

    elif hasattr(model, "coef_"):
        n_coefs = int(np.prod(model.coef_.shape))
        complexity["n_coefficients"] = n_coefs
        complexity["proxy_score"] = n_coefs
        complexity["notes"].append("Linear model: complexity ∝ number of coefficients")

    elif hasattr(model, "n_support_"):
        n_sv = int(np.sum(model.n_support_))
        complexity["n_support_vectors"] = n_sv
        complexity["proxy_score"] = n_sv
        complexity["notes"].append("SVM: complexity ∝ number of support vectors")

    else:
        complexity["proxy_score"] = -1
        complexity["notes"].append("Unable to estimate complexity from model type.")

    return complexity
