from typing import List, Dict, Any, Optional

def compute_risk_tier(model_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes the risk tier of a model based on its metadata.
    
    Factors (0-10 scale):
    - use_case_sensitivity
    - training_data_sensitivity
    - deployment_environment
    - regulatory_jurisdiction_count
    - governance_score_inverse
    """
    factors = []
    
    # 1. Use Case Sensitivity
    use_case = model_metadata.get("use_case_category", "other")
    use_case_scores = {
        "medical_diagnosis": 10,
        "credit_scoring": 9,
        "hiring": 8,
        "fraud_detection": 7,
        "predictive_maintenance": 5,
        "content_recommendation": 3,
        "other": 2
    }
    use_case_score = use_case_scores.get(use_case, 2)
    factors.append({
        "name": "Use Case Sensitivity",
        "score": use_case_score,
        "weight": 0.4,
        "justification": f"Category '{use_case}' evaluated as {use_case_score}/10 sensitivity."
    })

    # 2. Training Data Sensitivity
    data_sens = model_metadata.get("training_data_sensitivity", "internal")
    data_scores = {
        "restricted": 8,
        "confidential": 6,
        "internal": 3,
        "public": 0
    }
    data_score = data_scores.get(data_sens, 3)
    factors.append({
        "name": "Data Sensitivity",
        "score": data_score,
        "weight": 0.2,
        "justification": f"Data labeled as '{data_sens}' adds {data_score} points to risk profile."
    })

    # 3. Deployment Environment
    env = model_metadata.get("deployment_environment", "development")
    env_scores = {
        "production": 8,
        "staging": 3,
        "development": 1,
        "deprecated": 0
    }
    env_score = env_scores.get(env, 1)
    factors.append({
        "name": "Deployment Environment",
        "score": env_score,
        "weight": 0.15,
        "justification": f"Active in '{env}' environment."
    })

    # 4. Regulatory Jurisdictions
    jurisdictions = model_metadata.get("regulatory_jurisdictions", [])
    reg_score = min(10, len(jurisdictions) * 2)
    factors.append({
        "name": "Regulatory Overhead",
        "score": reg_score,
        "weight": 0.1,
        "justification": f"Subject to {len(jurisdictions)} jurisdictions ({', '.join(jurisdictions)})."
    })

    # 5. Governance Score Inverse
    gov_score = model_metadata.get("governance_score", 100)
    gov_inv_score = (100 - gov_score) / 10
    factors.append({
        "name": "Governance Gap",
        "score": gov_inv_score,
        "weight": 0.15,
        "justification": f"Governance score of {gov_score}% leaves a {gov_inv_score}/10 risk gap."
    })

    # Calculate Weighted Average
    composite_score = sum(f["score"] * f["weight"] for f in factors)
    
    # Map to tier
    if composite_score >= 7:
        tier = "critical"
    elif composite_score >= 5:
        tier = "high"
    elif composite_score >= 3:
        tier = "medium"
    else:
        tier = "low"

    return {
        "tier": tier,
        "composite_score": round(composite_score, 2),
        "factors": factors
    }
