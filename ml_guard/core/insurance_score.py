import datetime
import uuid
import structlog
from typing import Dict, List, Any, Optional
from sqlalchemy.future import select
from sqlalchemy import func
from app.db.models import Model, ScanRecord, RedTeamRun, ContractBreach, AIBOM, SecurityAlert
from ml_guard.core.compliance import evaluate_compliance

logger = structlog.get_logger()

INSURANCE_TIERS = [
    {"name": "platinum", "min": 900},
    {"name": "gold", "min": 800},
    {"name": "standard", "min": 600},
    {"name": "substandard", "min": 400},
    {"name": "uninsurable", "min": 0}
]

PREMIUM_LOOKUP = {
    "low": {
        "platinum": (4000, 12000),
        "gold": (5000, 15000),
        "standard": (7500, 22500),
        "substandard": (12500, 37500),
        "uninsurable": (25000, 75000)
    },
    "high": {
        "platinum": (40000, 120000),
        "gold": (50000, 150000),
        "standard": (75000, 225000),
        "substandard": (125000, 375000),
        "uninsurable": (250000, 750000)
    }
}

async def compute_insurance_score(model_id: uuid.UUID, db) -> Dict[str, Any]:
    """
    Computes an actuarial AI Insurance Score (0-1000) based on 6 key risk dimensions.
    """
    # 0. Fetch Model & Context
    m_result = await db.execute(select(Model).filter(Model.id == model_id))
    model = m_result.scalars().first()
    if not model:
        raise ValueError(f"Model {model_id} not found")

    dim_scores = {}
    risk_factors = []
    
    # --- DIMENSION 1: Model Reliability (200 pts max) ---
    s_result = await db.execute(
        select(ScanRecord.governance_score, ScanRecord.results_json)
        .filter(ScanRecord.model_id == model_id)
        .order_by(ScanRecord.created_at.desc())
        .limit(1)
    )
    latest_scan = s_result.first()
    gov_score = latest_scan[0] if latest_scan else 0
    
    if gov_score >= 80: d1 = 200
    elif gov_score >= 60: d1 = 140
    elif gov_score >= 40: d1 = 80
    else: d1 = 0
    dim_scores["reliability"] = d1
    if d1 < 200:
        risk_factors.append({
            "factor": "Sub-optimal Governance",
            "impact": 200 - d1,
            "recommendation": "Improve drift and fairness indicators to reach 80+ governance score."
        })

    # --- DIMENSION 2: Adversarial Robustness (200 pts max) ---
    r_result = await db.execute(
        select(RedTeamRun.robustness_score)
        .filter(RedTeamRun.model_id == model_id)
        .order_by(RedTeamRun.run_at.desc())
        .limit(1)
    )
    robustness_score = r_result.scalar() or 0
    d2 = int((robustness_score / 100.0) * 200)
    dim_scores["robustness"] = d2
    if d2 < 160:
        risk_factors.append({
            "factor": "Adversarial Vulnerability",
            "impact": 200 - d2,
            "recommendation": "Run exhaustive red teaming and apply prompt injection defenses."
        })

    # --- DIMENSION 3: Deployment Risk (150 pts max) ---
    metadata = model.metadata_json or {}
    risk_tier = metadata.get("risk_tier", "low")
    penalty = 75 if risk_tier == "high" else 0
    
    # Active contract breach rate
    # Simulated total predictions = 10,000 for rate calc
    b_result = await db.execute(
        select(func.count(ContractBreach.id)).filter(ContractBreach.model_id == str(model_id))
    )
    breach_count = b_result.scalar() or 0
    breach_rate = (breach_count / 10000.0) * 100
    
    if breach_rate < 1: d3_base = 150
    elif breach_rate <= 5: d3_base = 100
    else: d3_base = 50
    
    d3 = max(0, d3_base - penalty)
    dim_scores["deployment_risk"] = d3
    if penalty > 0:
        risk_factors.append({
            "factor": "High Stake Decision Environment",
            "impact": penalty,
            "recommendation": "Strict contract enforcement required for autonomous/medical systems."
        })

    # --- DIMENSION 4: Regulatory Compliance (200 pts max) ---
    results_json = latest_scan[1] if latest_scan else {}
    comp_results = evaluate_compliance(results_json)
    passed = sum(1 for r in comp_results if r["status"] == "pass")
    total = len(comp_results)
    compliance_ratio = (passed / total) if total > 0 else 0
    d4 = int(compliance_ratio * 200)
    dim_scores["compliance"] = d4
    if d4 < 180:
        risk_factors.append({
            "factor": "Regulatory Non-Compliance",
            "impact": 200 - d4,
            "recommendation": "Address gaps in AI Act Article mappings."
        })

    # --- DIMENSION 5: Supply Chain Integrity (150 pts max) ---
    a_result = await db.execute(select(AIBOM).filter(AIBOM.model_id == model_id))
    aibom = a_result.scalars().first()
    d5 = 0
    if aibom:
        d5 += 100
        # CVE penalty
        cves = 0
        if aibom.training_datasets:
            for ds in aibom.training_datasets:
                cves += len(ds.get("known_poisoning_cves", []))
        # Add dependency CVEs if stored there
        # The generate_aibom stores cve_alerts at top level usually, but metadata.json is key
        # We check whatever is in results_json
        
        if cves == 0: d5 += 50
        else: d5 = max(0, d5 - (cves * 25))
    dim_scores["supply_chain"] = d5
    if not aibom:
        risk_factors.append({
            "factor": "Unknown Supply Chain Provenance",
            "impact": 150,
            "recommendation": "Generate AIBOM to verify model and data origin."
        })

    # --- DIMENSION 6: Incident History (100 pts max) ---
    ninety_days_ago = datetime.datetime.utcnow() - datetime.timedelta(days=90)
    al_result = await db.execute(
        select(func.count(SecurityAlert.id))
        .filter(
            SecurityAlert.model_id == str(model_id),
            SecurityAlert.severity == "CRITICAL",
            SecurityAlert.created_at >= ninety_days_ago
        )
    )
    critical_alerts = al_result.scalar() or 0
    if critical_alerts == 0: d6 = 100
    elif critical_alerts <= 2: d6 = 60
    else: d6 = 0
    dim_scores["incidents"] = d6
    if critical_alerts > 0:
        risk_factors.append({
            "factor": "Active Security Incidents",
            "impact": 100 - d6,
            "recommendation": f"Resolve {critical_alerts} critical alerts to lower actuarial risk."
        })

    # --- Final Sum & Tier ---
    total_score = sum(dim_scores.values())
    tier = "uninsurable"
    for t in INSURANCE_TIERS:
        if total_score >= t["min"]:
            tier = t["name"]
            break
            
    # --- Premium Calculation ---
    premium_tier_map = PREMIUM_LOOKUP.get(risk_tier, PREMIUM_LOOKUP["low"])
    p_min, p_max = premium_tier_map.get(tier, premium_tier_map["uninsurable"])

    return {
        "model_id": str(model_id),
        "total_score": total_score,
        "tier": tier,
        "dimension_scores": dim_scores,
        "risk_factors": risk_factors,
        "estimated_annual_premium_usd_range": {"min": p_min, "max": p_max},
        "generated_at": datetime.datetime.utcnow().isoformat(),
        "valid_for_days": 90
    }
