"""
compliance.py — Regulatory Compliance Mapping Layer
"""

from typing import Dict, List, Optional

REGULATORY_MAPPINGS = [
    {
        "framework": "eu_ai_act",
        "control": "Article 9",
        "title": "Risk Management System",
        "description": "Establish a continuous iterative risk management system.",
        "required_mlguard_checks": ["risk_score"],
        "pass_threshold": 30.0,
        "threshold_type": "max" 
    },
    {
        "framework": "eu_ai_act",
        "control": "Article 10",
        "title": "Data and Data Governance",
        "description": "Training datasets shall be subject to appropriate data governance.",
        "required_mlguard_checks": ["fairness_score", "data_quality_score"],
        "pass_threshold": 50.0, 
        "threshold_type": "min" 
    },
    {
        "framework": "eu_ai_act",
        "control": "Article 13",
        "title": "Transparency and Provision of Information",
        "description": "Ensure operation is sufficiently transparent to enable explainability.",
        "required_mlguard_checks": ["explainability_score"],
        "pass_threshold": 60.0,
        "threshold_type": "min"
    },
    {
        "framework": "eu_ai_act",
        "control": "Article 15",
        "title": "Accuracy, Robustness and Cybersecurity",
        "description": "High-risk AI systems shall achieve an appropriate level of accuracy.",
        "required_mlguard_checks": ["accuracy_score", "security_score"],
        "pass_threshold": 70.0,
        "threshold_type": "min"
    },
    {
        "framework": "eu_ai_act",
        "control": "Article 72",
        "title": "Post-market Monitoring",
        "description": "Document a post-market monitoring system.",
        "required_mlguard_checks": ["drift_score", "telemetry_score"],
        "pass_threshold": 50.0,
        "threshold_type": "min" 
    },
    {
        "framework": "nist_rmf",
        "control": "GOVERN-1.1",
        "title": "Policies and procedures are in place",
        "description": "Policies and procedures are established for AI risk management.",
        "required_mlguard_checks": ["governance_score"],
        "pass_threshold": 80.0,
        "threshold_type": "min"
    },
    {
        "framework": "nist_rmf",
        "control": "MAP-1.5",
        "title": "Lineage and dependencies mapped",
        "description": "Data and model lineage and dependencies are identified.",
        "required_mlguard_checks": ["lineage_score"],
        "pass_threshold": 80.0,
        "threshold_type": "min"
    },
    {
        "framework": "nist_rmf",
        "control": "MEASURE-2.5",
        "title": "Performance Tracking",
        "description": "Performance metrics are identified and tracked.",
        "required_mlguard_checks": ["performance_score"],
        "pass_threshold": 75.0,
        "threshold_type": "min"
    },
    {
        "framework": "nist_rmf",
        "control": "MEASURE-2.6",
        "title": "Fairness and Bias",
        "description": "Fairness and bias metrics are identified and tracked.",
        "required_mlguard_checks": ["fairness_score"],
        "pass_threshold": 80.0,
        "threshold_type": "min"
    },
    {
        "framework": "nist_rmf",
        "control": "MANAGE-2.2",
        "title": "Risk Mitigation",
        "description": "Mechanisms for mitigating risks are implemented.",
        "required_mlguard_checks": ["risk_score"],
        "pass_threshold": 40.0,
        "threshold_type": "max"
    }
]

def evaluate_compliance(governance_report: dict) -> List[Dict]:
    results = []
    for mapping in REGULATORY_MAPPINGS:
        framework = mapping["framework"]
        control = mapping["control"]
        title = mapping["title"]
        description = mapping["description"]
        checks = mapping["required_mlguard_checks"]
        thresh = mapping["pass_threshold"]
        ttype = mapping["threshold_type"]
        
        gaps = []
        evidence_points = []
        is_pass = True
        is_fail = False
        is_partial = False
        
        for check in checks:
            val = governance_report.get(check)
            if val is None:
                gaps.append(f"Missing '{check}' metric")
                is_fail = True
                is_pass = False
                continue
                
            if ttype == "min":
                if val >= thresh:
                    evidence_points.append(f"{check} = {val} (>= {thresh})")
                else:
                    gaps.append(f"{check} = {val} (target >= {thresh})")
                    is_pass = False
                    # strict fail if fairness < 50 for Article 10, or generally if below threshold
                    # Let's say if it fails any threshold, it's a fail.
                    is_fail = True
            else: # max threshold
                if val <= thresh:
                    evidence_points.append(f"{check} = {val} (<= {thresh})")
                else:
                    gaps.append(f"{check} = {val} (target <= {thresh})")
                    is_pass = False
                    is_fail = True
                    
        # Let's add partial logic if we miss one of multiple checks, but pass another
        if is_fail and len(evidence_points) > 0:
            status = "partial" # some passed, some failed -> partial
            # Wait, the EU AI Act Article 10 with fairness < 50 test requires it to fail!
            # If it passes data_quality_score but fails fairness_score, it should fail Article 10.
            # So if ANY check fails, it fails.
            status = "fail"
        elif is_fail:
            status = "fail"
        else:
            status = "pass"
            
        results.append({
            "framework": framework,
            "control": control,
            "title": title,
            "description": description,
            "status": status,
            "evidence": "; ".join(evidence_points) if evidence_points else "None",
            "gap": "; ".join(gaps) if gaps else None
        })
    
    return results
