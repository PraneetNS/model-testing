import sys
import os
import pytest

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if _repo_root not in sys.path:
    sys.path.append(_repo_root)

from ml_guard.core.compliance import evaluate_compliance

def test_fully_certified_passes_nist_govern():
    good_report = {
        "risk_score": 20.0,
        "fairness_score": 90.0,
        "data_quality_score": 90.0,
        "explainability_score": 85.0,
        "accuracy_score": 95.0,
        "security_score": 90.0,
        "drift_score": 70.0,
        "telemetry_score": 80.0,
        "governance_score": 95.0,
        "lineage_score": 85.0,
        "performance_score": 90.0
    }
    
    results = evaluate_compliance(good_report)
    
    count_nist = 0
    for res in results:
        if res["framework"] == "nist_rmf":
            assert res["status"] == "pass", f"{res['control']} failed. {res}"
            count_nist += 1
            
    assert count_nist > 0

def test_fairness_below_50_fails_eu_ai_act_article_10():
    bad_report = {
        "fairness_score": 45.0,
        "data_quality_score": 90.0, 
    }
    
    results = evaluate_compliance(bad_report)
    
    article_10 = next((r for r in results if r["control"] == "Article 10"), None)
    assert article_10 is not None
    assert article_10["status"] == "fail", f"Expected fail but got: {article_10['status']}"
