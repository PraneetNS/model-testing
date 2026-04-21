# Working ML Guard Audit Pipeline State

This document serves as a backup reference of the structural fixes applied to ensure the ML Guard Audit pipeline functions stably with the Next.js frontend UI.

## 1. Synchronous Auditing Fallback (Avoid Infinite Loading)
Due to intermittent issues with local Redis/RabbitMQ or Celery worker crashes, `run_governance_audit_task.delay()` combined with frontend polling can cause an endless "spinning" loader if tasks fail silently.

**Solution applied in `backend/app/routers/audit.py`:**
Forces inline (synchronous fallback) execution by bypassing Celery. 
```python
        # run_governance_audit_task.delay(**encrypted_payload)
        # celery_ok = True
        logger.info("Bypassing Celery: Forcing inline execution for audit task")
        celery_ok = False # Forces the inline calculation method inside the same endpoint
```

## 2. API Response Population for Dashboards
The polling endpoint `backend/app/routers/gate.py` -> `get_gate_result()` was only returning a minimal status object (`{"status": "COMPLETED", "score": ...}`). 
The Next.js frontend, however, deeply relies on a rich JSON response representing all charts.

**Solution applied to `gate.py`:**
```python
        import json
        try:
            results_data = json.loads(scan.results_json) if isinstance(scan.results_json, str) else scan.results_json
        except Exception:
            results_data = {}

        return {
            "status": "COMPLETED",
            "scan_id": str(scan.id),
            "job_id": str(job.id),
            "model_id": str(job.model_id),
            "governance": results_data.get("governance", {"governance_score": scan.governance_score}),
            "risk_score": scan.risk_score,
            "risk_level": scan.risk_level,
            "metrics": results_data.get("metrics", {}),
            "drift": results_data.get("drift", {}),
            "top_drifted_ranked": results_data.get("top_drifted_ranked", []),
            "top5_drifted_features": results_data.get("top5_drifted_features", []),
            "overfitting_gap": results_data.get("overfitting_gap", {}),
            "target_drift": results_data.get("target_drift", {}),
            "calibration": results_data.get("calibration", {}),
            "leakage": results_data.get("leakage", {}),
            "policy": results_data.get("policy", {}),
            "advisories": results_data.get("advisories", []),
            "fingerprint": results_data.get("fingerprint"),
            "complexity": results_data.get("complexity", {}),
            "score": scan.governance_score,
            "verdict": scan.gate_status,
            "breach_count": len(scan.checks_run) if scan.checks_run else 0
        }
```

## 3. RiskEngine Object Compatibility
The API was crashing in inline mode due to a legacy method signature (`RiskEngine().compute()`). 

**Solution applied in `audit.py`:**
Updated the method call and reconstructed the payload explicitly into a format supported by `RiskEngine().calculate_risk_score()`.
```python
        from app.domain.services.risk_engine import RiskEngine
        max_psi = max([v.get("PSI", 0) for v in drift_report.values() if isinstance(v, dict)], default=0.0)
        drifted_count = sum(1 for v in drift_report.values() if isinstance(v, dict) and v.get("drift_flag", False))
        re_metrics = {
             "accuracy_delta": ov_gap.get("accuracy_gap", 0.0),
             "psi": max_psi,
             "brier_score": calibration.get("brier_score", 0.0) if calibration else 0.0,
             "drifted_features_count": drifted_count,
             "calibration_flag": calibration.get("overconfident_flag", False) if calibration else False
        }
        risk_result = RiskEngine().calculate_risk_score(re_metrics)
```
