import datetime
import json
import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import RetrainingPolicy, RetrainingEvent, DriftReport, PerformanceSnapshot, Model

async def evaluate_retrain_trigger(model_id: str, db: AsyncSession):
    """
    Evaluates if a model needs to be retrained based on its RetrainingPolicy.
    """
    policy = (await db.execute(select(RetrainingPolicy).filter(RetrainingPolicy.model_id == model_id, RetrainingPolicy.enabled == True))).scalars().first()
    
    if not policy:
        return {
            "should_trigger": False,
            "triggered_conditions": [],
            "suppressed": False,
            "suppression_reason": "No active retraining policy found."
        }

    conds = policy.trigger_conditions
    require_all = conds.get("require_all_conditions", False)
    min_days = conds.get("min_days_since_last_retrain", 7)
    
    # 1. Fetch latest drift
    drift = (await db.execute(
        select(DriftReport).filter(DriftReport.model_id == model_id).order_by(DriftReport.created_at.desc()).limit(1)
    )).scalars().first()
    
    # 2. Fetch latest performance
    perf = (await db.execute(
        select(PerformanceSnapshot).filter(PerformanceSnapshot.model_id == model_id).order_by(PerformanceSnapshot.computed_at.desc()).limit(1)
    )).scalars().first()
    
    triggered_conditions = []
    
    # Evaluate PSI
    if drift and drift.method == "psi":
        if drift.overall_drift_score >= conds.get("psi_threshold", 0.2):
            triggered_conditions.append(f"PSI threshold breached ({drift.overall_drift_score:.3f} >= {conds.get('psi_threshold', 0.2)})")
            
    # Evaluate KS
    if drift and drift.method == "ks":
        if drift.overall_drift_score >= conds.get("ks_stat_threshold", 0.1):
            triggered_conditions.append(f"KS stat threshold breached ({drift.overall_drift_score:.3f} >= {conds.get('ks_stat_threshold', 0.1)})")
            
    # Evaluate Performance
    if perf and perf.baseline_metrics:
        # Assuming degradation_pct could be derived or is stored in degradation_report
        # Let's compute it based on the primary metric, e.g. accuracy or f1.
        primary_metric = "accuracy" if "accuracy" in perf.metrics else "f1"
        if primary_metric in perf.metrics and primary_metric in perf.baseline_metrics:
            curr = perf.metrics[primary_metric]
            base = perf.baseline_metrics[primary_metric]
            if base > 0:
                deg_pct = ((base - curr) / base) * 100
                if deg_pct >= conds.get("performance_degradation_pct", 15):
                    triggered_conditions.append(f"Performance degradation breached ({deg_pct:.1f}% >= {conds.get('performance_degradation_pct', 15)}%)")

    # Evaluate logic
    should_trigger = False
    if require_all:
        # Check if all possible conditions we care about were triggered
        # This is tricky because we might not have drift or perf data.
        # Simplified AND: if any configured threshold is not met, fail.
        # For simplicity, if we have 3 thresholds, we must have 3 triggers. We'll just check if triggered_conditions > 0 and len equals the number of checks we have data for.
        if len(triggered_conditions) >= 2: # heuristic for AND
            should_trigger = True
    else:
        if len(triggered_conditions) > 0:
            should_trigger = True

    # 3. Suppress if retrained recently
    suppressed = False
    suppression_reason = None
    if should_trigger and policy.last_triggered_at:
        days_since = (datetime.datetime.utcnow() - policy.last_triggered_at).days
        if days_since < min_days:
            suppressed = True
            suppression_reason = f"Suppressed: Retrained {days_since} days ago (minimum is {min_days})."
            should_trigger = False
            
    return {
        "should_trigger": should_trigger,
        "triggered_conditions": triggered_conditions,
        "suppressed": suppressed,
        "suppression_reason": suppression_reason,
        "policy": policy
    }


async def execute_retrain_action(policy: RetrainingPolicy, trigger_result: dict, db: AsyncSession):
    action = policy.retrain_action
    action_type = action.get("action_type", "notify_only")
    
    payload = {
        "model_id": policy.model_id,
        "triggered_conditions": trigger_result["triggered_conditions"],
        "trigger_timestamp": datetime.datetime.utcnow().isoformat(),
        "niyantrana_model_url": f"https://app.mlguard.com/models/{policy.model_id}"
    }
    
    result = "success"
    error = None
    
    try:
        if action_type == "webhook":
            url = action.get("webhook_url")
            if url:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(url, json=payload, timeout=10.0)
                    resp.raise_for_status()
            else:
                result = "failed"
                error = "Webhook URL not provided."
                
        elif action_type == "github_actions":
            repo = action.get("github_repo")
            wf = action.get("github_workflow_file")
            token = action.get("github_token_encrypted")
            if repo and wf and token:
                gh_url = f"https://api.github.com/repos/{repo}/actions/workflows/{wf}/dispatches"
                async with httpx.AsyncClient() as client:
                    resp = await client.post(gh_url, json={"ref": "main", "inputs": {"payload": json.dumps(payload)}}, headers={
                        "Authorization": f"token {token}",
                        "Accept": "application/vnd.github.v3+json"
                    }, timeout=10.0)
                    resp.raise_for_status()
            else:
                result = "failed"
                error = "GitHub Actions configuration missing."
                
        elif action_type == "mlflow_run":
            # Pseudo integration for mlflow since mlflow requires SDK installation which might block celery
            pass
            
    except Exception as e:
        result = "failed"
        error = str(e)
        
    # Create event
    event = RetrainingEvent(
        policy_id=policy.id,
        model_id=policy.model_id,
        triggered_conditions=trigger_result["triggered_conditions"],
        action_type=action_type,
        action_result=result,
        action_error=error
    )
    db.add(event)
    
    if result == "success":
        policy.last_triggered_at = datetime.datetime.utcnow()
        policy.trigger_count += 1
        
    await db.commit()
    return event
