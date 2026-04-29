from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime
import uuid

from app.db.session import get_db
from app.db.models import RetrainingPolicy, RetrainingEvent, Model
from app.core.auth import AuthContext, get_auth_context
from core.retraining import evaluate_retrain_trigger, execute_retrain_action

router = APIRouter(prefix="/api/v1/models", tags=["Retraining"])

# --- Schemas ---

class TriggerConditions(BaseModel):
    psi_threshold: float = 0.2
    ks_stat_threshold: float = 0.1
    performance_degradation_pct: float = 15.0
    min_days_since_last_retrain: int = 7
    require_all_conditions: bool = False

class RetrainAction(BaseModel):
    action_type: str = "notify_only" # notify_only, webhook, github_actions, mlflow_run
    webhook_url: Optional[str] = None
    github_repo: Optional[str] = None
    github_workflow_file: Optional[str] = None
    github_token_encrypted: Optional[str] = None

class RetrainingPolicySchema(BaseModel):
    enabled: bool
    trigger_conditions: TriggerConditions
    retrain_action: RetrainAction

class RetrainingEventSchema(BaseModel):
    id: uuid.UUID
    policy_id: uuid.UUID
    model_id: str
    triggered_at: datetime
    triggered_conditions: List[str]
    action_type: str
    action_result: str
    action_error: Optional[str]

# --- Endpoints ---

@router.get("/{model_id}/retraining-policy")
async def get_retraining_policy(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context)
):
    """Fetch the retraining policy for a model."""
    stmt = select(RetrainingPolicy).where(RetrainingPolicy.model_id == model_id)
    res = await db.execute(stmt)
    policy = res.scalars().first()
    
    if not policy:
        # Return default policy if none exists
        return {
            "enabled": False,
            "trigger_conditions": TriggerConditions().dict(),
            "retrain_action": RetrainAction().dict()
        }
    
    return {
        "enabled": policy.enabled,
        "trigger_conditions": policy.trigger_conditions,
        "retrain_action": policy.retrain_action,
        "last_triggered_at": policy.last_triggered_at,
        "trigger_count": policy.trigger_count
    }

@router.post("/{model_id}/retraining-policy")
async def upsert_retraining_policy(
    model_id: str,
    payload: RetrainingPolicySchema,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context)
):
    """Create or update a retraining policy."""
    stmt = select(RetrainingPolicy).where(RetrainingPolicy.model_id == model_id)
    res = await db.execute(stmt)
    policy = res.scalars().first()
    
    if not policy:
        policy = RetrainingPolicy(
            model_id=model_id,
            enabled=payload.enabled,
            trigger_conditions=payload.trigger_conditions.dict(),
            retrain_action=payload.retrain_action.dict()
        )
        db.add(policy)
    else:
        policy.enabled = payload.enabled
        policy.trigger_conditions = payload.trigger_conditions.dict()
        policy.retrain_action = payload.retrain_action.dict()
    
    await db.commit()
    return {"status": "success", "message": "Retraining policy updated."}

@router.post("/{model_id}/retraining-policy/simulate")
async def simulate_retraining_trigger(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context)
):
    """Simulate a retraining trigger evaluation without executing actions."""
    result = await evaluate_retrain_trigger(model_id, db)
    
    # Enrich result for UI
    return {
        "should_trigger": result["should_trigger"],
        "triggered_conditions": result["triggered_conditions"],
        "suppressed": result["suppressed"],
        "suppression_reason": result["suppression_reason"],
        "timestamp": datetime.utcnow().isoformat()
    }

@router.post("/{model_id}/retraining-policy/trigger-now")
async def trigger_retraining_now(
    model_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context)
):
    """Force a retraining trigger evaluation and execution if conditions met (or force if requested)."""
    # For "Trigger Now" we usually bypass suppression or even conditions if forced, 
    # but here we'll follow the policy but ignore suppression for the manual trigger.
    
    # 1. Evaluate
    result = await evaluate_retrain_trigger(model_id, db)
    
    # 2. Check if we should trigger anyway (Manual override)
    # If the user clicks "Trigger Now", they probably want it to happen regardless of conditions
    # but for a "safe" trigger, we'll check if any conditions are met.
    
    policy = result.get("policy")
    if not policy:
        raise HTTPException(status_code=404, detail="Retraining policy not found for this model.")

    if not result["triggered_conditions"]:
        # If no conditions met, we can still force it if it's a manual trigger
        result["triggered_conditions"] = ["Manual trigger by user"]
    
    # Reset suppression for manual trigger
    result["suppressed"] = False
    
    # 3. Execute
    event = await execute_retrain_action(policy, result, db)
    
    return {
        "status": "success",
        "event_id": str(event.id),
        "action_result": event.action_result,
        "triggered_conditions": event.triggered_conditions
    }

@router.get("/{model_id}/retraining-events")
async def list_retraining_events(
    model_id: str,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context)
):
    """Fetch the history of retraining events for a model."""
    stmt = select(RetrainingEvent).where(RetrainingEvent.model_id == model_id).order_by(RetrainingEvent.triggered_at.desc()).limit(limit)
    res = await db.execute(stmt)
    events = res.scalars().all()
    
    return [
        {
            "id": str(e.id),
            "triggered_at": e.triggered_at,
            "triggered_conditions": e.triggered_conditions,
            "action_type": e.action_type,
            "action_result": e.action_result,
            "action_error": e.action_error
        }
        for e in events
    ]
