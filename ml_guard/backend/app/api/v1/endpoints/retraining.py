from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Dict, Any

from app.api.v1 import deps
from app.db.models import RetrainingPolicy, RetrainingEvent
from ml_guard.core.retraining import evaluate_retrain_trigger, execute_retrain_action

router = APIRouter(tags=["retraining"])

@router.post("/{model_id}/retraining-policy")
async def create_or_update_policy(model_id: str, payload: dict, db: AsyncSession = Depends(deps.get_db)):
    policy = (await db.execute(select(RetrainingPolicy).filter(RetrainingPolicy.model_id == model_id))).scalars().first()
    
    if not policy:
        policy = RetrainingPolicy(model_id=model_id)
        db.add(policy)
        
    policy.enabled = payload.get("enabled", False)
    
    # Defaults handled here if missing
    trigger_conditions = payload.get("trigger_conditions", {})
    trigger_conditions.setdefault("psi_threshold", 0.2)
    trigger_conditions.setdefault("ks_stat_threshold", 0.1)
    trigger_conditions.setdefault("performance_degradation_pct", 15)
    trigger_conditions.setdefault("min_days_since_last_retrain", 7)
    trigger_conditions.setdefault("require_all_conditions", False)
    
    policy.trigger_conditions = trigger_conditions
    policy.retrain_action = payload.get("retrain_action", {"action_type": "notify_only"})
    
    await db.commit()
    await db.refresh(policy)
    return policy


@router.get("/{model_id}/retraining-policy")
async def get_policy(model_id: str, db: AsyncSession = Depends(deps.get_db)):
    policy = (await db.execute(select(RetrainingPolicy).filter(RetrainingPolicy.model_id == model_id))).scalars().first()
    if not policy:
        raise HTTPException(status_code=404, detail="Policy not found")
    return policy


@router.post("/{model_id}/retraining-policy/simulate")
async def simulate_trigger(model_id: str, db: AsyncSession = Depends(deps.get_db)):
    result = await evaluate_retrain_trigger(model_id, db)
    # Remove SQLAlchemy object before returning
    if "policy" in result:
        del result["policy"]
    return result


@router.post("/{model_id}/retraining-policy/trigger-now")
async def trigger_now(model_id: str, db: AsyncSession = Depends(deps.get_db)):
    policy = (await db.execute(select(RetrainingPolicy).filter(RetrainingPolicy.model_id == model_id))).scalars().first()
    if not policy:
        raise HTTPException(status_code=404, detail="Policy not found")
        
    trigger_result = {
        "should_trigger": True,
        "triggered_conditions": ["Manual Admin Trigger"],
        "suppressed": False,
        "suppression_reason": None
    }
    
    event = await execute_retrain_action(policy, trigger_result, db)
    return {"status": "dispatched", "event_id": event.id, "action_result": event.action_result}


@router.get("/{model_id}/retraining-events")
async def get_events(model_id: str, db: AsyncSession = Depends(deps.get_db)):
    events = (await db.execute(
        select(RetrainingEvent)
        .filter(RetrainingEvent.model_id == model_id)
        .order_by(RetrainingEvent.triggered_at.desc())
        .limit(20)
    )).scalars().all()
    return events
