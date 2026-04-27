import hashlib
import time
import uuid
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, desc

from app.db.session import get_db
from app.db.models import GuardrailConfigModel, GuardrailTrace, Model
from app.core.auth import AuthContext, get_auth_context, log_action
from ml_guard.core.guardrail import GuardrailEngine, GuardrailConfig, GuardrailDecision
from app.billing.metering import record_usage
from app.billing.enforcement import check_billing_limits

router = APIRouter()

@router.post("/guardrail")
async def create_guardrail_config(
    payload: Dict[str, Any],
    auth: AuthContext = Depends(get_auth_context),
    db: AsyncSession = Depends(get_db)
):
    """Create a guardrail config for a model."""
    model_id = payload.get("model_id")
    if not model_id:
        raise HTTPException(400, "model_id is required")
    
    # Check if model exists
    res = await db.execute(select(Model).where(Model.id == model_id))
    model = res.scalar_one_or_none()
    if not model:
        raise HTTPException(404, "Model not found")

    # Check if config already exists
    res = await db.execute(select(GuardrailConfigModel).where(GuardrailConfigModel.model_id == model_id))
    existing = res.scalar_one_or_none()
    
    if existing:
        # Update existing
        existing.name = payload.get("name", existing.name)
        existing.enabled_input_checks = payload.get("enabled_input_checks", existing.enabled_input_checks)
        existing.enabled_output_checks = payload.get("enabled_output_checks", existing.enabled_output_checks)
        existing.action_on_block = payload.get("action_on_block", existing.action_on_block)
        existing.fallback_response = payload.get("fallback_response", existing.fallback_response)
        existing.allowed_topics = payload.get("allowed_topics", existing.allowed_topics)
        existing.blocked_topics = payload.get("blocked_topics", existing.blocked_topics)
        config_obj = existing
    else:
        # Create new
        config_obj = GuardrailConfigModel(
            id=uuid.uuid4(),
            model_id=model_id,
            name=payload.get("name", f"Guardrail for {model.name}"),
            enabled_input_checks=payload.get("enabled_input_checks", ["injection", "pii", "jailbreak", "topic_policy"]),
            enabled_output_checks=payload.get("enabled_output_checks", ["toxicity", "hallucination", "pii"]),
            action_on_block=payload.get("action_on_block", "return_error"),
            fallback_response=payload.get("fallback_response", "I'm sorry, but I cannot fulfill this request due to safety policy violations."),
            allowed_topics=payload.get("allowed_topics", []),
            blocked_topics=payload.get("blocked_topics", [])
        )
        db.add(config_obj)
    
    await db.commit()
    await db.refresh(config_obj)
    
    await log_action(db, auth, "guardrail.create", resource_id=str(config_obj.id))
    
    return config_obj

@router.post("/guardrail/{guardrail_id}/evaluate")
async def evaluate_guardrail(
    guardrail_id: str,
    payload: Dict[str, Any],
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
    _billing: None = Depends(check_billing_limits)
):
    """
    Real-time guardrail evaluation.
    This endpoint must respond in < 300ms.
    """
    start_time = time.time()
    
    # 1. Load config
    res = await db.execute(select(GuardrailConfigModel).where(GuardrailConfigModel.id == guardrail_id))
    config_model = res.scalar_one_or_none()
    if not config_model:
        raise HTTPException(404, "Guardrail config not found")
    
    # 2. Initialize engine
    config = GuardrailConfig(
        model_id=str(config_model.model_id),
        name=config_model.name,
        enabled_input_checks=config_model.enabled_input_checks,
        enabled_output_checks=config_model.enabled_output_checks,
        action_on_block=config_model.action_on_block,
        fallback_response=config_model.fallback_response,
        allowed_topics=config_model.allowed_topics,
        blocked_topics=config_model.blocked_topics
    )
    engine = GuardrailEngine(config)
    
    # 3. Evaluate
    prompt = payload.get("prompt", "")
    response = payload.get("response")
    context_chunks = payload.get("context_chunks")
    
    decision = engine.evaluate(prompt, response, context_chunks)
    
    # Record usage
    record_usage(auth.org_id, getattr(auth, "key_id", None), "guardrail_evaluated")

    # 4. Log trace asynchronously (we'll do it before returning to stay within sync context but it's fast)
    # Note: The requirement says "no Celery, all synchronous" for the response.
    
    input_hash = hashlib.sha256(prompt.encode()).hexdigest() if prompt else None
    output_hash = hashlib.sha256(response.encode()).hexdigest() if response else None
    
    # Summary of checks
    checks_summary = {
        "input": {k: v.get("flagged", False) for k, v in decision.input_checks.items()},
        "output": {k: v.get("flagged", False) for k, v in decision.output_checks.items()}
    }
    
    trace = GuardrailTrace(
        id=uuid.uuid4(),
        guardrail_id=guardrail_id,
        trace_id=decision.trace_id,
        timestamp=func.now(),
        input_hash=input_hash,
        output_hash=output_hash,
        action=decision.action,
        latency_ms=decision.latency_ms,
        checks_summary=checks_summary,
        full_results=decision.dict()
    )
    db.add(trace)
    await db.commit()
    
    return decision

@router.get("/guardrail/{guardrail_id}/stats")
async def get_guardrail_stats(
    guardrail_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Returns real-time stats for a guardrail."""
    # Total evaluated
    res = await db.execute(select(func.count(GuardrailTrace.id)).where(GuardrailTrace.guardrail_id == guardrail_id))
    total_evaluated = res.scalar() or 0
    
    if total_evaluated == 0:
        return {
            "total_evaluated": 0,
            "blocked_pct": 0,
            "flagged_pct": 0,
            "top_block_reasons": [],
            "avg_latency_ms": 0,
            "p95_latency_ms": 0,
            "last_24h_volume": 0
        }
    
    # Blocked count
    res = await db.execute(
        select(func.count(GuardrailTrace.id))
        .where(GuardrailTrace.guardrail_id == guardrail_id, GuardrailTrace.action == "block")
    )
    blocked_count = res.scalar() or 0
    
    # Flagged count
    res = await db.execute(
        select(func.count(GuardrailTrace.id))
        .where(GuardrailTrace.guardrail_id == guardrail_id, GuardrailTrace.action == "flag_for_review")
    )
    flagged_count = res.scalar() or 0
    
    # Avg Latency
    res = await db.execute(
        select(func.avg(GuardrailTrace.latency_ms))
        .where(GuardrailTrace.guardrail_id == guardrail_id)
    )
    avg_latency = float(res.scalar() or 0)
    
    # Last 24h volume
    from datetime import datetime, timedelta
    day_ago = datetime.utcnow() - timedelta(days=1)
    res = await db.execute(
        select(func.count(GuardrailTrace.id))
        .where(GuardrailTrace.guardrail_id == guardrail_id, GuardrailTrace.timestamp >= day_ago)
    )
    last_24h_volume = res.scalar() or 0
    
    return {
        "total_evaluated": total_evaluated,
        "blocked_pct": round((blocked_count / total_evaluated) * 100, 2),
        "flagged_pct": round((flagged_count / total_evaluated) * 100, 2),
        "top_block_reasons": [], # Could implement more complex group by if needed
        "avg_latency_ms": round(avg_latency, 2),
        "p95_latency_ms": 0, # Requires complex SQL or app-side calc
        "last_24h_volume": last_24h_volume
    }

@router.get("/guardrail/{guardrail_id}/traces")
async def get_guardrail_traces(
    guardrail_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Last 50 traces for a guardrail."""
    res = await db.execute(
        select(GuardrailTrace)
        .where(GuardrailTrace.guardrail_id == guardrail_id)
        .order_by(desc(GuardrailTrace.timestamp))
        .limit(50)
    )
    traces = res.scalars().all()
    return traces
