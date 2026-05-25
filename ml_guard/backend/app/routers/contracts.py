"""
contracts.py — Model Behavior Contract Router

Endpoints for creating/managing behavioral contracts and querying breaches.
All contract mutations and queries are protected by the standard X-API-Key auth.

Routes:
  POST   /api/v1/contracts                               — create contract
  GET    /api/v1/contracts/{model_id}                    — list contracts
  PATCH  /api/v1/contracts/{contract_id}/deactivate      — deactivate
  DELETE /api/v1/contracts/{contract_id}                 — hard delete
  GET    /api/v1/contracts/{model_id}/breaches           — list breaches
  GET    /api/v1/contracts/{model_id}/breach-summary     — summary + penalty
  PATCH  /api/v1/contracts/breaches/{breach_id}/resolve  — mark resolved
  POST   /api/v1/contracts/validate                      — dry-run check
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db.session import get_db
from app.db.models import ModelContract, ContractBreach
from app.services.contract_engine import ContractEngine

router = APIRouter()
_engine = ContractEngine()


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class PromiseSchema(BaseModel):
    name: str = Field(..., description="Human-readable promise name")
    type: str = Field(
        ...,
        description="Promise type: output | latency | distribution | feature_range | fairness"
    )
    metric: str = Field(..., description="Metric key to evaluate")
    operator: str = Field(..., description="Comparison operator: lte | gte | lt | gt | eq | neq")
    threshold: float = Field(..., description="Numeric threshold value")
    severity: str = Field(default="HIGH", description="Breach severity: LOW | MEDIUM | HIGH | CRITICAL")
    action: str = Field(default="alert", description="On breach: alert | flag | block")
    window_hours: Optional[int] = Field(default=24, description="Rolling window for distribution/fairness checks")
    protected_attribute: Optional[str] = Field(default=None, description="Feature key for fairness checks")
    feature_key: Optional[str] = Field(default=None, description="Feature key for feature_range checks")


class ContractCreate(BaseModel):
    name: str = Field(..., description="Contract display name")
    model_id: str = Field(..., description="Model ID this contract applies to")
    version: str = Field(default="1.0", description="Contract version label")
    description: Optional[str] = Field(default=None)
    promises: List[PromiseSchema] = Field(..., min_length=1, description="List of behavioral promises")
    breach_grace_period_minutes: int = Field(default=5, description="Grace period before penalties apply")
    breach_window_minutes: int = Field(default=60, description="Window for penalty calculation")


class ValidateRequest(BaseModel):
    model_id: str
    prediction: Any
    prediction_proba: Optional[float] = None
    features: Optional[Dict[str, Any]] = None
    latency_ms: Optional[float] = None


# ── CRUD ───────────────────────────────────────────────────────────────────────

@router.post("/contracts", status_code=201, tags=["contracts"])
async def create_contract(
    req: ContractCreate,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Create a new behavioral contract for a model.

    Once created and active, every prediction ingested via
    POST /api/v1/ingest/predict will be checked against all promises.
    Breaches are stored in contract_breaches and deducted from the
    governance score.
    """
    contract = ModelContract(
        id=uuid.uuid4(),
        model_id=req.model_id,
        name=req.name,
        version=req.version,
        description=req.description,
        promises=[p.model_dump() for p in req.promises],
        is_active=True,
        breach_grace_period_minutes=req.breach_grace_period_minutes,
        breach_window_minutes=req.breach_window_minutes,
    )
    db.add(contract)
    await db.commit()
    await db.refresh(contract)

    return {
        "contract_id": str(contract.id),
        "model_id": contract.model_id,
        "name": contract.name,
        "version": contract.version,
        "promises_count": len(req.promises),
        "status": "active",
        "breach_grace_period_minutes": contract.breach_grace_period_minutes,
        "breach_window_minutes": contract.breach_window_minutes,
        "created_at": contract.created_at.isoformat(),
    }


@router.get("/contracts", tags=["contracts"])
async def list_all_contracts(
    model_id: Optional[str] = Query(None),
    active_only: bool = Query(default=False),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    q = select(ModelContract)
    if model_id:
        q = q.filter(ModelContract.model_id == model_id)
    if active_only:
        q = q.filter(ModelContract.is_active.is_(True))
    contracts = (await db.execute(q.order_by(ModelContract.created_at.desc()))).scalars().all()
    
    # Map model names dynamically to each contract item
    from app.db.models import Model
    models_res = await db.execute(select(Model))
    models = models_res.scalars().all()
    model_map = {str(m.id): m.name for m in models}

    items = [
        {
            "id": str(c.id),
            "model_id": str(c.model_id),
            "model_name": model_map.get(str(c.model_id), "Unknown Model"),
            "name": c.name,
            "contract_type": "behavioral",
            "status": "active" if c.is_active else "inactive",
            "definition": c.promises,
            "created_at": c.created_at.isoformat(),
        }
        for c in contracts
    ]
    return {"items": items, "total": len(items)}

@router.get("/contracts/{model_id}", tags=["contracts"])
async def list_contracts(
    model_id: str,
    active_only: bool = Query(default=False),
    db: AsyncSession = Depends(get_db),
) -> List[Dict[str, Any]]:
    """List all contracts (or only active ones) for a model."""
    q = select(ModelContract).filter(ModelContract.model_id == model_id)
    if active_only:
        q = q.filter(ModelContract.is_active.is_(True))
    contracts = (await db.execute(q.order_by(ModelContract.created_at.desc()))).scalars().all()

    return [
        {
            "contract_id": str(c.id),
            "name": c.name,
            "version": c.version,
            "description": c.description,
            "is_active": c.is_active,
            "breach_grace_period_minutes": c.breach_grace_period_minutes,
            "breach_window_minutes": c.breach_window_minutes,
            "promises_count": len(c.promises or []),
            "promises": c.promises,
            "created_at": c.created_at.isoformat(),
        }
        for c in contracts
    ]


@router.patch("/contracts/{contract_id}/deactivate", tags=["contracts"])
async def deactivate_contract(
    contract_id: str,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Deactivate a contract. Stops checking inbound predictions immediately.
    Historical breaches are preserved.
    """
    contract = await db.get(ModelContract, contract_id)
    if not contract:
        raise HTTPException(status_code=404, detail="Contract not found")
    contract.is_active = False
    await db.commit()
    return {"status": "deactivated", "contract_id": contract_id}


@router.delete("/contracts/{contract_id}", status_code=204, tags=["contracts"])
async def delete_contract(
    contract_id: str,
    db: AsyncSession = Depends(get_db),
) -> None:
    """
    Hard-delete a contract and all associated breaches (CASCADE).
    Use with caution — breach history will be permanently lost.
    """
    contract = await db.get(ModelContract, contract_id)
    if not contract:
        raise HTTPException(status_code=404, detail="Contract not found")
    await db.delete(contract)
    await db.commit()


# ── Breach Reporting ────────────────────────────────────────────────────────────

@router.get("/contracts/{model_id}/breaches", tags=["contracts"])
async def get_breaches(
    model_id: str,
    hours: int = Query(default=24, ge=1, le=720),
    resolved: Optional[bool] = Query(default=None),
    severity: Optional[str] = Query(default=None),
    promise_type: Optional[str] = Query(default=None),
    db: AsyncSession = Depends(get_db),
) -> List[Dict[str, Any]]:
    """
    List contract breaches for a model within a time window.
    Supports filtering by resolved status, severity, and promise type.
    """
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    q = select(ContractBreach).filter(
        ContractBreach.model_id == model_id,
        ContractBreach.created_at >= cutoff,
    )
    if resolved is not None:
        q = q.filter(ContractBreach.resolved == resolved)
    if severity:
        q = q.filter(ContractBreach.severity == severity.upper())
    if promise_type:
        q = q.filter(ContractBreach.promise_type == promise_type)

    breaches = (await db.execute(q.order_by(ContractBreach.created_at.desc()).limit(500))).scalars().all()

    return [
        {
            "breach_id": str(b.id),
            "contract_id": str(b.contract_id),
            "promise_name": b.promise_name,
            "promise_type": b.promise_type,
            "expected": b.expected,
            "actual": b.actual,
            "severity": b.severity,
            "resolved": b.resolved,
            "created_at": b.created_at.isoformat(),
            "prediction_log_id": str(b.prediction_log_id) if b.prediction_log_id else None,
        }
        for b in breaches
    ]


@router.get("/contracts/{contract_id}/breach-summary", tags=["contracts"])
async def get_breach_summary(
    contract_id: str,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Governance-linked breach summary for a contract.
    """
    return await _engine.get_contract_breach_summary(db, contract_id)


@router.patch("/contracts/breaches/{breach_id}/resolve", tags=["contracts"])
async def resolve_breach(
    breach_id: str,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Mark a specific breach as resolved."""
    breach = await db.get(ContractBreach, breach_id)
    if not breach:
        raise HTTPException(status_code=404, detail="Breach not found")
    breach.resolved = True
    await db.commit()
    return {"status": "resolved", "breach_id": breach_id}


@router.patch("/contracts/{model_id}/breaches/resolve-all", tags=["contracts"])
async def resolve_all_breaches(
    model_id: str,
    hours: int = Query(default=24),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Bulk-resolve all open breaches for a model in the last N hours."""
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    q = select(ContractBreach).filter(
        ContractBreach.model_id == model_id,
        ContractBreach.created_at >= cutoff,
        ContractBreach.resolved.is_(False),
    )
    updated = (await db.execute(q)).scalars().all()
    for b in updated:
        b.resolved = True
    await db.commit()
    return {"resolved_count": len(updated), "model_id": model_id}


# ── Dry-run validation ─────────────────────────────────────────────────────────

@router.post("/contracts/validate", tags=["contracts"])
async def validate_prediction(
    req: ValidateRequest,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Dry-run contract check for a prediction without persisting anything.
    Useful for testing contracts before activating them or during CI.
    """
    breaches = await _engine.check_prediction(
        db=db,
        model_id=req.model_id,
        prediction=req.prediction,
        prediction_proba=req.prediction_proba,
        features=req.features or {},
        latency_ms=req.latency_ms,
        log_id=None,
    )
    return {
        "model_id": req.model_id,
        "compliant": len(breaches) == 0,
        "breach_count": len(breaches),
        "breaches": breaches,
    }


# ── Manual Evaluation ─────────────────────────────────────────────────────────

@router.post("/contracts/{contract_id}/evaluate", tags=["contracts"])
async def evaluate_contract(
    contract_id: str,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Manually evaluate a contract's promises against the latest 100 predictions.
    Any violations are recorded as ContractBreach records.
    """
    contract = await db.get(ModelContract, uuid.UUID(contract_id))
    if not contract:
        raise HTTPException(status_code=404, detail="Contract not found")
        
    # Get associated model
    from app.db.models import Model
    result = await db.execute(select(Model).filter(Model.id == contract.model_id))
    model = result.scalars().first()
    
    # Query prediction logs
    from app.db.models import PredictionLog, ModelVersion, ContractBreach
    from sqlalchemy import or_
    
    v_result = await db.execute(select(ModelVersion.id).filter(ModelVersion.model_id == contract.model_id))
    version_ids = v_result.scalars().all()
    
    conds = [
        PredictionLog.model_id == str(contract.model_id)
    ]
    if model:
        conds.append(PredictionLog.model_id == model.name)
    if version_ids:
        conds.append(PredictionLog.model_version_id.in_(version_ids))
        
    stmt = select(PredictionLog).filter(or_(*conds)).order_by(PredictionLog.timestamp.desc()).limit(100)
    logs_result = await db.execute(stmt)
    logs = logs_result.scalars().all()
    
    if not logs:
        return {
            "status": "evaluated",
            "verdict": "PASSED",
            "breach_rate": 0.0,
            "predictions_evaluated": 0,
            "breach_count": 0
        }
        
    breached_logs_count = 0
    total_breaches_created = 0
    
    for log in logs:
        log_had_breach = False
        features_dict = log.features or {}
        
        for promise in (contract.promises or []):
            breach_dict = await _engine._check_promise(
                db=db,
                model_id=str(contract.model_id),
                contract_id=str(contract.id),
                promise=promise,
                prediction=log.prediction,
                prediction_proba=log.prediction_proba,
                features=features_dict,
                latency_ms=log.latency_ms,
                log_id=str(log.id),
            )
            
            if breach_dict:
                log_had_breach = True
                
                # Check if this breach already exists for this prediction log and promise
                existing_stmt = select(ContractBreach).filter(
                    ContractBreach.contract_id == contract.id,
                    ContractBreach.prediction_log_id == log.id,
                    ContractBreach.promise_name == promise.get("name")
                )
                existing_res = await db.execute(existing_stmt)
                existing = existing_res.scalars().first()
                
                if not existing:
                    # Save a new breach
                    new_breach = ContractBreach(
                        id=uuid.uuid4(),
                        contract_id=contract.id,
                        model_id=str(contract.model_id),
                        promise_name=promise.get("name", "unknown"),
                        promise_type=promise.get("type", "unknown"),
                        expected=str(promise.get("threshold", "")),
                        actual=str(breach_dict.get("actual", "")),
                        prediction_log_id=log.id,
                        severity=promise.get("severity", "HIGH"),
                        resolved=False,
                    )
                    db.add(new_breach)
                    total_breaches_created += 1
                    
        if log_had_breach:
            breached_logs_count += 1
            
    if total_breaches_created > 0:
        await db.commit()
        
    breach_rate = breached_logs_count / len(logs)
    verdict = "FAILED" if breach_rate > 0.05 else "PASSED"
    
    return {
        "status": "evaluated",
        "verdict": verdict,
        "breach_rate": round(breach_rate, 4),
        "predictions_evaluated": len(logs),
        "breach_count": breached_logs_count
    }

