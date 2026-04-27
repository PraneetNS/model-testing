import csv
import io
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, desc

from app.db.session import get_db
from app.db.models import Model, ScanRecord, utcnow
from app.core.auth import AuthContext, get_auth_context
from app.services.inventory_service import compute_next_validation, auto_tier_model_logic
from ml_guard.core.risk_tiering import compute_risk_tier

router = APIRouter(prefix="/api/inventory", tags=["Inventory"])

@router.get("")
async def list_inventory(
    risk_tier: Optional[str] = None,
    environment: Optional[str] = None,
    overdue_validation: Optional[bool] = None,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context)
):
    """Returns all models with their risk and compliance status."""
    stmt = select(Model)
    
    if risk_tier:
        stmt = stmt.where(Model.risk_tier == risk_tier)
    if environment:
        stmt = stmt.where(Model.deployment_environment == environment)
    if overdue_validation:
        stmt = stmt.where(Model.next_validation_due_at < utcnow())
        
    res = await db.execute(stmt)
    models = res.scalars().all()
    
    results = []
    for m in models:
        # Get last governance score
        last_scan = await db.execute(
            select(ScanRecord)
            .where(ScanRecord.model_id == m.id)
            .order_by(desc(ScanRecord.created_at))
            .limit(1)
        )
        scan = last_scan.scalar_one_or_none()
        
        results.append({
            "id": str(m.id),
            "name": m.name,
            "risk_tier": m.risk_tier,
            "governance_score": scan.governance_score if scan else None,
            "deployment_environment": m.deployment_environment,
            "next_validation_due_at": m.next_validation_due_at.isoformat() if m.next_validation_due_at else None,
            "business_owner": m.business_owner,
            "technical_owner": m.technical_owner,
            "compliance_status": scan.gate_status if scan else "PENDING"
        })
        
    return results

@router.get("/dashboard")
async def inventory_dashboard(
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context)
):
    """Aggregate stats for the inventory dashboard."""
    # Total Models
    res = await db.execute(select(func.count(Model.id)))
    total_models = res.scalar() or 0
    
    # By Risk Tier
    res = await db.execute(select(Model.risk_tier, func.count(Model.id)).group_by(Model.risk_tier))
    by_risk_tier = {row[0] or "unassigned": row[1] for row in res.all()}
    
    # By Environment
    res = await db.execute(select(Model.deployment_environment, func.count(Model.id)).group_by(Model.deployment_environment))
    by_environment = {row[0] or "unassigned": row[1] for row in res.all()}
    
    # Overdue Validations
    res = await db.execute(select(func.count(Model.id)).where(Model.next_validation_due_at < utcnow()))
    overdue_count = res.scalar() or 0
    
    # Without Owner
    res = await db.execute(select(func.count(Model.id)).where(Model.business_owner == None))
    no_owner_count = res.scalar() or 0
    
    # Avg Gov Score by Tier
    # This is a bit more complex, let's just return placeholders for now or do a joined query
    
    return {
        "total_models": total_models,
        "by_risk_tier": by_risk_tier,
        "by_environment": by_environment,
        "overdue_validations_count": overdue_count,
        "models_without_owner": no_owner_count,
    }

@router.put("/{model_id}/metadata")
async def update_metadata(
    model_id: str,
    payload: Dict[str, Any],
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context)
):
    """Update risk metadata fields."""
    model = await db.get(Model, model_id)
    if not model:
        raise HTTPException(404, "Model not found")
    
    fields = [
        "risk_tier", "risk_tier_justification", "use_case_category",
        "business_owner", "technical_owner", "deployment_environment",
        "model_type", "training_data_sensitivity", "last_validated_at",
        "regulatory_jurisdictions"
    ]
    
    for f in fields:
        if f in payload:
            val = payload[f]
            if f == "last_validated_at" and val:
                val = datetime.fromisoformat(val.replace("Z", ""))
            setattr(model, f, val)
    
    await db.commit()
    
    # Recalculate validation schedule if tier changed
    await compute_next_validation(str(model.id), db)
    
    return model

@router.post("/{model_id}/auto-tier")
async def auto_tier_model(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context)
):
    """Runs compute_risk_tier and saves the result."""
    result = await auto_tier_model_logic(model_id, db)
    if not result:
        raise HTTPException(404, "Model not found")
    return result

@router.get("/{model_id}/validation-calendar")
async def validation_calendar(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context)
):
    """Returns validation history + upcoming schedule."""
    model = await db.get(Model, model_id)
    if not model:
        raise HTTPException(404, "Model not found")
        
    return {
        "model_id": str(model.id),
        "last_validated_at": model.last_validated_at,
        "next_validation_due_at": model.next_validation_due_at,
        "validation_frequency_days": model.validation_frequency_days,
        "history": [] # Could extend with a ValidationLog table if needed
    }

@router.get("/export")
async def export_inventory(
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context)
):
    """Returns all inventory data as a downloadable CSV."""
    res = await db.execute(select(Model))
    models = res.scalars().all()
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    header = [
        "ID", "Name", "Risk Tier", "Environment", "Owner", "Technical Owner",
        "Next Validation", "Model Type", "Jurisdictions", "Use Case"
    ]
    writer.writerow(header)
    
    for m in models:
        writer.writerow([
            str(m.id), m.name, m.risk_tier, m.deployment_environment,
            m.business_owner, m.technical_owner,
            m.next_validation_due_at.isoformat() if m.next_validation_due_at else "",
            m.model_type, ",".join(m.regulatory_jurisdictions or []), m.use_case_category
        ])
        
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=model_inventory.csv"}
    )
