from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from typing import List, Optional
from uuid import UUID

from app.api.v1 import deps
from app.infrastructure.persistence import models as sql_models
from app.domain.services.governance_engine import GovernanceEngine

router = APIRouter()

@router.get("/projects")
async def get_projects(
    db: AsyncSession = Depends(deps.get_db),
    current_user: sql_models.User = Depends(deps.get_current_active_user)
):
    """List all projects for the current tenant."""
    engine = GovernanceEngine(db)
    return await engine.list_projects(current_user.tenant_id)

@router.get("/project/{project_id}/history")
async def get_project_history(
    project_id: str,
    db: AsyncSession = Depends(deps.get_db),
    current_user: sql_models.User = Depends(deps.get_current_active_user)
):
    """Get the full evaluation history for a project."""
    engine = GovernanceEngine(db)
    return await engine.get_project_history(project_id)

@router.get("/project/{project_id}/drift")
async def get_drift_trends(
    project_id: str,
    feature_name: Optional[str] = None,
    db: AsyncSession = Depends(deps.get_db),
    current_user: sql_models.User = Depends(deps.get_current_active_user)
):
    """Get time-series drift metrics for the project."""
    engine = GovernanceEngine(db)
    logs = await engine.get_drift_trends(project_id, feature_name)
    
    # Format for charts
    return [
        {
            "timestamp": log.timestamp,
            "feature": log.feature_name,
            "psi": log.psi_score
        } for log in logs
    ]

@router.get("/audit-trail")
async def get_audit_trail(
    db: AsyncSession = Depends(deps.get_db),
    current_user: sql_models.User = Depends(deps.get_current_active_user)
):
    """
    Fetch the audit trail for compliance. 
    Auditors can see all tenant logs, Developers see their own.
    """
    stmt = select(sql_models.AuditLog)
    if current_user.role != "auditor" and current_user.role != "admin":
        stmt = stmt.filter(sql_models.AuditLog.user_id == current_user.id)
    else:
        # Join with users to filter by tenant
        stmt = stmt.join(sql_models.User).filter(sql_models.User.tenant_id == current_user.tenant_id)
    
    result = await db.execute(stmt.order_by(sql_models.AuditLog.timestamp.desc()))
    return result.scalars().all()

from pydantic import BaseModel

class OverrideRequest(BaseModel):
    run_id: str
    reason: str

@router.post("/project/{project_id}/override")
async def manual_override(
    project_id: str,
    request: OverrideRequest,
    db: AsyncSession = Depends(deps.get_db),
    current_user: sql_models.User = Depends(deps.get_current_active_user)
):
    """Enable manual override with audit log for blocked deployments."""
    run = (await db.execute(select(sql_models.TestRun).filter(sql_models.TestRun.id == request.run_id, sql_models.TestRun.project_id == project_id))).scalars().first()
    if not run:
        raise HTTPException(status_code=404, detail="TestRun not found")
        
    if run.deployment_allowed:
        return {"status": "already_allowed"}
        
    # Apply override
    run.deployment_allowed = True
    
    # Audit log
    audit = sql_models.AuditLog(
        user_id=current_user.id,
        action="manual_override",
        details={
            "run_id": str(run.id),
            "project_id": project_id,
            "reason": request.reason
        }
    )
    db.add(audit)
    await db.commit()
    
    return {"status": "success", "message": "Deployment allowed via manual override"}
