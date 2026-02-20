from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from uuid import UUID

from app.api.v1 import deps
from app.infrastructure.persistence import models as sql_models
from app.domain.services.governance_engine import GovernanceEngine

router = APIRouter()

@router.get("/projects")
def get_projects(
    db: Session = Depends(deps.get_db),
    current_user: sql_models.User = Depends(deps.get_current_active_user)
):
    """List all projects for the current tenant."""
    engine = GovernanceEngine(db)
    return engine.list_projects(current_user.tenant_id)

@router.get("/project/{project_id}/history")
def get_project_history(
    project_id: str,
    db: Session = Depends(deps.get_db),
    current_user: sql_models.User = Depends(deps.get_current_active_user)
):
    """Get the full evaluation history for a project."""
    engine = GovernanceEngine(db)
    return engine.get_project_history(project_id)

@router.get("/project/{project_id}/drift")
def get_drift_trends(
    project_id: str,
    feature_name: Optional[str] = None,
    db: Session = Depends(deps.get_db),
    current_user: sql_models.User = Depends(deps.get_current_active_user)
):
    """Get time-series drift metrics for the project."""
    engine = GovernanceEngine(db)
    logs = engine.get_drift_trends(project_id, feature_name)
    
    # Format for charts
    return [
        {
            "timestamp": log.timestamp,
            "feature": log.feature_name,
            "psi": log.psi_score
        } for log in logs
    ]

@router.get("/audit-trail")
def get_audit_trail(
    db: Session = Depends(deps.get_db),
    current_user: sql_models.User = Depends(deps.get_current_active_user)
):
    """
    Fetch the audit trail for compliance. 
    Auditors can see all tenant logs, Developers see their own.
    """
    query = db.query(sql_models.AuditLog)
    if current_user.role != "auditor" and current_user.role != "admin":
        query = query.filter(sql_models.AuditLog.user_id == current_user.id)
    else:
        # Join with users to filter by tenant
        query = query.join(sql_models.User).filter(sql_models.User.tenant_id == current_user.tenant_id)
        
    return query.order_by(sql_models.AuditLog.timestamp.desc()).all()
