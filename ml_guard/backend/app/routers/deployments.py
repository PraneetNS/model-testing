"""
Deployments Router.
Endpoints for environment management and model promotion.
"""
import uuid
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.db.session import get_db
from app.db.models import (
    Deployment, ModelVersion, Model, Environment, utcnow
)
from app.core.auth import AuthContext, require_role, log_action

router = APIRouter()


# ═══════════════════════════════════════════════
# LIST ENVIRONMENTS
# ═══════════════════════════════════════════════
@router.get("/deployments/environments")
async def list_environments(
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_role("viewer")),
):
    """List all deployment environments."""
    envs = db.query(Environment).filter(
        (Environment.org_id == auth.org_id) | (Environment.org_id.is_(None))
    ).all()

    # If no environments exist, seed defaults
    if not envs:
        for env_name in ["DEV", "STAGING", "PRODUCTION"]:
            env = Environment(
                org_id=auth.org_id,
                name=env_name,
                description=f"{env_name} deployment environment",
            )
            db.add(env)
        db.commit()
        envs = db.query(Environment).filter(Environment.org_id == auth.org_id).all()

    return [
        {
            "id": str(e.id),
            "name": e.name,
            "description": e.description,
            "is_active": e.is_active,
        }
        for e in envs
    ]


# ═══════════════════════════════════════════════
# PROMOTE MODEL (DEV → STAGING → PRODUCTION)
# ═══════════════════════════════════════════════
@router.post("/deployments/promote")
async def promote_model(
    version_id: str,
    target_environment: str,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer")),
):
    """Promote a model version to a target environment."""
    version = db.get(ModelVersion, version_id)
    if not version:
        raise HTTPException(404, "Model version not found.")

    promotion_order = {"DEV": 0, "STAGING": 1, "PRODUCTION": 2}
    if target_environment not in promotion_order:
        raise HTTPException(400, "Target must be DEV, STAGING, or PRODUCTION.")

    # Check governance gate for PRODUCTION
    if target_environment == "PRODUCTION":
        if version.governance_score is not None and version.governance_score < 70:
            raise HTTPException(
                403,
                f"Governance score {version.governance_score} below PRODUCTION threshold (70). "
                "Model cannot be promoted."
            )

    # Check promotion path: must have been deployed to the previous environment
    if promotion_order[target_environment] > 0:
        prev_envs = ["DEV"] if target_environment == "STAGING" else ["DEV", "STAGING"]
        existing = db.query(Deployment).filter(
            Deployment.version_id == version_id,
            Deployment.environment.in_(prev_envs),
            Deployment.status == "ACTIVE",
        ).first()
        if not existing:
            raise HTTPException(
                400,
                f"Model must be deployed to {' or '.join(prev_envs)} before promoting to {target_environment}."
            )

    # Create deployment
    deployment = Deployment(
        version_id=version_id,
        environment=target_environment,
        status="ACTIVE",
        deployed_by=auth.user_id,
    )
    db.add(deployment)
    db.commit()
    db.refresh(deployment)
    log_action(db, auth, "deployment.promote", "deployment", str(deployment.id), {
        "version_id": version_id, "target": target_environment
    })

    return {
        "deployment_id": str(deployment.id),
        "version_id": version_id,
        "environment": target_environment,
        "status": "ACTIVE",
        "promoted_at": str(deployment.deployment_date),
    }


# ═══════════════════════════════════════════════
# ROLLBACK DEPLOYMENT
# ═══════════════════════════════════════════════
@router.post("/deployments/rollback")
async def rollback_deployment(
    deployment_id: str,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer")),
):
    """Rollback a specific deployment."""
    deployment = db.get(Deployment, deployment_id)
    if not deployment:
        raise HTTPException(404, "Deployment not found.")

    deployment.status = "ROLLED_BACK"
    db.commit()
    log_action(db, auth, "deployment.rollback", "deployment", deployment_id, {
        "environment": deployment.environment
    })

    return {
        "deployment_id": deployment_id,
        "status": "ROLLED_BACK",
        "environment": deployment.environment,
    }


# ═══════════════════════════════════════════════
# LIST DEPLOYMENTS
# ═══════════════════════════════════════════════
@router.get("/deployments")
async def list_deployments(
    environment: str = None,
    status: str = None,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_role("viewer")),
):
    """List all deployments with optional filtering."""
    q = db.query(Deployment)
    if environment:
        q = q.filter(Deployment.environment == environment)
    if status:
        q = q.filter(Deployment.status == status)

    total = q.count()
    offset = (page - 1) * per_page
    deployments = q.order_by(Deployment.created_at.desc()).offset(offset).limit(per_page).all()

    items = []
    for d in deployments:
        version = db.get(ModelVersion, str(d.version_id))
        model = db.get(Model, str(version.model_id)) if version else None
        items.append({
            "deployment_id": str(d.id),
            "model_name": model.name if model else "Unknown",
            "version_number": version.version_number if version else None,
            "environment": d.environment,
            "status": d.status,
            "governance_score": version.governance_score if version else None,
            "deployed_at": str(d.deployment_date),
        })

    return {"total": total, "page": page, "per_page": per_page, "items": items}
