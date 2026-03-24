"""
Model Registry Router.
Endpoints for enterprise model lifecycle management:
versioning, deployment, and governance tracking.
"""
import uuid
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.db.session import get_db
from app.db.models import (
    Model, ModelVersion, Deployment, Environment, AuditLog, utcnow
)
from app.core.auth import AuthContext, require_role, log_action

router = APIRouter()


# ═══════════════════════════════════════════════
# REGISTER MODEL
# ═══════════════════════════════════════════════
@router.post("/models/register")
async def register_model(
    model_name: str,
    description: str = "",
    owner: str = "",
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer")),
):
    """Register a new model in the registry."""
    # Check if already exists in org
    existing = db.query(Model).filter(
        Model.name == model_name,
        Model.project_id.isnot(None),
    ).first()
    if existing:
        return {
            "model_id": str(existing.id),
            "model_name": existing.name,
            "status": "already_exists",
        }

    model = Model(
        name=model_name,
        provider=owner or "ML Guard Registry",
        metadata_json={"description": description, "owner": owner, "registered_via": "api"},
        created_by=auth.user_id,
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    log_action(db, auth, "model.register", "model", str(model.id), {"name": model_name})

    return {
        "model_id": str(model.id),
        "model_name": model.name,
        "status": "registered",
        "created_at": str(model.created_at),
    }


# ═══════════════════════════════════════════════
# CREATE MODEL VERSION
# ═══════════════════════════════════════════════
@router.post("/models/version")
async def create_version(
    model_id: str,
    framework: str = "",
    artifact_url: str = "",
    parameters_count: int = None,
    training_dataset: str = "",
    governance_score: float = None,
    risk_class: str = None,
    description: str = "",
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer")),
):
    """Create a new version for an existing model."""
    model = db.get(Model, model_id)
    if not model:
        raise HTTPException(404, "Model not found.")

    # Auto-increment version number
    max_v = db.query(func.max(ModelVersion.version_number)).filter(
        ModelVersion.model_id == model_id
    ).scalar() or 0

    version = ModelVersion(
        model_id=model_id,
        version_number=max_v + 1,
        framework=framework,
        artifact_url=artifact_url,
        parameters_count=parameters_count,
        training_dataset=training_dataset,
        governance_score=governance_score,
        risk_class=risk_class,
        description=description,
        created_by=auth.user_id,
    )
    db.add(version)
    db.commit()
    db.refresh(version)
    log_action(db, auth, "model.version", "model_version", str(version.id), {
        "model_id": model_id, "version": max_v + 1
    })

    return {
        "version_id": str(version.id),
        "model_id": model_id,
        "version_number": version.version_number,
        "status": "created",
    }


# ═══════════════════════════════════════════════
# DEPLOY MODEL VERSION
# ═══════════════════════════════════════════════
@router.post("/models/deploy")
async def deploy_model(
    version_id: str,
    environment: str = "DEV",
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer")),
):
    """Deploy a model version to an environment."""
    version = db.get(ModelVersion, version_id)
    if not version:
        raise HTTPException(404, "Model version not found.")

    if environment not in ("DEV", "STAGING", "PRODUCTION"):
        raise HTTPException(400, "Environment must be DEV, STAGING, or PRODUCTION.")

    # If deploying to PRODUCTION, check governance score
    if environment == "PRODUCTION" and version.governance_score is not None:
        if version.governance_score < 70:
            raise HTTPException(
                403,
                f"Governance score {version.governance_score} is below the PRODUCTION threshold of 70. "
                "Deployment blocked by governance policy."
            )

    deployment = Deployment(
        version_id=version_id,
        environment=environment,
        status="ACTIVE",
        deployed_by=auth.user_id,
    )
    db.add(deployment)
    db.commit()
    db.refresh(deployment)
    log_action(db, auth, "model.deploy", "deployment", str(deployment.id), {
        "version_id": version_id, "environment": environment
    })

    return {
        "deployment_id": str(deployment.id),
        "version_id": version_id,
        "environment": environment,
        "status": "ACTIVE",
        "deployed_at": str(deployment.deployment_date),
    }


# ═══════════════════════════════════════════════
# LIST MODELS
# ═══════════════════════════════════════════════
@router.get("/models")
async def list_models(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_role("viewer")),
):
    """List all registered models with version counts."""
    offset = (page - 1) * per_page
    total = db.query(func.count(Model.id)).scalar() or 0
    models = db.query(Model).order_by(Model.created_at.desc()).offset(offset).limit(per_page).all()

    items = []
    for m in models:
        version_count = db.query(func.count(ModelVersion.id)).filter(
            ModelVersion.model_id == m.id
        ).scalar() or 0
        latest_version = db.query(ModelVersion).filter(
            ModelVersion.model_id == m.id
        ).order_by(ModelVersion.version_number.desc()).first()

        items.append({
            "model_id": str(m.id),
            "name": m.name,
            "provider": m.provider,
            "version_count": version_count,
            "latest_version": latest_version.version_number if latest_version else 0,
            "latest_governance_score": latest_version.governance_score if latest_version else None,
            "latest_risk_class": latest_version.risk_class if latest_version else None,
            "created_at": str(m.created_at),
        })

    return {"total": total, "page": page, "per_page": per_page, "items": items}


# ═══════════════════════════════════════════════
# LIST MODEL VERSIONS
# ═══════════════════════════════════════════════
@router.get("/models/{model_id}/versions")
async def list_versions(
    model_id: str,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_role("viewer")),
):
    """List all versions of a specific model."""
    model = db.get(Model, model_id)
    if not model:
        raise HTTPException(404, "Model not found.")

    versions = db.query(ModelVersion).filter(
        ModelVersion.model_id == model_id
    ).order_by(ModelVersion.version_number.desc()).all()

    items = []
    for v in versions:
        deployments = db.query(Deployment).filter(Deployment.version_id == v.id).all()
        items.append({
            "version_id": str(v.id),
            "version_number": v.version_number,
            "framework": v.framework,
            "parameters_count": v.parameters_count,
            "governance_score": v.governance_score,
            "risk_class": v.risk_class,
            "artifact_url": v.artifact_url,
            "deployments": [
                {"environment": d.environment, "status": d.status, "date": str(d.deployment_date)}
                for d in deployments
            ],
            "created_at": str(v.created_at),
        })

    return {
        "model_id": model_id,
        "model_name": model.name,
        "versions": items,
    }
