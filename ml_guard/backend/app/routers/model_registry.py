"""
Model Registry Router.
Endpoints for enterprise model lifecycle management:
versioning, deployment, and governance tracking.
"""
import uuid
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func
from app.db.session import get_db
from app.db.models import (
    Model, ModelVersion, Deployment, Environment, AuditLog, utcnow
)
from app.core.auth import AuthContext, require_role, log_action
from pydantic import BaseModel, Field, field_validator
import re

class ModelRegisterSchema(BaseModel):
    model_name: str = Field(..., min_length=1, max_length=128)
    description: Optional[str] = ""
    owner: Optional[str] = ""

    @field_validator("model_name")
    @classmethod
    def validate_filename(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_\-\.]{1,128}$", v):
            raise ValueError("model_name must match [a-zA-Z0-9_\-\.]{1,128}")
        if any(char in v for char in ";&|><$()"):
            raise ValueError("Shell metacharacters not allowed in model_name")
        return v

class ModelVersionSchema(BaseModel):
    model_id: str
    framework: Optional[str] = ""
    artifact_url: Optional[str] = ""
    parameters_count: Optional[int] = None
    training_dataset: Optional[str] = ""
    governance_score: Optional[float] = None
    risk_class: Optional[str] = None
    description: Optional[str] = ""

    @field_validator("artifact_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if v and any(char in v for char in ";&|><$()"):
            raise ValueError("Shell metacharacters not allowed in artifact_url")
        return v

router = APIRouter()


# ═══════════════════════════════════════════════
# REGISTER MODEL
# ═══════════════════════════════════════════════
@router.post("/models/register")
async def register_model(
    data: ModelRegisterSchema,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer")),
):
    """Register a new model in the registry."""
    # Check if already exists in org
    existing = (await db.execute(select(Model).filter(
        Model.name == data.model_name,
        Model.project_id.isnot(None)
    ))).scalars().first()
    
    if existing:
        return {
            "model_id": str(existing.id),
            "model_name": existing.name,
            "status": "already_exists",
        }

    model = Model(
        name=data.model_name,
        provider=data.owner or "ML Guard Registry",
        metadata_json={"description": data.description, "owner": data.owner, "registered_via": "api"},
        created_by=auth.user_id,
    )
    db.add(model)
    await db.commit()
    await db.refresh(model)
    await log_action(db, auth, "model.register", "model", str(model.id), {"name": data.model_name})

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
    data: ModelVersionSchema,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer")),
):
    """Create a new version for an existing model."""
    model = (await db.get(Model, data.model_id))
    if not model:
        raise HTTPException(404, "Model not found.")

    # Auto-increment version number
    max_v = (await db.execute(select(func.max(ModelVersion.version_number)).filter(
        ModelVersion.model_id == data.model_id
    ))).scalar() or 0

    version = ModelVersion(
        model_id=data.model_id,
        version_number=max_v + 1,
        framework=data.framework,
        artifact_url=data.artifact_url,
        parameters_count=data.parameters_count,
        training_dataset=data.training_dataset,
        governance_score=data.governance_score,
        risk_class=data.risk_class,
        description=data.description,
        created_by=auth.user_id,
    )
    db.add(version)
    await db.commit()
    await db.refresh(version)
    await log_action(db, auth, "model.version", "model_version", str(version.id), {
        "model_id": data.model_id, "version": max_v + 1
    })

    return {
        "version_id": str(version.id),
        "model_id": data.model_id,
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
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer")),
):
    """Deploy a model version to an environment."""
    version = await db.get(ModelVersion, version_id)
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
    await db.commit()
    await db.refresh(deployment)
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
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("viewer")),
):
    """List all registered models with version counts."""
    offset = (page - 1) * per_page
    
    # Async count
    total_stmt = select(func.count(Model.id))
    total = (await db.execute(total_stmt)).scalar() or 0
    
    # Async items
    models_stmt = select(Model).order_by(Model.created_at.desc()).offset(offset).limit(per_page)
    models = (await db.execute(models_stmt)).scalars().all()

    items = []
    for m in models:
        # Async version count
        v_count_stmt = select(func.count(ModelVersion.id)).filter(ModelVersion.model_id == m.id)
        version_count = (await db.execute(v_count_stmt)).scalar() or 0
        
        # Async latest version
        latest_v_stmt = select(ModelVersion).filter(ModelVersion.model_id == m.id).order_by(ModelVersion.version_number.desc()).limit(1)
        latest_version = (await db.execute(latest_v_stmt)).scalars().first()

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
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("viewer")),
):
    """List all versions of a specific model."""
    model = await db.get(Model, model_id)
    if not model:
        raise HTTPException(404, "Model not found.")

    stmt = select(ModelVersion).filter(
        ModelVersion.model_id == model_id
    ).order_by(ModelVersion.version_number.desc())
    versions = (await db.execute(stmt)).scalars().all()

    items = []
    for v in versions:
        d_stmt = select(Deployment).filter(Deployment.version_id == v.id)
        deployments = (await db.execute(d_stmt)).scalars().all()
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
