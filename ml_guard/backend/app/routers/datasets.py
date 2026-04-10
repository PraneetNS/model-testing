"""
Dataset Lineage Router.
Endpoints for dataset registration, versioning, and lineage tracking.
"""
import uuid
import hashlib
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func
from app.db.session import get_db
from app.db.models import (
    Dataset, DatasetVersion, LineageLink, ModelVersion, Model, utcnow
)
from app.core.auth import AuthContext, require_role, log_action

router = APIRouter()


# ═══════════════════════════════════════════════
# REGISTER DATASET
# ═══════════════════════════════════════════════
@router.post("/datasets/register")
async def register_dataset(
    dataset_name: str,
    model_id: str,
    dataset_type: str = "training",
    row_count: int = 0,
    schema_info: str = "",
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer")),
):
    """Register a new dataset in the lineage store."""
    model = db.get(Model, model_id)
    if not model:
        raise HTTPException(404, "Model not found.")

    schema_hash = hashlib.sha256(schema_info.encode()).hexdigest()[:32] if schema_info else None

    dataset = Dataset(
        model_id=model_id,
        type=dataset_type,
        metadata_json={"name": dataset_name, "schema_hash": schema_hash},
        row_count=row_count,
        fingerprint=schema_hash,
    )
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)
    log_action(db, auth, "dataset.register", "dataset", str(dataset.id), {"name": dataset_name})

    return {
        "dataset_id": str(dataset.id),
        "name": dataset_name,
        "type": dataset_type,
        "status": "registered",
    }


# ═══════════════════════════════════════════════
# CREATE DATASET VERSION
# ═══════════════════════════════════════════════
@router.post("/datasets/version")
async def create_dataset_version(
    dataset_id: str,
    storage_url: str = "",
    row_count: int = None,
    feature_count: int = None,
    schema_hash: str = "",
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer")),
):
    """Create a new versioned snapshot of a dataset."""
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(404, "Dataset not found.")

    max_v = db.query(func.max(DatasetVersion.version_number)).filter(
        DatasetVersion.dataset_id == dataset_id
    ).scalar() or 0

    version = DatasetVersion(
        dataset_id=dataset_id,
        version_number=max_v + 1,
        storage_url=storage_url,
        schema_hash=schema_hash,
        row_count=row_count,
        feature_count=feature_count,
        created_by=auth.user_id,
    )
    db.add(version)
    await db.commit()
    await db.refresh(version)
    log_action(db, auth, "dataset.version", "dataset_version", str(version.id), {
        "dataset_id": dataset_id, "version": max_v + 1
    })

    return {
        "version_id": str(version.id),
        "dataset_id": dataset_id,
        "version_number": version.version_number,
        "status": "created",
    }


# ═══════════════════════════════════════════════
# LIST DATASETS
# ═══════════════════════════════════════════════
@router.get("/datasets")
async def list_datasets(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("viewer")),
):
    """List all registered datasets."""
    offset = (page - 1) * per_page
    total = db.query(func.count(Dataset.id)).scalar() or 0
    datasets = (await db.execute(select(Dataset).order_by(Dataset.created_at.desc()).offset(offset).limit(per_page))).scalars().all()

    items = []
    for d in datasets:
        version_count = db.query(func.count(DatasetVersion.id)).filter(
            DatasetVersion.dataset_id == d.id
        ).scalar() or 0
        model = db.get(Model, str(d.model_id))

        name = (d.metadata_json or {}).get("name")
        if not name:
            name = f"{model.name} Assets" if model else "Dataset Asset"

        items.append({
            "dataset_id": str(d.id),
            "name": name,
            "type": d.type,
            "row_count": d.row_count,
            "model_name": model.name if model else None,
            "version_count": version_count,
            "created_at": str(d.created_at),
        })

    return {"total": total, "page": page, "per_page": per_page, "items": items}


# ═══════════════════════════════════════════════
# GET DATASET LINEAGE
# ═══════════════════════════════════════════════
@router.get("/datasets/{dataset_id}/lineage")
async def get_lineage(
    dataset_id: str,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("viewer")),
):
    """Get full lineage for a dataset: versions and linked models."""
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(404, "Dataset not found.")

    versions = db.query(DatasetVersion).filter(
        DatasetVersion.dataset_id == dataset_id
    ).order_by(DatasetVersion.version_number.desc()).all()

    lineage = []
    for v in versions:
        links = db.query(LineageLink).filter(
            LineageLink.dataset_version_id == v.id
        ).all()
        linked_models = []
        for link in links:
            mv = db.get(ModelVersion, str(link.model_version_id))
            if mv:
                model = db.get(Model, str(mv.model_id))
                linked_models.append({
                    "model_name": model.name if model else "Unknown",
                    "model_version": mv.version_number,
                    "link_type": link.link_type,
                    "created_at": str(link.created_at),
                })
        lineage.append({
            "version_id": str(v.id),
            "version_number": v.version_number,
            "row_count": v.row_count,
            "feature_count": v.feature_count,
            "schema_hash": v.schema_hash,
            "storage_url": v.storage_url,
            "linked_models": linked_models,
            "created_at": str(v.created_at),
        })

    return {
        "dataset_id": dataset_id,
        "name": (dataset.metadata_json or {}).get("name", "Unknown"),
        "type": dataset.type,
        "lineage": lineage,
    }
