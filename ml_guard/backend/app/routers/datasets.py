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
from pydantic import BaseModel

router = APIRouter()


class DatasetRegisterSchema(BaseModel):
    dataset_name: str
    model_id: str
    dataset_type: str = "training"
    row_count: int = 0
    schema_info: str = ""

# ═══════════════════════════════════════════════
# REGISTER DATASET
# ═══════════════════════════════════════════════
@router.post("/datasets/register")
async def register_dataset(
    data: DatasetRegisterSchema,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer")),
):
    """Register a new dataset in the lineage store."""
    dataset_name = data.dataset_name
    model_id = data.model_id
    dataset_type = data.dataset_type
    row_count = data.row_count
    schema_info = data.schema_info

    model = await db.get(Model, model_id)
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
    await log_action(db, auth, "dataset.register", "dataset", str(dataset.id), {"name": dataset_name})

    return {
        "dataset_id": str(dataset.id),
        "name": dataset_name,
        "type": dataset_type,
        "status": "registered",
    }


from fastapi import UploadFile, File, Form
import os
import tempfile

# ═══════════════════════════════════════════════
# UPLOAD DATASET
# ═══════════════════════════════════════════════
@router.post("/datasets/upload")
async def upload_dataset(
    model_id: str = Form(...),
    dataset_name: str = Form(...),
    dataset_type: str = Form("training"),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer")),
):
    """Upload a file and register it as a dataset."""
    model = await db.get(Model, model_id)
    if not model:
        raise HTTPException(404, "Model not found.")

    # Save file to a temporary location or persistent storage
    # For now, we'll use a local 'uploads' directory
    upload_dir = os.path.join(os.getcwd(), "ml_guard", "backend", "uploads", "datasets")
    os.makedirs(upload_dir, exist_ok=True)
    
    file_ext = os.path.splitext(file.filename)[1]
    safe_name = f"{uuid.uuid4()}{file_ext}"
    dest_path = os.path.join(upload_dir, safe_name)
    
    h = hashlib.sha256()
    row_count = 0
    
    with open(dest_path, "wb") as f:
        content = await file.read()
        f.write(content)
        h.update(content)
    
    sha256 = h.hexdigest()
    
    # Try to get row count if CSV
    try:
        import pandas as pd
        if file_ext.lower() == ".csv":
            df = pd.read_csv(dest_path, nrows=100) # Peek
            # We don't read full file to avoid OOM in API, but let's assume we can for smallish files
            df_full = pd.read_csv(dest_path)
            row_count = len(df_full)
            feature_count = len(df_full.columns)
        elif file_ext.lower() == ".parquet":
            df_full = pd.read_parquet(dest_path)
            row_count = len(df_full)
            feature_count = len(df_full.columns)
        else:
            feature_count = 0
    except Exception as e:
        logger.warning(f"Failed to parse uploaded dataset {file.filename}: {e}")
        feature_count = 0

    dataset = Dataset(
        model_id=model_id,
        type=dataset_type,
        metadata_json={
            "name": dataset_name,
            "filename": file.filename,
            "source": "upload",
            "sha256": sha256
        },
        row_count=row_count,
        fingerprint=sha256[:32],
    )
    db.add(dataset)
    await db.flush()

    version = DatasetVersion(
        dataset_id=dataset.id,
        version_number=1,
        storage_url=dest_path,
        schema_hash=sha256[:32],
        row_count=row_count,
        feature_count=feature_count,
        created_by=auth.user_id,
    )
    db.add(version)
    await db.commit()
    await log_action(db, auth, "dataset.upload", "dataset", str(dataset.id), {"name": dataset_name})

    return {
        "dataset_id": str(dataset.id),
        "name": dataset_name,
        "status": "uploaded",
        "row_count": row_count
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
    dataset = await db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(404, "Dataset not found.")

    res = await db.execute(select(func.max(DatasetVersion.version_number)).filter(
        DatasetVersion.dataset_id == dataset_id
    ))
    max_v = res.scalar() or 0

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
    await log_action(db, auth, "dataset.version", "dataset_version", str(version.id), {
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
    total = (await db.execute(select(func.count(Dataset.id)))).scalar() or 0
    datasets = (await db.execute(select(Dataset).order_by(Dataset.created_at.desc()).offset(offset).limit(per_page))).scalars().all()

    items = []
    for d in datasets:
        version_count = (await db.execute(
            select(func.count(DatasetVersion.id)).filter(DatasetVersion.dataset_id == d.id)
        )).scalar() or 0
        model_result = await db.execute(select(Model).filter(Model.id == str(d.model_id)))
        model = model_result.scalar_one_or_none()

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
    dataset = await db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(404, "Dataset not found.")

    res = await db.execute(
        select(DatasetVersion).filter(DatasetVersion.dataset_id == dataset_id).order_by(DatasetVersion.version_number.desc())
    )
    versions = res.scalars().all()

    lineage = []
    for v in versions:
        res = await db.execute(select(LineageLink).filter(LineageLink.dataset_version_id == v.id))
        links = res.scalars().all()
        linked_models = []
        for link in links:
            mv = await db.get(ModelVersion, str(link.model_version_id))
            if mv:
                model = await db.get(Model, str(mv.model_id))
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
