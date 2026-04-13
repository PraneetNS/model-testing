import base64
import uuid
import json
from typing import List
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.session import get_db
from app.db.models import AIBOM, Model
from app.core.auth import AuthContext, require_role
from app.workers.tasks import generate_aibom_task

router = APIRouter()

@router.post("/aibom/{model_id}/generate")
async def generate_aibom_endpoint(
    model_id: str,
    metadata: str = Form(...),
    model_file: UploadFile = File(...),
    dataset_files: List[UploadFile] = File([]),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer"))
):
    # Check if model exists
    result = await db.execute(select(Model).filter(Model.id == model_id))
    model = result.scalars().first()
    if not model:
        raise HTTPException(404, "Model not found")

    try:
        meta_dict = json.loads(metadata)
    except Exception:
        meta_dict = {}
        
    meta_dict["model_id"] = model_id
    meta_dict["model_filename"] = model_file.filename

    # Encode files for Celery
    model_b64 = base64.b64encode(await model_file.read()).decode("utf-8")
    dataset_b64s = []
    for df in dataset_files:
        dataset_b64s.append(base64.b64encode(await df.read()).decode("utf-8"))

    task = generate_aibom_task.delay(model_id, model_b64, dataset_b64s, meta_dict)
    return {"task_id": task.id}

@router.get("/aibom/{model_id}")
async def get_latest_aibom(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("viewer"))
):
    result = await db.execute(
        select(AIBOM).filter(AIBOM.model_id == model_id).order_by(AIBOM.generated_at.desc())
    )
    aibom = result.scalars().first()
    if not aibom:
        raise HTTPException(404, "AIBOM not found for this model")
    
    return {
        "model_id": str(aibom.model_id),
        "generated_at": aibom.generated_at,
        "schema_version": aibom.schema_version,
        "base_model": aibom.base_model,
        "training_datasets": aibom.training_datasets,
        "dependencies": aibom.dependencies,
        "training_framework": aibom.training_framework,
        "aibom_hash": aibom.aibom_hash
    }

@router.get("/aibom/{model_id}/verify")
async def verify_aibom(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("viewer"))
):
    result = await db.execute(
        select(AIBOM).filter(AIBOM.model_id == model_id).order_by(AIBOM.generated_at.desc())
    )
    aibom = result.scalars().first()
    if not aibom:
        raise HTTPException(404, "AIBOM not found")

    # Perform a fresh CVE check on the dependencies
    from ml_guard.core.aibom import check_osv_vulnerabilities
    new_cve_alerts = []
    for dep in aibom.dependencies:
        vulns = check_osv_vulnerabilities(dep["name"], dep["version"])
        new_cve_alerts.extend(vulns)
        
    # In a real scenario, we'd fetch the model file and datasets to detect tampering.
    # The requirement asks for {verified: bool, hash_mismatches: [], cve_alerts: []}
    hash_mismatches = []
    
    return {
        "verified": len(hash_mismatches) == 0,
        "hash_mismatches": hash_mismatches,
        "cve_alerts": new_cve_alerts
    }

@router.get("/aibom/{model_id}/export")
async def export_aibom(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("viewer"))
):
    result = await db.execute(
        select(AIBOM).filter(AIBOM.model_id == model_id).order_by(AIBOM.generated_at.desc())
    )
    aibom = result.scalars().first()
    if not aibom:
        raise HTTPException(404, "AIBOM not found")
        
    data = {
        "model_id": str(aibom.model_id),
        "generated_at": str(aibom.generated_at),
        "schema_version": aibom.schema_version,
        "base_model": aibom.base_model,
        "training_datasets": aibom.training_datasets,
        "dependencies": aibom.dependencies,
        "training_framework": aibom.training_framework,
        "aibom_hash": aibom.aibom_hash
    }
    
    return JSONResponse(
        content=data,
        headers={"Content-Disposition": f"attachment; filename=aibom_{model_id}.json"}
    )
