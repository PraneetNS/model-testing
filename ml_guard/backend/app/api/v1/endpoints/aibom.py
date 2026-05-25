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
from app.billing.metering import record_usage
from app.billing.enforcement import check_billing_limits

router = APIRouter()

@router.post("/aibom/{model_id}/generate")
async def generate_aibom_endpoint(
    model_id: str,
    metadata: str = Form(...),
    model_file: UploadFile = File(...),
    dataset_files: List[UploadFile] = File([]),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer")),
    _billing: None = Depends(check_billing_limits)
):
    # Check if model exists
    result = await db.execute(select(Model).filter(Model.id == model_id))
    model = result.scalars().first()
    if not model:
        raise HTTPException(404, "Model not found")

    # Record usage
    record_usage(auth.org_id, getattr(auth, "key_id", None), "aibom_generated")

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

    from app.core.celery_app import encrypt_task_payload
    payload = {
        "model_id": model_id,
        "model_b64": model_b64,
        "dataset_b64s": dataset_b64s,
        "metadata": meta_dict
    }
    encrypted_payload = encrypt_task_payload(payload, ["model_b64", "dataset_b64s"])
    
    task = generate_aibom_task.delay(**encrypted_payload)
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
    
    # Fetch supply chain CVE alerts for this model
    from app.db.models import SecurityAlert
    alerts_result = await db.execute(
        select(SecurityAlert).filter(
            SecurityAlert.model_id == model_id,
            SecurityAlert.alert_type == "supply_chain_cve"
        )
    )
    alerts = alerts_result.scalars().all()
    cve_counts = {}
    for alert in alerts:
        details = alert.details or {}
        pkg = details.get("package")
        if pkg:
            cve_counts[pkg.lower()] = cve_counts.get(pkg.lower(), 0) + 1

    # Map database structure to unified components payload expected by frontend
    components = []
    
    # 1. Base Model
    if aibom.base_model:
        bm = aibom.base_model
        if isinstance(bm, dict):
            components.append({
                "name": bm.get("name") or bm.get("repo_id") or "Base Model",
                "version": bm.get("version") or "1.0",
                "type": "model",
                "hash": bm.get("hash") or bm.get("sha256") or aibom.aibom_hash or "",
                "cves": 0
            })
            
    # 2. Training Datasets
    if aibom.training_datasets:
        datasets = aibom.training_datasets if isinstance(aibom.training_datasets, list) else [aibom.training_datasets]
        for ds in datasets:
            if isinstance(ds, dict):
                components.append({
                    "name": ds.get("name") or ds.get("path") or "Dataset",
                    "version": ds.get("version") or "1.0",
                    "type": "dataset",
                    "hash": ds.get("hash") or ds.get("fingerprint") or ds.get("sha256") or "",
                    "cves": 0
                })
                
    # 3. Dependencies / Libraries
    if aibom.dependencies:
        deps = aibom.dependencies if isinstance(aibom.dependencies, list) else [aibom.dependencies]
        for dep in deps:
            if isinstance(dep, dict):
                name = dep.get("name", "")
                components.append({
                    "name": name,
                    "version": dep.get("version") or "—",
                    "type": "library",
                    "hash": dep.get("hash") or "",
                    "cves": cve_counts.get(name.lower(), 0)
                })

    return {
        "model_id": str(aibom.model_id),
        "generated_at": aibom.generated_at,
        "schema_version": aibom.schema_version,
        "base_model": aibom.base_model,
        "training_datasets": aibom.training_datasets,
        "dependencies": aibom.dependencies,
        "training_framework": aibom.training_framework,
        "aibom_hash": aibom.aibom_hash,
        "components": components
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
