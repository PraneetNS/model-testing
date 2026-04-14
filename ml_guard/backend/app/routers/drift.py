from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.session import get_db
from app.db.models import Job, DriftResult

router = APIRouter()

@router.get("/drift/health")
async def drift_health():
    """Health check endpoint for statistical stability (PSI Drift) module."""
    return {"status": "drift router active", "version": "7.2.0"}

@router.post("/drift/evaluate")
async def evaluate_drift(model_id: str, baseline_id: str, current_id: str, db: AsyncSession = Depends(get_db)):
    """Placeholder for PSI Drift evaluation as described in README."""
    return {"message": "Drift evaluation triggered (v7.2 placeholder)", "model_id": model_id, "status": "PENDING"}

@router.get("/drift/{job_id}")
async def get_drift_results(job_id: str, db: AsyncSession = Depends(get_db)):
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    result = (await db.execute(select(DriftResult).filter(DriftResult.job_id == job_id))).scalars().first()
    if not result:
        return {"status": job.status, "error": job.error, "result": None}

    return {
        "status": job.status,
        "result": {
            "metrics": result.computed_metrics_json,
            "severity_counts": result.severity_counts,
            "module_status": result.status
        }
    }

import sys
import os
import uuid
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if _repo_root not in sys.path:
    sys.path.append(_repo_root)

# Import our new ML Guard core embedding module 
from ml_guard.core.drift import compute_embedding_drift
from app.db.models import EmbeddingBatch

class EmbeddingIngestRequest(BaseModel):
    embeddings: List[List[float]]
    batch_id: Optional[str] = None
    timestamp: Optional[datetime] = None

@router.post("/drift/{model_id}/embedding-ingest")
async def ingest_embedding_drift(model_id: str, payload: EmbeddingIngestRequest, db: AsyncSession = Depends(get_db)):
    batch_id = payload.batch_id or str(uuid.uuid4())
    record = EmbeddingBatch(
        model_id=model_id,
        batch_id=batch_id,
        embeddings=payload.embeddings,
        timestamp=payload.timestamp or datetime.utcnow()
    )
    db.add(record)
    await db.commit()
    return {"status": "success", "batch_id": batch_id, "size": len(payload.embeddings)}

@router.get("/drift/{model_id}/embedding-report")
async def get_embedding_report(model_id: str, n_batches: int = 5, db: AsyncSession = Depends(get_db)):
    # 1. Fetch reference baseline (for simplicity, first recorded batch for this model)
    ref_batch = (await db.execute(select(EmbeddingBatch).filter(EmbeddingBatch.model_id == model_id).order_by(EmbeddingBatch.timestamp.asc()).limit(1))).scalars().first()
    
    if not ref_batch:
        raise HTTPException(status_code=404, detail="No baseline embeddings found for model")
        
    # 2. Fetch last N batches
    current_batches = (await db.execute(select(EmbeddingBatch).filter(EmbeddingBatch.model_id == model_id, EmbeddingBatch.id != ref_batch.id).order_by(EmbeddingBatch.timestamp.desc()).limit(n_batches))).scalars().all()
    
    if not current_batches:
        raise HTTPException(status_code=404, detail="No current embeddings found for model to compare against baseline")
    
    current_embeddings = []
    for batch in current_batches:
        current_embeddings.extend(batch.embeddings)
        
    report = compute_embedding_drift(ref_batch.embeddings, current_embeddings)
    return report
