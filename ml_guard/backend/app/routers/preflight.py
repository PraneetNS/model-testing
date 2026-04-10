from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.session import get_db
from app.db.models import Job, PreflightResult

router = APIRouter()

@router.get("/preflight/health")
async def preflight_health():
    """Health check for Data Quality & Preflight audit module."""
    return {"status": "preflight router active", "version": "7.2.0"}

@router.post("/preflight/evaluate")
async def run_preflight(model_id: str, db: AsyncSession = Depends(get_db)):
    """Placeholder for data quality scans as described in README."""
    return {"message": "Data Quality Preflight triggered (v7.2 placeholder)", "model_id": model_id, "status": "PENDING"}

@router.get("/preflight/{job_id}")
async def get_preflight_results(job_id: str, db: AsyncSession = Depends(get_db)):
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    result = (await db.execute(select(PreflightResult).filter(PreflightResult.job_id == job_id))).scalars().first()
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
