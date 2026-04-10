from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.session import get_db
from app.db.models import Job, PerformanceResult

router = APIRouter()

@router.get("/performance/health")
async def performance_health():
    """Health check for ML Model Performance evaluation module."""
    return {"status": "performance router active", "version": "7.2.0"}

@router.post("/performance/evaluate")
async def evaluate_performance(model_id: str, dataset_id: str, db: AsyncSession = Depends(get_db)):
    """Placeholder for Accuracy/F1/ROC audits as described in README."""
    return {"message": "Performance evaluation triggered (v7.2 placeholder)", "model_id": model_id, "status": "PENDING"}

@router.get("/performance/{job_id}")
async def get_performance_results(job_id: str, db: AsyncSession = Depends(get_db)):
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    result = (await db.execute(select(PerformanceResult).filter(PerformanceResult.job_id == job_id))).scalars().first()
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
