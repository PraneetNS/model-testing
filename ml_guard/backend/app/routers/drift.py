from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.db.models import Job, DriftResult

router = APIRouter()

@router.get("/drift/health")
async def drift_health():
    """Health check endpoint for statistical stability (PSI Drift) module."""
    return {"status": "drift router active", "version": "7.2.0"}

@router.post("/drift/evaluate")
async def evaluate_drift(model_id: str, baseline_id: str, current_id: str, db: Session = Depends(get_db)):
    """Placeholder for PSI Drift evaluation as described in README."""
    return {"message": "Drift evaluation triggered (v7.2 placeholder)", "model_id": model_id, "status": "PENDING"}

@router.get("/drift/{job_id}")
async def get_drift_results(job_id: str, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    result = db.query(DriftResult).filter(DriftResult.job_id == job_id).first()
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
