from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.db.models import Job, DriftResult

router = APIRouter()

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
