from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.db.models import Job, GovernanceResult

router = APIRouter()

@router.get("/governance/health")
async def governance_health():
    """Health check for the Enterprise Governance Synthesis module."""
    return {"status": "governance router active", "version": "7.2.0"}

@router.post("/governance/evaluate")
async def evaluate_governance(model_id: str, db: Session = Depends(get_db)):
    """Placeholder for weighted governance scoring as described in README."""
    return {"message": "Full governance audit synthesis triggered (v7.2 placeholder)", "model_id": model_id, "status": "PENDING"}

@router.get("/governance/{job_id}")
async def get_governance_results(job_id: str, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    result = db.query(GovernanceResult).filter(GovernanceResult.job_id == job_id).first()
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
