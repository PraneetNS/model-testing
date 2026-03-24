from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.db.models import Job, ScanRecord

router = APIRouter()

@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str, db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    response = {
        "job_id": job_id,
        "status": job.status,
        "error": job.error,
        "created_at": str(job.created_at)
    }

    if job.status == "COMPLETED":
        # Check for related ScanRecord
        scan = db.query(ScanRecord).filter(ScanRecord.job_id == job_id).first()
        if not scan:
             # Fallback: check by trigger context or most recent
             scan = db.query(ScanRecord).order_by(ScanRecord.created_at.desc()).first()
        
        if scan:
            response["results"] = scan.results_json
            response["scan_id"] = str(scan.id)
            response["governance_score"] = scan.governance_score
            response["gate_status"] = scan.gate_status
            response["risk_score"] = scan.risk_score
            response["risk_level"] = scan.risk_level
            # Map top-level results for UI compatibility
            if scan.results_json:
                response.update(scan.results_json)

    return response
