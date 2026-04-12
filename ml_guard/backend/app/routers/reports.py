from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.session import get_db
from app.db.models import Model, ReportCard
from app.tasks.reports import generate_governance_report
from datetime import datetime
import structlog

# Optional: slowapi for rate limiting (needs installation)
# from slowapi import Limiter
# from slowapi.util import get_remote_address

router = APIRouter()
logger = structlog.get_logger()
# limiter = Limiter(key_func=get_remote_address)

@router.post("/{model_id}/generate")
async def start_report_generation(model_id: str, db: AsyncSession = Depends(get_db)):
    """Async trigger for governance report card synthesis."""
    model = db.query(Model).get(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
        
    task = generate_governance_report.delay(model_id)
    return {"task_id": task.id, "status": "PENDING", "estimated_seconds": 20}

@router.get("/status/{task_id}")
async def get_report_status(task_id: str):
    """Polling endpoint for report generation progress via Celery backend."""
    result = generate_governance_report.AsyncResult(task_id)
    response = {
        "task_id": task_id,
        "status": result.status,
    }
    if result.ready():
        res_data = result.get()
        response.update(res_data)
        
    return response

@router.get("/verify/{cert_hash}")
async def verify_certificate(request: Request, cert_hash: str, db: AsyncSession = Depends(get_db)):
    """PUBLIC verification endpoint for external auditors - integrated with gate.py logic."""
    report = (await db.execute(select(ReportCard).filter(ReportCard.cert_hash == cert_hash))).scalars().first()
    if not report:
        return {"valid": False, "message": "Certificate not found."}
        
    model = db.query(Model).get(report.model_id)
    return {
        "valid": not report.is_revoked,
        "model_name": model.name if model else "Unknown",
        "issued_at": report.issued_at.isoformat(),
        "overall_score": report.overall_score,
        "verdict": report.verdict,
        "revoked": report.is_revoked,
        "revocation_reason": report.revocation_reason if report.is_revoked else None
    }

@router.get("/{model_id}/history")
async def get_report_history(model_id: str, db: AsyncSession = Depends(get_db)):
    """Timeline of all generated certificates for a specific model."""
    reports = db.query(ReportCard)\
        .filter(ReportCard.model_id == model_id)\
        .order_by(ReportCard.issued_at.desc())\
        .all()
        
    return [{
        "cert_hash": r.cert_hash,
        "issued_at": r.issued_at.isoformat(),
        "overall_score": r.overall_score,
        "verdict": r.verdict,
        "is_revoked": r.is_revoked
    } for r in reports]

@router.post("/revoke/{cert_hash}")
async def revoke_certificate(cert_hash: str, reason: str = "Model decommissioned", db: AsyncSession = Depends(get_db)):
    """Administrative revocation of professional AI governance certificates."""
    report = (await db.execute(select(ReportCard).filter(ReportCard.cert_hash == cert_hash))).scalars().first()
    if not report:
        raise HTTPException(status_code=404, detail="Certificate not found")
        
    report.is_revoked = True
    report.revocation_reason = reason
    report.revoked_at = datetime.utcnow()
    await db.commit()
    
    return {"message": "Certificate revoked successfully.", "cert_hash": cert_hash}

@router.get("/download/{cert_hash}")
async def download_report_pdf(cert_hash: str, db: AsyncSession = Depends(get_db)):
    """Secure PDF artifact retrieval from cloud object storage via signed URI (Simulated)."""
    report = (await db.execute(select(ReportCard).filter(ReportCard.cert_hash == cert_hash))).scalars().first()
    if not report:
        raise HTTPException(status_code=404, detail="Certificate not found")
        
    return {"download_url": f"https://minio.mlguard.io/{report.pdf_path}", "cert_hash": cert_hash}

import sys
import os
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if _repo_root not in sys.path:
    sys.path.append(_repo_root)
from ml_guard.core.compliance import evaluate_compliance

@router.get("/{model_id}/compliance")
async def get_compliance_report(model_id: str, framework: str = "all", db: AsyncSession = Depends(get_db)):
    report = (await db.execute(select(ReportCard).filter(ReportCard.model_id == model_id).order_by(ReportCard.issued_at.desc()).limit(1))).scalars().first()
    
    if not report:
        # Fallback to an empty dict if no report card, just to return the schema
        metrics = {}
    else:
        metrics = report.metric_snapshots
        
    all_results = evaluate_compliance(metrics)
    
    if framework != "all":
        all_results = [r for r in all_results if r["framework"] == framework.lower()]
        
    return {"results": all_results}
