from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.session import get_db
from app.db.models import Model, ReportCard
from app.tasks.reports import generate_governance_report
from datetime import datetime
import structlog
from app.billing.metering import record_usage
from app.billing.enforcement import check_billing_limits
from app.core.auth import AuthContext, require_role

# Optional: slowapi for rate limiting (needs installation)
# from slowapi import Limiter
# from slowapi.util import get_remote_address

router = APIRouter()
logger = structlog.get_logger()
# limiter = Limiter(key_func=get_remote_address)

@router.get("/reports")
async def list_reports(model_id: str = None, db: AsyncSession = Depends(get_db)):
    """List generated report cards."""
    q = select(ReportCard).order_by(ReportCard.issued_at.desc())
    if model_id:
        q = q.filter(ReportCard.model_id == model_id)
    reports = (await db.execute(q)).scalars().all()
    
    return {
        "items": [
            {
                "id": r.cert_hash,
                "model_id": str(r.model_id),
                "report_type": "governance",
                "created_at": r.issued_at.isoformat(),
                "file_url": r.pdf_path
            }
            for r in reports
        ]
    }

@router.post("/reports/{model_id}/pdf")
async def start_report_generation(
    model_id: str, 
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer")),
    _billing: None = Depends(check_billing_limits)
):
    """Async trigger for governance report card synthesis."""
    model = await db.get(Model, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
        
    # Record usage
    record_usage(auth.org_id, getattr(auth, "key_id", None), "governance_report_generated")
    
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
        
    model = await db.get(Model, report.model_id)
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
    reports = (await db.execute(select(ReportCard)
        .filter(ReportCard.model_id == model_id)
        .order_by(ReportCard.issued_at.desc()))).scalars().all()
        
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
    """Serve the stored PDF report — streams directly from MinIO if stored there."""
    from fastapi.responses import FileResponse, StreamingResponse
    import os, io

    report = (await db.execute(select(ReportCard).filter(ReportCard.cert_hash == cert_hash))).scalars().first()
    if not report:
        raise HTTPException(status_code=404, detail="Certificate not found")

    filename = f"GovernanceReport_{cert_hash[:12]}.pdf"

    # 1. Serve local file if it exists
    if report.pdf_path and os.path.isfile(report.pdf_path):
        return FileResponse(
            report.pdf_path,
            media_type="application/pdf",
            filename=filename,
        )

    # 2. Stream from MinIO / S3-compatible storage
    if report.pdf_path:
        try:
            import boto3
            s3 = boto3.client(
                "s3",
                endpoint_url=os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
                aws_access_key_id=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
                aws_secret_access_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
                region_name="us-east-1",
            )
            bucket = os.getenv("MINIO_BUCKET", "mlguard")
            obj = s3.get_object(Bucket=bucket, Key=report.pdf_path)
            pdf_bytes = obj["Body"].read()
            return StreamingResponse(
                io.BytesIO(pdf_bytes),
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
        except Exception as exc:
            logger.warning("MinIO download failed", cert_hash=cert_hash, error=str(exc))
            raise HTTPException(status_code=503, detail=f"Could not retrieve PDF from storage: {exc}")

    raise HTTPException(status_code=404, detail="No PDF available for this report yet.")

try:
    from ml_guard.core.compliance import evaluate_compliance
except ImportError:
    def evaluate_compliance(metrics: dict) -> list:
        """Fallback: returns empty compliance results if ml_guard.core is not installed."""
        return []

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
