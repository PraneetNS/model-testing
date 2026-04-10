import os
import tempfile
import asyncio
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.session import SessionLocal
from app.db.models import Model
from app.models.report_card import ReportCard
from app.services.report_card.builder import ReportCardBuilder
from app.services.report_card.llm_summary import ExecutiveSummaryGenerator
from app.services.report_card.pdf import PDFGenerator
from app.core.celery_app import celery_app
import structlog

logger = structlog.get_logger()

# Mock MinIO helper for the task
async def upload_to_minio(file_path: str, destination: str):
    logger.info("Uploading report to MinIO", destination=destination)
    await asyncio.sleep(1) # Simulate upload
    return f"minio://{destination}"

@celery_app.task(name="app.services.report_card.generate_governance_report", bind=True, max_retries=3, default_retry_delay=10)
async def generate_governance_report(model_id: str):
    """
    Async task to synthesize audit data, generate LLM summary, 
    render PDF, and persist to storage and database.
    """
    db = SessionLocal()
    try:
        # 1. Aggregrate & Score
        builder = ReportCardBuilder(db, model_id)
        audit_data = builder.aggregate_audit_data()
        if not audit_data:
            logger.error("No audit data found to generate report", model_id=model_id)
            return {"status": "FAILED", "error": "No audit data found."}

        score, verdict = builder.compute_governance_score(audit_data)
        cert_hash = builder.generate_cert_hash(model_id, audit_data['audit_timestamp'], score)

        # 2. Check Collision 
        existing = (await db.execute(select(ReportCard).filter(ReportCard.cert_hash == cert_hash))).scalars().first()
        if existing:
            logger.info("Report already exists for this audit snapshot", cert_hash=cert_hash)
            return {"status": "SUCCESS", "cert_hash": cert_hash}

        # 3. Generate Executive Summary (Async)
        loop = asyncio.get_event_loop()
        llm_gen = ExecutiveSummaryGenerator()
        summary = loop.run_until_complete(llm_gen.generate_summary(audit_data))

        # 4. Render PDF
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
            tmp_path = tmp_pdf.name
        
        report_data = {
            "model_name": builder.model.name,
            "overall_score": score,
            "verdict": verdict,
            "issued_at": datetime.utcnow().isoformat(),
            "cert_hash": cert_hash,
            "metric_snapshots": audit_data,
            "executive_summary": summary
        }
        
        pdf_gen = PDFGenerator(tmp_path)
        pdf_gen.generate(report_data)

        # 5. Upload to MinIO
        minio_dest = f"reports/{model_id}/{cert_hash}.pdf"
        s3_url = loop.run_until_complete(upload_to_minio(tmp_path, minio_dest))

        # 6. Persist to DB
        new_report = ReportCard(
            model_id=model_id,
            cert_hash=cert_hash,
            overall_score=score,
            verdict=verdict,
            executive_summary=summary,
            metric_snapshots=audit_data,
            pdf_path=minio_dest
        )
        db.add(new_report)
        await db.commit()

        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        return {"status": "SUCCESS", "cert_hash": cert_hash}

    except Exception as e:
        logger.error("Governance report generation failed", model_id=model_id, error=str(e))
        db.rollback()
        return {"status": "FAILED", "error": str(e)}
    finally:
        db.close()
