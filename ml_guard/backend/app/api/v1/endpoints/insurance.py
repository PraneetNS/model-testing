import uuid
import datetime
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.session import get_db
from app.db.models import Model, SecurityAlert, ScanRecord
from ml_guard.core.insurance_score import compute_insurance_score, INSURANCE_TIERS

router = APIRouter()

def get_tier_rank(tier_name: str) -> int:
    for i, t in enumerate(reversed(INSURANCE_TIERS)):
        if t["name"] == tier_name:
            return i
    return -1

@router.get("/governance/{model_id}/insurance-score")
async def get_insurance_score(
    model_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """Returns the full actuarial InsuranceReport for an AI model."""
    try:
        return await compute_insurance_score(model_id, db)
    except ValueError as e:
        raise HTTPException(404, str(e))

@router.post("/governance/{model_id}/insurance-score/refresh")
async def refresh_insurance_score(
    model_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """Forces a recomputation and checks for insurance tier downgrades."""
    try:
        # 1. Look for previous tier in metadata or last scan result
        result = await db.execute(
            select(ScanRecord.results_json)
            .filter(ScanRecord.model_id == model_id)
            .order_by(ScanRecord.created_at.desc())
            .limit(1)
        )
        prev_scan = result.scalar() or {}
        old_tier = prev_scan.get("insurance_tier", "standard")
        
        # 2. Compute new score
        report = await compute_insurance_score(model_id, db)
        new_tier = report["tier"]
        
        # 3. Detect Downgrade
        if get_tier_rank(new_tier) < get_tier_rank(old_tier):
            # Create Alert
            alert = SecurityAlert(
                model_id=str(model_id),
                alert_type="insurance_tier_downgrade",
                severity="HIGH",
                details={
                    "old_tier": old_tier,
                    "new_tier": new_tier,
                    "score": report["total_score"],
                    "message": f"AI Liability Insurance Tier dropped from {old_tier} to {new_tier}. Premium rates may increase."
                }
            )
            db.add(alert)
            await db.commit()
            
        return {"status": "refreshed", "report": report}
    except ValueError as e:
        raise HTTPException(404, str(e))

@router.get("/governance/{model_id}/insurance-score/pdf")
async def get_insurance_pdf(
    model_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    """Returns a professionally formatted PDF actuarial report."""
    import tempfile
    import os
    from app.services.report_card.pdf import PDFGenerator
    from app.services.report_card.builder import ReportCardBuilder
    
    try:
        # 1. Compute Score
        report = await compute_insurance_score(model_id, db)
        
        # 2. Get Model Info
        model = await db.get(Model, model_id)
        
        # 3. Aggregate some audit data for the breakdown pages (optional but better)
        builder = ReportCardBuilder(db, str(model_id))
        await builder.initialize()
        audit_data = await builder.aggregate_audit_data() or {}
        
        # 4. Prepare data for PDFGenerator
        report_data = {
            "model_name": model.name if model else "Unknown",
            "overall_score": audit_data.get("governance_score", 0),
            "verdict": "CERTIFIED" if audit_data.get("governance_score", 0) > 80 else "CONDITIONAL",
            "issued_at": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "cert_hash": str(uuid.uuid4()).replace("-", ""),
            "metric_snapshots": audit_data,
            "executive_summary": "Actuarial assessment of AI model risk based on 6 core dimensions of liability.",
            "insurance_report": report
        }
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            pdf_gen = PDFGenerator(tmp_path)
            pdf_gen.generate(report_data)
            
            with open(tmp_path, "rb") as f:
                content = f.read()
                
            return Response(
                content=content,
                media_type="application/pdf"
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except Exception as e:
        import structlog
        structlog.get_logger().error("pdf_generation_failed", error=str(e))
        raise HTTPException(500, f"Failed to generate insurance PDF: {str(e)}")
