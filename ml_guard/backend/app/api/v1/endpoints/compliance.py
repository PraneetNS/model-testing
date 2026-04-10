from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import structlog
from app.api.v1 import deps
from app.infrastructure.persistence import models as sql_models
from app.domain.services.compliance import ComplianceService
from app.domain.models.test_suite import QualityGateResult

logger = structlog.get_logger(__name__)
router = APIRouter()
compliance_service = ComplianceService()

@router.get("/report/{run_id}")
async def get_compliance_report(
    run_id: str,
    db: AsyncSession = Depends(deps.get_db),
    current_user: sql_models.User = Depends(deps.get_current_active_user)
):
    """
    Generate a full compliance audit report for a specific test run.
    """
    run = (await db.execute(select(sql_models.TestRun).filter(sql_models.TestRun.id == run_id))).scalars().first()
    if not run:
        raise HTTPException(status_code=404, detail="Test run not found")

    # Reconstruct QualityGateResult from DB for the compliance engine
    # In a full impl, we'd have a mapper, but here we rebuild from the JSON blobs
    result = QualityGateResult(
        run_id=str(run.id),
        project_id=str(run.project_id),
        model_version=run.model_version,
        test_suite=run.suite_name,
        score=run.score,
        deployment_allowed=run.deployment_allowed,
        results=run.results_raw,
        risk_level=run.summary_metrics.get("risk_level", "Unknown"),
        reproducibility_token=run.summary_metrics.get("reproducibility_token"),
        environment_config=run.summary_metrics.get("environment_config", {}),
        execution_metadata=run.summary_metrics.get("execution_metadata", {}),
        model_profile=run.summary_metrics.get("model_profile", {}),
        feature_importance=run.summary_metrics.get("feature_importance", [])
    )

    report = compliance_service.generate_audit_report(result)
    return report

@router.get("/export/pdf/{run_id}")
async def export_audit_pdf(run_id: str):
    """
    Placeholder for PDF generation.
    In production, this would use a library like ReportLab or WeasyPrint.
    """
    return {"message": "PDF Export initialized. Template: Enterprise Audit v1.0"}
