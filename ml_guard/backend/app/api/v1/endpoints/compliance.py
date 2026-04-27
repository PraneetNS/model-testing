from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import structlog
from app.api.v1 import deps
from app.infrastructure.persistence import models as sql_models
from app.domain.services.compliance import ComplianceService
from app.domain.models.test_suite import QualityGateResult

logger = structlog.get_logger(__name__)
from app.billing.metering import record_usage
from app.billing.enforcement import check_billing_limits
from app.core.auth import AuthContext, require_role
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

import hashlib
from fastapi.responses import Response

@router.get("/{model_id}/pack/{pack_name}")
async def get_compliance_pack(
    model_id: str, 
    pack_name: str, 
    db: AsyncSession = Depends(deps.get_db),
    auth: AuthContext = Depends(require_role("viewer")),
    _billing: None = Depends(check_billing_limits)
):
    # Record usage
    record_usage(auth.org_id, getattr(auth, "key_id", None), "compliance_pack_run")
    checks = []
    import sys
    import os
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../"))
    if _repo_root not in sys.path:
        sys.path.append(_repo_root)
        
    if pack_name == "sr_11_7":
        from ml_guard.core.compliance_packs.sr_11_7 import generate_sr117_report
        checks = await generate_sr117_report(model_id, db)
    elif pack_name == "eu_ai_act":
        from ml_guard.core.compliance_packs.eu_ai_act import generate_eu_ai_act_report
        checks = await generate_eu_ai_act_report(model_id, db)
    elif pack_name == "rbi_mlrg":
        from ml_guard.core.compliance_packs.rbi_mlrg import generate_rbi_mlrg_report
        checks = await generate_rbi_mlrg_report(model_id, db)
    elif pack_name == "fda_ai":
        from ml_guard.core.compliance_packs.fda_ai_guidance import generate_fda_ai_guidance_report
        checks = await generate_fda_ai_guidance_report(model_id, db)
    else:
        raise HTTPException(status_code=404, detail="Pack not found")
        
    passed_checks = sum(1 for c in checks if c["status"] == "pass")
    total = len(checks)
    score = (passed_checks / total * 100) if total > 0 else 0
    status = "compliant" if score == 100 else ("partial" if score > 0 else "non_compliant")
    
    return {"status": status, "score": score, "checks": checks}

@router.get("/{model_id}/pack/{pack_name}/pdf")
async def get_compliance_pack_pdf(
    model_id: str, 
    pack_name: str, 
    db: AsyncSession = Depends(deps.get_db),
    auth: AuthContext = Depends(require_role("ml_engineer")),
    _billing: None = Depends(check_billing_limits)
):
    # Record usage for certificate issuance
    record_usage(auth.org_id, getattr(auth, "key_id", None), "compliance_certificate_issued")
    pack_data = await get_compliance_pack(model_id, pack_name, db, auth, _billing)
    checks = pack_data["checks"]
    score = pack_data["score"]
    
    import tempfile
    from app.services.report_card.pdf import PDFGenerator
    from app.db.models import Model
    import datetime
    import os
    import json
    
    model = await db.get(Model, model_id)
    model_name = model.name if model else model_id
    
    metric_snapshots = {c["title"]: (100 if c["status"] == "pass" else 0) for c in checks}
    
    report_data = {
        "model_name": model_name,
        "overall_score": score,
        "verdict": "COMPLIANT" if pack_data["status"] == "compliant" else "NON-COMPLIANT",
        "issued_at": datetime.datetime.utcnow().isoformat(),
        "cert_hash": "CompliancePackPDF",
        "metric_snapshots": metric_snapshots,
        "executive_summary": f"Compliance Pack: {pack_name.upper()}. Status: {pack_data['status'].upper()}.",
        "include_compliance": False
    }
    
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
        tmp_path = tmp_pdf.name
        
    pdf_gen = PDFGenerator(tmp_path)
    pdf_gen.generate(report_data)
    
    with open(tmp_path, "rb") as f:
        pdf_bytes = f.read()
    
    os.remove(tmp_path)
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()
    
    headers = {
        "Content-Disposition": f"attachment; filename=compliance_{pack_name}_{model_id}.pdf",
        "X-Report-Hash": pdf_hash
    }
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)

@router.get("/packs/available")
async def get_available_packs():
    return [
        {"name": "sr_11_7", "framework": "SR 11-7", "jurisdiction": "US", "version": "1.0", "check_count": 4},
        {"name": "eu_ai_act", "framework": "EU AI Act", "jurisdiction": "EU", "version": "1.0", "check_count": 5},
        {"name": "rbi_mlrg", "framework": "RBI MLRG", "jurisdiction": "India", "version": "1.0", "check_count": 4},
        {"name": "fda_ai", "framework": "FDA AI Guidance", "jurisdiction": "US", "version": "1.0", "check_count": 3}
    ]

