import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.models import ReportCard, ScanRecord, ExplainabilityResult, PerformanceSnapshot, DriftReport, Model

async def check_model_validation_documentation(model_id: str, db: AsyncSession):
    reports = (await db.execute(select(ReportCard).filter(ReportCard.model_id == model_id))).scalars().all()
    audits = (await db.execute(select(ScanRecord).filter(ScanRecord.model_id == model_id))).scalars().all()
    shaps = (await db.execute(select(ExplainabilityResult).filter(ExplainabilityResult.model_id == model_id))).scalars().all()
    
    missing = []
    if not reports: missing.append("governance report")
    if not audits: missing.append("audit entry")
    if not shaps: missing.append("SHAP explanation run")
    
    return {"passed": len(missing) == 0, "missing": missing}

async def check_performance_benchmarking(model_id: str, db: AsyncSession):
    snapshots = (await db.execute(select(PerformanceSnapshot).filter(PerformanceSnapshot.model_id == model_id).order_by(PerformanceSnapshot.computed_at.asc()))).scalars().all()
    if len(snapshots) < 2:
        return {"passed": False, "reason": "Requires baseline and current snapshots"}
    
    baseline = snapshots[0].metrics.get("accuracy", 1.0) if snapshots[0].metrics else 1.0
    current = snapshots[-1].metrics.get("accuracy", 1.0) if snapshots[-1].metrics else 1.0
    degradation = (baseline - current) / max(baseline, 0.0001)
    passed = degradation <= 0.15
    return {"passed": passed, "degradation": degradation}

async def check_independent_validation(model_id: str, db: AsyncSession):
    model = await db.get(Model, model_id)
    if not model:
        return {"passed": False, "reason": "Model not found"}
    
    scan = (await db.execute(select(ScanRecord).filter(ScanRecord.model_id == model_id).order_by(ScanRecord.created_at.desc()))).scalars().first()
    if not scan:
        return {"passed": False, "reason": "No scan records found"}
        
    passed = str(model.created_by) != str(scan.triggered_by)
    return {"passed": passed, "validator_key_id": str(scan.triggered_by)}

async def check_ongoing_monitoring(model_id: str, db: AsyncSession):
    thirty_days_ago = datetime.datetime.utcnow() - datetime.timedelta(days=30)
    drifts = (await db.execute(select(DriftReport).filter(DriftReport.model_id == model_id, DriftReport.created_at >= thirty_days_ago).order_by(DriftReport.created_at.desc()))).scalars().all()
    passed = len(drifts) > 0
    return {"passed": passed, "last_drift_check": drifts[0].created_at.isoformat() if drifts else None}

async def generate_sr117_report(model_id: str, db: AsyncSession):
    r1 = await check_model_validation_documentation(model_id, db)
    r2 = await check_performance_benchmarking(model_id, db)
    r3 = await check_independent_validation(model_id, db)
    r4 = await check_ongoing_monitoring(model_id, db)
    
    return [
        {"article": "SR 11-7", "title": "Model Validation Documentation", "status": "pass" if r1["passed"] else "fail", "evidence": f"Missing: {r1.get('missing', [])}", "remediation": "Generate governance report and SHAP explanations"},
        {"article": "SR 11-7", "title": "Performance Benchmarking", "status": "pass" if r2["passed"] else "fail", "evidence": f"Degradation: {r2.get('degradation', 'N/A')}", "remediation": "Retrain model to reduce performance degradation"},
        {"article": "SR 11-7", "title": "Independent Validation", "status": "pass" if r3["passed"] else "fail", "evidence": f"Validator: {r3.get('validator_key_id', 'None')}", "remediation": "Have a different user/API key trigger the audit"},
        {"article": "SR 11-7", "title": "Ongoing Monitoring", "status": "pass" if r4["passed"] else "fail", "evidence": f"Last drift check: {r4.get('last_drift_check', 'None')}", "remediation": "Run drift detection"}
    ]
