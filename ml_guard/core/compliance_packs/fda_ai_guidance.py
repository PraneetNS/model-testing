import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.models import Model, DriftReport, ExplainabilityResult, ReportCard

async def check_predetermined_change_control(model_id: str, db: AsyncSession):
    model = await db.get(Model, model_id)
    if not model:
        return {"passed": False, "reason": "Model not found"}
        
    passed = model.parent_model_id is not None or model.version == 1
    return {"passed": passed}

async def check_real_world_performance_monitoring(model_id: str, db: AsyncSession):
    model = await db.get(Model, model_id)
    if not model:
        return {"passed": False, "reason": "Model not found"}
        
    thirty_days_ago = datetime.datetime.utcnow() - datetime.timedelta(days=30)
    drifts = (await db.execute(select(DriftReport).filter(DriftReport.model_id == model_id))).scalars().all()
    
    active_30_days = model.created_at <= thirty_days_ago
    passed = len(drifts) > 0 and active_30_days
    return {"passed": passed}

async def check_transparency_disclosure(model_id: str, db: AsyncSession):
    shaps = (await db.execute(select(ExplainabilityResult).filter(ExplainabilityResult.model_id == model_id))).scalars().all()
    reports = (await db.execute(select(ReportCard).filter(ReportCard.model_id == model_id))).scalars().all()
    
    passed = len(shaps) > 0 and len(reports) > 0
    return {"passed": passed}

async def generate_fda_ai_guidance_report(model_id: str, db: AsyncSession):
    r1 = await check_predetermined_change_control(model_id, db)
    r2 = await check_real_world_performance_monitoring(model_id, db)
    r3 = await check_transparency_disclosure(model_id, db)
    
    return [
        {"article": "FDA AI Guidance", "title": "Predetermined Change Control Plan", "status": "pass" if r1["passed"] else "fail", "evidence": "Lineage or v1.0 check", "remediation": "Establish model lineage"},
        {"article": "FDA AI Guidance", "title": "Real-World Performance Monitoring", "status": "pass" if r2["passed"] else "fail", "evidence": "Drift checks + active >= 30 days", "remediation": "Ensure model is active and monitored"},
        {"article": "FDA AI Guidance", "title": "Transparency and Disclosure", "status": "pass" if r3["passed"] else "fail", "evidence": "SHAP + Governance Report exists", "remediation": "Generate SHAP explanation and governance report"}
    ]
