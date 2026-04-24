import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.models import ReportCard, ScanRecord, ExplainabilityResult, PerformanceSnapshot, DriftReport, Model, RedTeamSession, Dataset

async def check_article_9_risk_management(model_id: str, db: AsyncSession):
    scan = (await db.execute(select(ScanRecord).filter(ScanRecord.model_id == model_id).order_by(ScanRecord.created_at.desc()))).scalars().first()
    risk_tier = scan.risk_level if scan and scan.risk_level else "UNKNOWN"
    
    passed = True
    remediation = []
    if risk_tier == "UNKNOWN":
        passed = False
        remediation.append("Run audit to establish risk_tier")
        
    if risk_tier in ["HIGH", "CRITICAL"]:
        red_teams = (await db.execute(select(RedTeamSession).filter(RedTeamSession.model_id == model_id))).scalars().all()
        if not red_teams:
            passed = False
            remediation.append("High-risk models require >= 1 red team session")
            
    return {"passed": passed, "risk_tier": risk_tier, "remediation": remediation}

async def check_article_10_data_governance(model_id: str, db: AsyncSession):
    datasets = (await db.execute(select(Dataset).filter(Dataset.model_id == model_id))).scalars().all()
    has_aibom = False
    for ds in datasets:
        if ds.fingerprint:
            has_aibom = True
            break
            
    return {"passed": has_aibom}

async def check_article_13_transparency(model_id: str, db: AsyncSession):
    shaps = (await db.execute(select(ExplainabilityResult).filter(ExplainabilityResult.model_id == model_id))).scalars().all()
    reports = (await db.execute(select(ReportCard).filter(ReportCard.model_id == model_id))).scalars().all()
    
    passed = bool(shaps and reports)
    return {"passed": passed, "report_url": f"/api/compliance/report/{reports[0].id}" if reports else None}

async def check_article_15_accuracy_robustness(model_id: str, db: AsyncSession):
    scan = (await db.execute(select(ScanRecord).filter(ScanRecord.model_id == model_id).order_by(ScanRecord.created_at.desc()))).scalars().first()
    perf_score = scan.governance_score if scan and scan.governance_score is not None else 0
    
    red_teams = (await db.execute(select(RedTeamSession).filter(RedTeamSession.model_id == model_id).order_by(RedTeamSession.created_at.desc()))).scalars().first()
    robustness_score = 100 if not red_teams else (red_teams.success_count / max(red_teams.total_attacks, 1) * 100) # simpler mock
    robustness_score = scan.results_json.get("robustness_score", robustness_score) if scan and scan.results_json else robustness_score
    
    passed = perf_score >= 60 and robustness_score >= 50
    return {"passed": passed, "performance_score": perf_score, "robustness_score": robustness_score}

async def check_article_72_post_market(model_id: str, db: AsyncSession):
    seven_days_ago = datetime.datetime.utcnow() - datetime.timedelta(days=7)
    drifts = (await db.execute(select(DriftReport).filter(DriftReport.model_id == model_id, DriftReport.created_at >= seven_days_ago))).scalars().all()
    passed = len(drifts) >= 3
    return {"passed": passed, "drift_checks_count": len(drifts)}

async def generate_eu_ai_act_report(model_id: str, db: AsyncSession):
    r1 = await check_article_9_risk_management(model_id, db)
    r2 = await check_article_10_data_governance(model_id, db)
    r3 = await check_article_13_transparency(model_id, db)
    r4 = await check_article_15_accuracy_robustness(model_id, db)
    r5 = await check_article_72_post_market(model_id, db)
    
    return [
        {"article": "Article 9", "title": "Risk Management System", "status": "pass" if r1["passed"] else "fail", "evidence": f"Risk tier: {r1.get('risk_tier')}", "remediation": ", ".join(r1.get("remediation", []))},
        {"article": "Article 10", "title": "Data Governance", "status": "pass" if r2["passed"] else "fail", "evidence": "AIBOM check", "remediation": "Create dataset entry with fingerprint"},
        {"article": "Article 13", "title": "Transparency", "status": "pass" if r3["passed"] else "fail", "evidence": "Report URL: " + str(r3.get("report_url")), "remediation": "Ensure SHAP and report card exist"},
        {"article": "Article 15", "title": "Accuracy and Robustness", "status": "pass" if r4["passed"] else "fail", "evidence": f"Perf: {r4.get('performance_score')}, Robustness: {r4.get('robustness_score')}", "remediation": "Improve performance and robustness"},
        {"article": "Article 72", "title": "Post-market Monitoring", "status": "pass" if r5["passed"] else "fail", "evidence": f"Checks in last 7 days: {r5.get('drift_checks_count')}", "remediation": "Enable regular drift monitoring"}
    ]
