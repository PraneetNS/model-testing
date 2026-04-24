from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.models import ReportCard, ScanRecord, RedTeamSession, Model

async def check_model_approval_workflow(model_id: str, db: AsyncSession):
    report = (await db.execute(select(ReportCard).filter(ReportCard.model_id == model_id).order_by(ReportCard.issued_at.desc()))).scalars().first()
    verdict = report.verdict if report else None
    passed = verdict in ["CERTIFIED", "CONDITIONAL"]
    return {"passed": passed, "verdict": verdict}

async def check_bias_testing(model_id: str, db: AsyncSession):
    scan = (await db.execute(select(ScanRecord).filter(ScanRecord.model_id == model_id).order_by(ScanRecord.created_at.desc()))).scalars().first()
    fairness_score = scan.fairness_risk_score if scan and scan.fairness_risk_score is not None else (scan.results_json.get("fairness_score", 0) if scan and scan.results_json else 0)
    passed = fairness_score >= 50
    return {"passed": passed, "fairness_score": fairness_score}

async def check_model_documentation(model_id: str, db: AsyncSession):
    report = (await db.execute(select(ReportCard).filter(ReportCard.model_id == model_id).order_by(ReportCard.issued_at.desc()))).scalars().first()
    if not report or not report.metric_snapshots:
        return {"passed": False, "reason": "No report card"}
    
    dimensions = ["drift_score", "overfitting_score", "calibration_score", "robustness_score", "fairness_score"]
    present = [d for d in dimensions if d in report.metric_snapshots]
    passed = len(present) == len(dimensions)
    return {"passed": passed, "dimensions_present": present}

async def check_stress_testing(model_id: str, db: AsyncSession):
    red_teams = (await db.execute(select(RedTeamSession).filter(RedTeamSession.model_id == model_id).order_by(RedTeamSession.created_at.desc()))).scalars().all()
    # Assume any red team session meets standard profile for now
    passed = len(red_teams) > 0
    return {"passed": passed, "last_stress_test": red_teams[0].created_at.isoformat() if red_teams else None}

async def generate_rbi_mlrg_report(model_id: str, db: AsyncSession):
    r1 = await check_model_approval_workflow(model_id, db)
    r2 = await check_bias_testing(model_id, db)
    r3 = await check_model_documentation(model_id, db)
    r4 = await check_stress_testing(model_id, db)
    
    return [
        {"article": "RBI MLRG", "title": "Model Approval Workflow", "status": "pass" if r1["passed"] else "fail", "evidence": f"Verdict: {r1.get('verdict')}", "remediation": "Approve model to CERTIFIED status"},
        {"article": "RBI MLRG", "title": "Bias Testing", "status": "pass" if r2["passed"] else "fail", "evidence": f"Fairness score: {r2.get('fairness_score')}", "remediation": "Run fairness check"},
        {"article": "RBI MLRG", "title": "Model Documentation", "status": "pass" if r3["passed"] else "fail", "evidence": f"Dimensions: {r3.get('dimensions_present')}", "remediation": "Ensure all 5 scoring dimensions are present"},
        {"article": "RBI MLRG", "title": "Stress Testing", "status": "pass" if r4["passed"] else "fail", "evidence": f"Last stress test: {r4.get('last_stress_test')}", "remediation": "Run a red team session"}
    ]
