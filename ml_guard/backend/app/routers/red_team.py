from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.session import get_db
from app.db.models import RedTeamSession, RedTeamAttack, Model
from app.tasks.red_team import execute_red_team_campaign
from app.services.red_team.reporting import generate_red_team_pdf
from app.core.security import decrypt_content

router = APIRouter()

@router.post("/start", response_model=dict)
async def start_red_team_session(model_id: str, max_attacks: int = 10, db: AsyncSession = Depends(get_db)):
    """Initialize a red-teaming session and dispatch to background worker."""
    model = db.query(Model).get(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
        
    session = RedTeamSession(model_id=model_id, total_attacks=0, success_count=0)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    
    # Fire Celery task
    execute_red_team_campaign.delay(str(session.id), max_attacks)
    
    return {"session_id": str(session.id), "status": "RUNNING", "message": "Campaign dispatched."}

@router.get("/{session_id}/report")
async def get_red_team_report(session_id: str, db: AsyncSession = Depends(get_db)):
    """Return structured findings and summary for a red-teaming session."""
    session = db.query(RedTeamSession).get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    attacks = (await db.execute(select(RedTeamAttack).filter(RedTeamAttack.session_id == session.id))).scalars().all()
    
    findings = []
    for a in attacks:
        findings.append({
            "id": str(a.id),
            "category": a.category,
            "severity": a.severity,
            "rounds": a.rounds,
            "is_successful": a.is_successful,
            "prompt": decrypt_content(a.encrypted_prompt),
            "response": decrypt_content(a.encrypted_response) if a.encrypted_response else None,
            "reasoning": a.judge_reasoning,
            "timestamp": a.created_at.isoformat()
        })
        
    return {
        "session_id": str(session.id),
        "model_id": str(session.model_id),
        "status": session.status,
        "summary": {
            "total_attacks": session.total_attacks,
            "success_rate": round(session.success_count / session.total_attacks, 4) if session.total_attacks > 0 else 0,
            "critical_vulnerabilities": len([f for f in findings if f["severity"] == "CRITICAL" and f["is_successful"]])
        },
        "findings": findings
    }

@router.get("/{session_id}/report/pdf")
async def export_red_team_pdf(session_id: str, db: AsyncSession = Depends(get_db)):
    """Generate and return a professional PDF summary of red-team findings (via Service)."""
    report_data = await get_red_team_report(session_id, db)
    # Move complexity to services/red_team/reporting.py
    tmp_path = generate_red_team_pdf(report_data, session_id)
    return FileResponse(tmp_path, filename=f"MLGuard_RedTeam_Report.pdf")
