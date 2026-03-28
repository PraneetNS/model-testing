from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from app.db.session import SessionLocal
from app.db.models import RedTeamSession, RedTeamAttack, Model
from app.services.red_team.tasks import execute_red_team_campaign
from app.core.security import decrypt_content
from io import BytesIO
import tempfile
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/start", response_model=dict)
async def start_red_team_session(model_id: str, max_attacks: int = 10, db: Session = Depends(get_db)):
    """Initialize a red-teaming session and dispatch to background worker."""
    model = db.query(Model).get(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
        
    session = RedTeamSession(model_id=model_id, total_attacks=0, success_count=0)
    db.add(session)
    db.commit()
    db.refresh(session)
    
    # Fire Celery task
    execute_red_team_campaign.delay(str(session.id), max_attacks)
    
    return {"session_id": str(session.id), "status": "RUNNING", "message": "Campaign dispatched."}

@router.get("/{session_id}/report")
async def get_red_team_report(session_id: str, db: Session = Depends(get_db)):
    """Return structured findings and summary for a red-teaming session."""
    session = db.query(RedTeamSession).get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    attacks = db.query(RedTeamAttack).filter(RedTeamAttack.session_id == session.id).all()
    
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
async def export_red_team_pdf(session_id: str, db: Session = Depends(get_db)):
    """Generate and return a professional PDF summary of red-team findings."""
    report_data = await get_red_team_report(session_id, db)
    
    tmp_path = os.path.join(tempfile.gettempdir(), f"RedTeamReport_{session_id}.pdf")
    doc = SimpleDocTemplate(tmp_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Title
    elements.append(Paragraph(f"ML Guard Red Team Autopilot Report", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Session: {session_id}", styles["Normal"]))
    elements.append(Paragraph(f"Model ID: {report_data['model_id']}", styles["Normal"]))
    elements.append(Spacer(1, 24))
    
    # Summary Stats
    elements.append(Paragraph("Executive Summary", styles["Heading2"]))
    sum_data = [
        ["Total Attacks", str(report_data['summary']['total_attacks'])],
        ["Success Rate", f"{report_data['summary']['success_rate']*100:.2f}%"],
        ["Critical Findings", str(report_data['summary']['critical_vulnerabilities'])]
    ]
    t = Table(sum_data, colWidths=[200, 100])
    t.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey), ('GRID', (0,0), (-1,-1), 1, colors.black)]))
    elements.append(t)
    elements.append(Spacer(1, 24))
    
    # Findings Table
    elements.append(Paragraph("Top Vulnerabilities", styles["Heading2"]))
    findings_data = [["Category", "Severity", "Rounds", "Status"]]
    for f in report_data["findings"][:10]: # Limit for PDF
         status = "BREACHED" if f["is_successful"] else "REFUSED"
         findings_data.append([f["category"], f["severity"], str(f["rounds"]), status])
    
    ft = Table(findings_data, colWidths=[150, 100, 50, 100])
    ft.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black), ('BACKGROUND', (0,0), (-1,0), colors.lightgrey)]))
    elements.append(ft)
    
    doc.build(elements)
    
    return FileResponse(tmp_path, filename=f"MLGuard_RedTeam_Report.pdf")
