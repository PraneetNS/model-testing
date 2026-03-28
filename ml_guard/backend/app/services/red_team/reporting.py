from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import os
import tempfile

def generate_red_team_pdf(report_data: dict, session_id: str) -> str:
    """
    Business logic for rendering adversarial audit findings into a PDF report.
    Extracted from router to maintain service-oriented architecture.
    """
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
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey), 
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    elements.append(t)
    elements.append(Spacer(1, 24))
    
    # Findings Table
    elements.append(Paragraph("Top Vulnerabilities (Sampleed)", styles["Heading2"]))
    findings_data = [["Category", "Severity", "Rounds", "Status"]]
    for f in report_data["findings"][:20]: # Show more in service-generated PDF
         status = "BREACHED" if f["is_successful"] else "REFUSED"
         findings_data.append([f["category"], f["severity"], str(f["rounds"]), status])
    
    ft = Table(findings_data, colWidths=[150, 100, 50, 100])
    ft.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 1, colors.black), 
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey)
    ]))
    elements.append(ft)
    
    doc.build(elements)
    return tmp_path
