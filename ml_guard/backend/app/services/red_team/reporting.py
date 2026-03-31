import os
import tempfile
import structlog

logger = structlog.get_logger()

# Optional dependency guard for reportlab in Red Teaming reports
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    logger.warning("reportlab not installed. Red Team PDF generation will be unavailable.")

def generate_red_team_pdf(report_data: dict, session_id: str) -> str:
    """
    Business logic for rendering adversarial audit findings into a PDF report.
    Guarded for environments without reportlab.
    """
    tmp_path = os.path.join(tempfile.gettempdir(), f"RedTeamReport_{session_id}.pdf")
    
    if not HAS_REPORTLAB:
        logger.error("Cannot generate Red Team PDF: reportlab is missing.")
        with open(tmp_path, 'w') as f:
            f.write(f"Red Team Audit Session {session_id} - PDF Generation Failed: reportlab missing.")
        return tmp_path

    doc = SimpleDocTemplate(tmp_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Title
    elements.append(Paragraph(f"ML Guard Red Team Autopilot Report", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Session: {session_id}", styles["Normal"]))
    elements.append(Paragraph(f"Model ID: {report_data.get('model_id', 'Unknown')}", styles["Normal"]))
    elements.append(Spacer(1, 24))
    
    # Summary Stats
    elements.append(Paragraph("Executive Summary", styles["Heading2"]))
    summary = report_data.get('summary', {})
    sum_data = [
        ["Total Attacks", str(summary.get('total_attacks', 0))],
        ["Success Rate", f"{summary.get('success_rate', 0.0)*100:.2f}%"],
        ["Critical Findings", str(summary.get('critical_vulnerabilities', 0))]
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
    elements.append(Paragraph("Top Vulnerabilities (Sampled)", styles["Heading2"]))
    findings_data = [["Category", "Severity", "Rounds", "Status"]]
    for f in report_data.get("findings", [])[:20]:
         status = "BREACHED" if f.get("is_successful") else "REFUSED"
         findings_data.append([f.get("category", ""), f.get("severity", ""), str(f.get("rounds", 1)), status])
    
    ft = Table(findings_data, colWidths=[150, 100, 50, 100])
    ft.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 1, colors.black), 
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey)
    ]))
    elements.append(ft)
    
    doc.build(elements)
    return tmp_path
