import os
import tempfile
import structlog

logger = structlog.get_logger()

# Optional dependency handling for reportlab
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.graphics.shapes import Drawing, Circle, String, Arc
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.lib.units import inch
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    logger.warning("reportlab not installed. PDF generation will be unavailable.")

class PDFGenerator:
    """
    Professional governance report card PDF generator.
    Creates 3-page certificates with scoring gauges and breakdown tables.
    """
    
    def __init__(self, output_path: str):
        self.output_path = output_path
        if not HAS_REPORTLAB:
            self.doc = None
            self.styles = None
            self.elements = []
            return

        self.doc = SimpleDocTemplate(output_path, pagesize=letter)
        self.styles = getSampleStyleSheet()
        self.elements = []

    def _create_score_gauge(self, score: float):
        """Draws a circular gauge SVG placeholder for the score."""
        if not HAS_REPORTLAB:
            return None

        d = Drawing(200, 200)
        # Background Circle
        d.add(Circle(100, 100, 80, strokeColor=colors.lightgrey, fillColor=colors.white))
        # Value Label
        color = colors.green if score > 80 else colors.orange if score > 60 else colors.red
        d.add(String(100, 100, f"{int(score)}", textAnchor="middle", fontSize=40, fontName="Helvetica-Bold", fillColor=color))
        d.add(String(100, 70, "Gov Score", textAnchor="middle", fontSize=10, fillColor=colors.grey))
        return d

    def generate(self, report_data: dict):
        """
        Assemble the 3-page PDF document.
        """
        if not HAS_REPORTLAB:
            logger.error("Cannot generate PDF: reportlab is missing.")
            # Create a dummy file so the task doesn't fail on missing file
            with open(self.output_path, 'w') as f:
                f.write("PDF Generation Failed: reportlab missing.")
            return

        # --- Page 1: Cover ---
        title_style = ParagraphStyle(
            'ReportTitle', 
            parent=self.styles['Title'], 
            fontSize=28, 
            spaceAfter=20, 
            textColor=colors.HexColor("#1A1A1A")
        )
        self.elements.append(Paragraph("ML GUARD v7.2", self.styles['Heading2']))
        self.elements.append(Paragraph("GOVERNANCE REPORT CARD", title_style))
        self.elements.append(Spacer(1, 0.5*inch))
        
        # Big Gauge
        self.elements.append(self._create_score_gauge(report_data.get('overall_score', 0)))
        
        self.elements.append(Spacer(1, 0.5*inch))
        self.elements.append(Paragraph(f"Model: {report_data.get('model_name', 'M-ID-XXXX')}", self.styles['Normal']))
        self.elements.append(Paragraph(f"Audit Date: {report_data.get('issued_at', '')}", self.styles['Normal']))
        self.elements.append(Paragraph(f"Certificate Hash: {report_data.get('cert_hash', '')[:16]}...", self.styles['Normal']))
        
        badge_color = colors.green if report_data.get('verdict') == "CERTIFIED" else colors.orange if report_data.get('verdict') == "CONDITIONAL" else colors.red
        badge_style = ParagraphStyle('Badge', parent=self.styles['Normal'], backColor=badge_color, textColor=colors.white, alignment=1, borderPadding=10)
        self.elements.append(Spacer(1, 0.5*inch))
        self.elements.append(Paragraph(f"VERDICT: {report_data.get('verdict')}", badge_style))
        
        if report_data.get('metric_snapshots', {}).get('shap_fairness_alert'):
            alert_style = ParagraphStyle('Alert', parent=self.styles['Normal'], backColor=colors.red, textColor=colors.white, alignment=1, borderPadding=5)
            self.elements.append(Spacer(1, 0.2*inch))
            self.elements.append(Paragraph("🚨 SHAP-Fairness Alert: Top drifted feature is sensitive!", alert_style))

        parent_score_data = report_data.get("parent_score_data")
        if parent_score_data:
            self.elements.append(Spacer(1, 0.3*inch))
            self.elements.append(Paragraph("<b>Lineage Context:</b>", self.styles['Heading4'] if 'Heading4' in self.styles else self.styles['Heading3']))
            self.elements.append(Paragraph(f"Derived from parent: {parent_score_data['name']} (Score: {parent_score_data['score']})", self.styles['Normal']))

        self.elements.append(PageBreak())
        
        # --- Page 2: Breakdown ---
        self.elements.append(Paragraph("Score Breakdown & Gate Analysis", self.styles['Heading2']))
        self.elements.append(Spacer(1, 0.2*inch))
        
        data = [["Metric Category", "Score", "Threshold", "Status"]]
        for k, v in report_data.get('metric_snapshots', {}).items():
            if isinstance(v, (int, float)):
                status = "PASS" if v > 80 else "WARN" if v > 60 else "FAIL"
                # Use str labels to avoid reportlab complex issues with names
                data.append([str(k).replace('_', ' ').capitalize(), f"{v:.1f}", "80.0", status])
        
        table = Table(data, colWidths=[2.5*inch, 1*inch, 1*inch, 1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
            ('TEXTCOLOR', (0,0), (-1,0), colors.black),
            ('GRID', (0,0), (-1,-1), 1, colors.lightgrey),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('ALIGN', (1,1), (-1,-1), 'CENTER')
        ]))
        self.elements.append(table)
        
        self.elements.append(PageBreak())
        
        # --- Page 3: Summary ---
        self.elements.append(Paragraph("Executive Summary", self.styles['Heading2']))
        self.elements.append(Spacer(1, 0.2*inch))
        self.elements.append(Paragraph(report_data.get('executive_summary', ''), self.styles['Normal']))
        
        self.elements.append(Spacer(1, 1*inch))
        self.elements.append(Paragraph("Authorized Compliance Signature:", self.styles['Normal']))
        self.elements.append(Spacer(1, 0.3*inch))
        self.elements.append(Paragraph("________________________________", self.styles['Normal']))
        self.elements.append(Paragraph("ML GUARD Governance Autopilot Framework", self.styles['Normal']))
        
        # Build document
        self.doc.build(self.elements)
        logger.info("PDF Report Generated Successfully", path=self.output_path)
