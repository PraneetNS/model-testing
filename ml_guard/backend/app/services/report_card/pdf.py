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

    def _add_insurance_section(self, insurance_data: dict):
        """Adds an actuarial risk profile section for insurance brokers."""
        if not HAS_REPORTLAB or not insurance_data:
            return

        self.elements.append(Paragraph("AI Liability Insurance Actuarial Audit", self.styles['Heading2']))
        self.elements.append(Spacer(1, 0.2*inch))
        self.elements.append(Paragraph(f"<b>Actuarial Tier:</b> {insurance_data.get('tier', 'Standard').upper()}", self.styles['Normal']))
        self.elements.append(Paragraph(f"<b>ML Guard Insurance Score:</b> {insurance_data.get('total_score', 0)} / 1000", self.styles['Normal']))
        self.elements.append(Spacer(1, 0.1*inch))
        
        # Premium Estimate
        premium = insurance_data.get('estimated_annual_premium_usd_range', {})
        self.elements.append(Paragraph(f"<b>Estimated Annual Premium:</b> ${premium.get('min', 0):,} - ${premium.get('max', 0):,} USD", self.styles['Normal']))
        self.elements.append(Spacer(1, 0.2*inch))

        # Risk Breakdown Table
        dim_data = [["Dimension", "Score (max)"]]
        for dim, score in insurance_data.get('dimension_scores', {}).items():
            dim_data.append([dim.replace('_', ' ').capitalize(), f"{score}"])
        
        dim_table = Table(dim_data, colWidths=[3*inch, 2*inch])
        dim_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('BOX', (0,0), (-1,-1), 1, colors.black),
        ]))
        self.elements.append(dim_table)
        self.elements.append(Spacer(1, 0.3*inch))
        
        # Risk Factors
        factors = insurance_data.get('risk_factors', [])
        if factors:
            self.elements.append(Paragraph("<b>Primary Actuarial Risk Factors:</b>", self.styles['Normal']))
            for f in factors[:3]:
                self.elements.append(Paragraph(f"• {f['factor']}: {f['recommendation']}", self.styles['Normal']))

        self.elements.append(PageBreak())

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
        risk_tier = report_data.get('risk_tier', 'N/A').upper()
        risk_color = colors.red if risk_tier == "CRITICAL" else colors.orange if risk_tier == "HIGH" else colors.yellow if risk_tier == "MEDIUM" else colors.green
        risk_style = ParagraphStyle('RiskBadge', parent=self.styles['Normal'], backColor=risk_color, textColor=colors.white if risk_tier in ["CRITICAL", "HIGH"] else colors.black, alignment=0, borderPadding=4)
        self.elements.append(Spacer(1, 0.1*inch))
        self.elements.append(Paragraph(f"RISK TIER: {risk_tier}", risk_style))
        self.elements.append(Spacer(1, 0.2*inch))
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
        
        # --- Supply Chain Integrity (AIBOM) ---
        aibom_data = report_data.get('metric_snapshots', {}).get('aibom')
        if aibom_data:
            aibom_hash = aibom_data.get('aibom_hash')
            if aibom_hash:
                self.elements.append(Spacer(1, 0.5*inch))
                self.elements.append(Paragraph("Supply Chain Integrity (AIBOM)", self.styles['Heading3']))
                self.elements.append(Spacer(1, 0.1*inch))
                self.elements.append(Paragraph(f"This model is cryptographically verified by an AI Bill of Materials (AIBOM).", self.styles['Normal']))
                self.elements.append(Paragraph(f"<b>AIBOM Manifest Hash:</b> {aibom_hash}", self.styles['Normal']))
                
                cve_alerts = aibom_data.get('cve_alerts', [])
                if cve_alerts:
                    alert_style = ParagraphStyle('AIBOMAlert', parent=self.styles['Normal'], textColor=colors.red)
                    self.elements.append(Paragraph(f"⚠️ Warning: {len(cve_alerts)} dependency vulnerabilities detected in supply chain scan.", alert_style))

        # --- AI Liability Insurance ---
        insurance_data = report_data.get('insurance_report')
        if insurance_data:
            self.elements.append(PageBreak())
            self._add_insurance_section(insurance_data)

        self.elements.append(Spacer(1, 1*inch))
        self.elements.append(Paragraph("Authorized Compliance Signature:", self.styles['Normal']))
        self.elements.append(Spacer(1, 0.3*inch))
        self.elements.append(Paragraph("________________________________", self.styles['Normal']))
        self.elements.append(Paragraph("ML GUARD Governance Autopilot Framework", self.styles['Normal']))
        
        # --- Page 4: Regulatory Compliance (Appendix) ---
        include_compliance = report_data.get('include_compliance', True) # Optionally included, default True
        if include_compliance:
            self.elements.append(PageBreak())
            self.elements.append(Paragraph("Appendix: Regulatory Compliance", self.styles['Heading2']))
            self.elements.append(Spacer(1, 0.2*inch))
            
            import os
            import sys
            _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../"))
            if _repo_root not in sys.path:
                sys.path.append(_repo_root)
            from ml_guard.core.compliance import evaluate_compliance
            
            comp_results = evaluate_compliance(report_data.get('metric_snapshots', {}))
            
            comp_data = [["Control", "Status", "Evidence", "Gap"]]
            for r in comp_results:
                status_text = r['status'].upper()
                # Wrap text slightly
                ev = r['evidence'][:40] + "..." if len(r['evidence']) > 40 else r['evidence']
                gap = (r['gap'][:40] + "...") if r['gap'] and len(r['gap']) > 40 else (r['gap'] or "-")
                comp_data.append([
                    r['control'],
                    status_text,
                    Paragraph(ev, self.styles['Normal']),
                    Paragraph(gap, self.styles['Normal'])
                ])
                
            table2 = Table(comp_data, colWidths=[1.5*inch, 1*inch, 2.5*inch, 2.5*inch])
            
            styles_list = [
                ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
                ('TEXTCOLOR', (0,0), (-1,0), colors.black),
                ('GRID', (0,0), (-1,-1), 1, colors.lightgrey),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
            ]
            for row_idx, r in enumerate(comp_results, start=1):
                if r['status'] == 'pass':
                    bg_color = colors.lightgreen
                elif r['status'] == 'fail':
                    bg_color = colors.lightcoral
                else:
                    bg_color = colors.navajowhite # amber
                    
                styles_list.append(('BACKGROUND', (0, row_idx), (-1, row_idx), bg_color))
            
            table2.setStyle(TableStyle(styles_list))
            self.elements.append(table2)

        # Build document
        self.doc.build(self.elements)
        logger.info("PDF Report Generated Successfully", path=self.output_path)
