from jinja2 import Template
from .runner import MLTestReport
import json
import os

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ML Guard - Test Report</title>
    <style>
        body { font-family: 'Inter', sans-serif; background: #0f172a; color: white; padding: 40px; }
        .summary { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 40px; }
        .card { background: #1e293b; padding: 20px; rounded: 12px; border: 1px solid #334155; }
        .card h3 { margin: 0; font-size: 12px; color: #94a3b8; text-transform: uppercase; }
        .card p { font-size: 32px; font-weight: bold; margin: 10px 0 0 0; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th { text-align: left; padding: 12px; border-bottom: 2px solid #334155; color: #94a3b8; }
        td { padding: 12px; border-bottom: 1px solid #1e293b; }
        .status-passed { color: #10b981; }
        .status-failed { color: #ef4444; }
        .status-warning { color: #f59e0b; }
        .severity-critical { border-left: 4px solid #ef4444; }
    </style>
</head>
<body>
    <h1>ML Governance Audit Report</h1>
    <p>Suite: {{ metadata.suite_name }} | Generated: {{ metadata.timestamp }}</p>
    
    <div class="summary">
        <div class="card"><h3>Quality Index</h3><p>{{ summary.quality_index }}%</p></div>
        <div class="card"><h3>Passed</h3><p>{{ summary.passed }}</p></div>
        <div class="card"><h3>Failed</h3><p>{{ summary.failed }}</p></div>
        <div class="card"><h3>Errors</h3><p>{{ summary.errors }}</p></div>
    </div>

    <table>
        <thead>
            <tr>
                <th>Test Case</th>
                <th>Status</th>
                <th>Severity</th>
                <th>Metric</th>
                <th>Explanation</th>
            </tr>
        </thead>
        <tbody>
            {% for r in results %}
            <tr class="severity-{{ r.severity }}">
                <td>{{ r.name }}</td>
                <td class="status-{{ r.status }}">{{ r.status.upper() }}</td>
                <td>{{ r.severity.upper() }}</td>
                <td>{{ r.metric_value }}</td>
                <td>{{ r.explanation }}<br/><small style="color:#64748b">Remediation: {{ r.remediation }}</small></td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
</body>
</html>
"""

class TestReporter:
    def generate_json(self, report: MLTestReport, path: str):
        with open(path, 'w') as f:
            f.write(report.json(indent=4))

    def generate_html(self, report: MLTestReport, path: str):
        template = Template(HTML_TEMPLATE)
        html = template.render(
            summary=report.summary,
            results=report.results,
            metadata=report.metadata
        )
        with open(path, 'w') as f:
            f.write(html)
