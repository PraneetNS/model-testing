# ML Guard Compliance Mode: Architecture & Design

Compliance Mode transforms technical test results into formal regulatory documentation, ensuring models adhere to enterprise and legal standards (e.g., EU AI Act, GDPR).

## 1. Compliance Architecture

The Compliance layer sits atop the core Testing Framework:
1.  **Orchestrator**: Executes technical tests.
2.  **Profiler**: Adds context (Protected attributes, Risk level).
3.  **Compliance Service**: Maps technical output into the five **Regulatory Pillars**:
    *   **Fairness**: Bias metrics and disparate impact.
    *   **Explainability**: SHAP global feature contribution and model transparency.
    *   **Stability**: Drift metrics (PSI/KS) and historical consistency.
    *   **Robustness**: Stability under noise and adversarial resistance.
    *   **Reproducibility**: Environment snapshots and dataset fingerprints.

## 2. Audit Report Structure (JSON)

Every test run can generate a formal `ModelAuditReport`:
```json
{
  "report_id": "AUDIT-uuid",
  "generated_at": "ISO-TIMESTAMP",
  "risk_summary": {
    "score": 85.2,
    "risk_level": "Medium",
    "deployment_allowed": true
  },
  "pillars": {
    "Fairness": {"status": "PASS", "failure_count": 0},
    "Explainability": {"status": "PASS", "failure_count": 0}
  },
  "compliance_checklist": [
    {"item": "Reproducibility Verified", "status": "YES"}
  ]
}
```

## 3. PDF Report Template Design

The exportable PDF is structured for human auditors and stakeholders:
- **Section I: Executive Summary**: High-level pass/fail and risk classification.
- **Section II: Model Identity**: Reproducibility token, environment config, and versioning.
- **Section III: Pillar Deep-Dive**: Detailed evidence for Fairness, Stability, and Robustness.
- **Section IV: Transparency Report**: Top 10 feature importance visualizations.
- **Section V: Dataset Documentation**: Fingerprints for training/validation sets.
- **Section VI: Governance Log**: Audit trail of when the test was run and by whom.

## 4. Risk Scoring Framework

ML Guard uses a **Multi-Tier Risk Assessment**:
- **Baseline Risk**: Determined by the `Profiler` based on dataset sensitivity.
- **Observed Risk**: Determined by the `RiskEngine` based on test failures.
- **Compliance Override**: Certain failures (e.g., Bias in a Critical project) trigger an immediate **Compliance FAIL** regardless of the numerical score.
