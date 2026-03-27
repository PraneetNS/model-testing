# ML Guard Enterprise — v7.2

## Enterprise AI Governance Platform

ML Guard is a production-grade, multi-tenant SaaS platform for comprehensive AI model governance.
It provides offline audit, real-time monitoring, bias detection, and LLM safety evaluation
through a unified dashboard with deterministic, transparent, and auditable governance logic.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                   FRONTEND (Next.js 18)              │
│  10+ Modules: Audit│Report│Fairness│Behavior│Stream  │
│          LLM Guard│History│AI Advisor│Enterprise     │
└─────────────────┬────────────────────────────────────┘
                  │ REST + WebSocket
┌─────────────────▼────────────────────────────────────┐
│              BACKEND (FastAPI + Celery)               │
│  Routers: audit│fairness│behavior│monitoring│         │
│           streaming│llm_eval│advisory│enterprise      │
│           policies│alerts│ci│history│gate             │
└─────────────────┬────────────────────────────────────┘
                  │
┌─────────────────▼────────────────────────────────────┐
│          CORE SCIENTIFIC ENGINE (ml_guard/core)       │
│  metrics│drift│fairness│llm_guard│stream_drift│       │
│  policy│governance_score│calibration│sensitivity│     │
│  leakage│evaluator│advisory│constraints               │
└─────────────────┬────────────────────────────────────┘
                  │
┌─────────────────▼────────────────────────────────────┐
│                   DATA LAYER                          │
│  Neon PostgreSQL │ Upstash Redis │ MinIO Object Store │
└──────────────────────────────────────────────────────┘
```

---

## v7.2 Feature Matrix

| Module                | Type         | Status    |
|-----------------------|-------------|-----------|
| Model Audit           | Classical ML | ✅ Stable |
| Fairness Analysis     | Classical ML | ✅ Stable |
| Behavior Testing      | Classical ML | ✅ Stable |
| Live Monitoring       | Production   | ✅ Stable |
| Stream Drift          | Real-time    | ✅ Enhanced|
| LLM Governance        | LLM Safety   | ✅ Stable |
| AI Advisory           | Copilot      | ✅ Stable |
| **Scan History**      | Governance   | ✅ Stable |
| **Model Report Card** | Compliance   | ✅ Stable |
| **CI/CD Sync Gate**   | Automation   | ✅ **New (v7.2)** |

---

## Core Modules

### 1. Model Audit (`/api/v1/audit/run`)
Full offline governance scan: accuracy, F1, PSI/KS/JSD drift, overfitting,
calibration, leakage detection, data quality. Produces governance score, risk score,
and enterprise intelligence stream events.

### 2. CI/CD Governance Gate (NEW v7.2)
Synchronous evaluation for pipeline integration:
- **/api/v1/gate/evaluate**: Accepts `mlguard.yaml` policy & model path.
- **Sync mode**: Responds with a PASSED/FAILED verdict in under 60s.
- **Badge Integration**: Returns SVG badge URLs for PR comments.

### 3. Scan History & Compare
Historical tracking of all governance scans with side-by-side comparison:
- **Trajectory Sparklines**: Visualizes governance score trends across model versions.
- **Side-by-Side Diff**: Compare two scans to identify metric regression or drift.

### 4. Model Report Card
Generate professional, printable governance certificates:
- **Compliance Proof**: Consolidated summary of policy gates and statistical scores.
- **Audit Sign-off**: Ready-to-use template for risk management reviews.

---

## 🛡️ CI/CD Pipeline Integration

### mlguard.yaml (Policy-as-Code)
```yaml
version: "1.0"
model_name: "CustomerChurnPredictor-V2"
max_psi: 0.15
min_accuracy: 0.88
max_hallucination_rate: 0.04
```

### mlguard CLI
```bash
python ml_guard/sdk/python/mlguard_cli.py check \
  --policy mlguard.yaml \
  --artifact models/latest_model.pkl
```

---

## Technical Stack

| Layer     | Technology                              |
|-----------|-----------------------------------------|
| Frontend  | Next.js 18, React, TailwindCSS, Lucide  |
| Backend   | FastAPI, Uvicorn, Celery, Upstash Redis |
| Database  | Neon PostgreSQL (Serverless)            |
| Storage   | MinIO Object Store (S3-Compatible)      |
| ML Core   | NumPy, Pandas, Scikit-learn, SciPy       |

---

## File Structure

```
ml_guard/
├── core/                          # Scientific engine
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI application
│   │   ├── routers/              # Modular API controllers
│   │   │   ├── gate.py           # Sync evaluation gate (v7.2)
│   │   │   └── ...
│   │   └── db/                   # Neon PostgreSQL integration
├── sdk/
│   └── python/
│       ├── mlguard_cli.py        # CI/CD CLI Tool (v7.2)
│       └── ...
├── frontend/                      # Governance Dashboard
└── ...
```

---

**ML Guard v7.2** — Standardizing AI Governance through deterministic compliance.
Built with a composition-first architecture to ensure auditability across the full model lifecycle.
