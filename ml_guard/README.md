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
│           policies│alerts│ci│history                  │
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
| **Scan History**      | Governance   | ✅ **New (v7.2)** |
| **Model Report Card** | Compliance   | ✅ **New (v7.2)** |
| **Live Notifications**| Alerting     | ✅ **New (v7.2)** |

---

## Core Modules

### 1. Model Audit (`/api/v1/audit/run`)
Full offline governance scan: accuracy, F1, PSI/KS/JSD drift, overfitting,
calibration, leakage detection, data quality. Produces governance score, risk score,
and enterprise intelligence stream events.

### 2. Scan History & Compare (NEW v7.2)
Historical tracking of all governance scans with side-by-side comparison:
- **Trajectory Sparklines**: Visualizes governance score trends across model versions.
- **Side-by-Side Diff**: Compare two scans to identify metric regression or drift.
- **Deep Audit Retrieval**: Instantly load full results from any historical scan.

### 3. Model Report Card (NEW v7.2)
Generate professional, printable governance certificates:
- **Compliance Proof**: Consolidated summary of policy gates and statistical scores.
- **Export Ready**: Optimized for PDF export and digital verification.
- **Audit Sign-off**: Ready-to-use template for risk management reviews.

### 4. Live Notifications Bell (NEW v7.2)
Real-time alerting integrated directly into the header:
- **Live Polling**: Monitors critical system alerts and model failures every 30s.
- **Severity Coding**: Immediate visual feedback for Critical, Warning, and Info events.
- **Unread Tracking**: Smart badge indicates new unaddressed governance events.

### 5. Fairness & Bias Detection
Enterprise-grade bias detection with metrics: Statistical Parity Difference (SPD), Equal Opportunity Difference (EOD), and Disparate Impact Ratio (DIR).

### 6. LLM Governance
Deterministic safety checks: Prompt Injection Detection, Toxicity Scoring, Hallucination Risk, and Response Stability.

### 7. Streaming Drift Detection (WebSocket)
Real-time sliding-window drift monitoring with adaptive thresholding and consecutive-window alerting.

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
│   │   └── db/                   # Neon PostgreSQL integration
├── frontend/
│   ├── src/app/dashboard/
│   │   ├── page.tsx               # Unified dashboard shell
│   │   ├── modules/
│   │   │   ├── ScanHistoryModule.tsx    # History & Compare (v7.2)
│   │   │   ├── ModelReportCardModule.tsx# Compliance Certificates (v7.2)
│   │   │   └── ...
│   │   └── components/
│   │       ├── NotificationsBell.tsx    # Live alerting flyout (v7.2)
│   └── ...
```

---

**ML Guard v7.2** — Standardizing AI Governance through deterministic compliance.
Built with a composition-first architecture to ensure auditability across the full model lifecycle.
