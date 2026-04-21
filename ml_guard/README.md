# 🛡️ ML Guard Enterprise — v8.5

## Enterprise AI Governance & Safety Platform

ML Guard is a high-performance, multi-tenant platform designed for technical AI governance. It provides a comprehensive suite of tools for offline model auditing, real-time streaming drift detection, LLM safety evaluation, and automated quality gates.

---

## 🚀 Recent Updates (v8.5)
- **Live Streaming Drift Monitor**: Real-time PSI, JSD, and stability tracking via WebSockets and REST ingestion.
- **LLM Guard & Safety**: Automated evaluation of prompt/response pairs for toxicity, injection, and hallucination.
- **Production Performance Probes**: Monitoring endpoint health, latency (p95), and error rates in enterprise environments.
- **Enhanced Security Middleware**: Integrated CSP management and injection detection for robust production deployments.

---

## 🏗️ Architecture

```mermaid
graph TD
    subgraph Frontend [Next.js 15 Dashboard]
        A[Governance Control Center]
        B[Real-time Stream Monitor]
        C[LLM Safety Guard]
        D[Audit & History]
    end

    subgraph Backend [FastAPI + Celery + Redis]
        E[Audit API]
        F[Streaming Router]
        G[LLM Eval Engine]
        H[Task Orchestrator]
    end

    subgraph Core [Governance Engine]
        I[Statistical Drift (PSI/JSD)]
        J[Safety Patterns]
        K[Risk Calibration]
    end

    subgraph Data [Storage]
        L[PostgreSQL]
        M[Redis]
        N[Object Storage]
    end

    Frontend -->|REST/WebSockets| Backend
    Backend -->|Async Tasks| Backend
    Backend -->|Analyzes| Core
    Backend -->|Persists| Data
```

---

## 🛠️ Feature Matrix

| Module | Type | Status | Key Metrics |
| :--- | :--- | :--- | :--- |
| **Model Audit** | Offline | ✅ Production | Accuracy, F1, PSI/KS Drift, Calibration |
| **Streaming Drift** | Live | ✅ **Stable** | Rolling PSI, JSD, Stability Score, Brier Score |
| **LLM Guard** | GenAI | ✅ **Stable** | Toxicity, Injection, Hallucination, Stability |
| **Performance Probe**| Infra | ✅ **Stable** | p95 Latency, Error Rate, CPU/Memory Utilization |
| **CI/CD Integration**| Automation | ✅ Enhanced | Blocking Quality Gates, Polling Status |
| **SHAP Explainer** | Transparency | ✅ Stable | Global/Local Feature Importance |

---

## 📡 API Usage Guide

### 1. Streaming Drift Ingestion
Submit live predictions to track distribution drift in real-time.
- **Endpoint**: `POST /api/v1/stream/production?model_id={model_id}`
- **Payload**:
```json
{
  "prediction": 0.82,
  "confidence": 0.95,
  "actual": 1.0,
  "features": [1.4, 0.45, 2.1]
}
```

### 2. LLM Safety Evaluation
Audit LLM interactions for governance and safety violations.
- **Endpoint**: `POST /api/v1/llm/evaluate`
- **Payload**:
```json
{
  "prompt": "Summarize the financial logs.",
  "response": "The report shows a 5% revenue growth.",
  "model_name": "gpt-4o-secure"
}
```

### 3. Production Health Probes
Log performance metrics from production inference endpoints.
- **Endpoint**: `POST /api/v1/monitoring/log`
- **Payload**:
```json
{
  "endpoint_url": "/api/v1/predict/v1",
  "status": "HEALTHY",
  "avg_latency_ms": 114.5,
  "p95_latency_ms": 156.2,
  "error_rate_pct": 0.01,
  "probe_count": 1000
}
```

---

## 🔧 Getting Started

### Prerequisites
- **Python 3.10+** (Backend)
- **Node.js 20+** (Frontend)
- **Redis** & **PostgreSQL**

### Installation
1. **Clone & Setup**:
   ```bash
   git clone https://github.com/PraneetNS/model-testing.git
   cd ml_guard
   ```

2. **Backend**:
   ```bash
   cd backend
   pip install -r requirements.txt
   uvicorn app.main:app --reload --port 8000
   ```

3. **Frontend**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

4. **Celery Worker**:
   ```bash
   cd backend
   celery -A app.core.celery_app worker --loglevel=info -P solo
   ```

---

**ML Guard v8.5** — Technical AI Governance for the Modern Enterprise.
© 2026 FireFlink ML Research. Proprietary & Confidential.
