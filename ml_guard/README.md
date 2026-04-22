# 🛡️ ML Guard Enterprise — v8.5

## Enterprise AI Governance, Safety & Observability Platform

ML Guard is a high-performance, multi-tenant platform designed for technical AI governance. It provides a comprehensive suite of tools for offline model auditing, real-time streaming drift detection, LLM safety evaluation, and automated quality gates.

---

## 🏗️ Architecture Overview

ML Guard follows a modern distributed architecture designed for low-latency analysis and high-throughput background processing.

```mermaid
graph TD
    subgraph Client_Layer [Frontend - Next.js 15]
        A["Governance Control Center (React)"]
        B["Real-time Stream Monitor (WebSockets)"]
        C["LLM Safety Guard UI"]
        D["System Admin & Policy Dashboard"]
    end

    subgraph Service_Layer [Backend - FastAPI]
        E["Ingestion API (Sync/Async)"]
        F["Streaming Engine (Rolling Windows)"]
        G["LLM Evaluation Engine (Pattern Matching)"]
        H["Policy Enforcement Gate"]
    end

    subgraph Worker_Layer [Task Orchestration - Celery]
        I["Governance Audit Worker"]
        J["Statistical Analysis Tasks"]
        K["Alerting & Notification Engine"]
    end

    subgraph Data_Layer [Persistence & State]
        L[("PostgreSQL (Metadata & Results)")]
        M[("Redis (Queue & Real-time State)")]
        N[("Object Storage (Model Artifacts/PDFs)")]
    end

    Client_Layer -->|REST/WS| Service_Layer
    Service_Layer -->|Broker| M
    M -->|Task Queue| Worker_Layer
    Service_Layer -->|CRUD| L
    Worker_Layer -->|Persists| L
    Worker_Layer -->|Stores| N
```

---

## 🔄 Core Workflows

### 1. Model Audit Lifecycle
The model audit workflow ensures that every model deployment meets enterprise standards for accuracy, fairness, and robustness.

```mermaid
sequenceDiagram
    participant User as ML Engineer
    participant API as Backend API
    participant Worker as Celery Worker
    participant DB as Database
    
    User->>API: Upload Model & Datasets (/api/v1/audit/run)
    API->>DB: Create ScanRecord (Status: PENDING)
    API->>Worker: Dispatch Governance Task
    API-->>User: Return Submission Token
    Worker->>Worker: Run Accuracy/F1 Tests
    Worker->>Worker: Calculate PSI/JSD Drift
    Worker->>Worker: Run Security Scans (Poisoning/Extraction)
    Worker->>Worker: Evaluate Governance Score
    Worker->>DB: Update ScanRecord (Status: COMPLETED)
    User->>API: Poll for Results (/api/v1/gate/result/{token})
    API-->>User: Return Detailed Report & Deployment Status
```

### 2. Real-time Streaming Drift
ML Guard monitors live production traffic and detects distribution shifts before they impact business value.

```mermaid
graph LR
    P[Production App] -->|POST /stream/production| I[Ingestion Engine]
    I --> W[Rolling Window Manager]
    W --> S[Statistical Engine]
    S -->|Calculates| D[PSI / JSD / Stability]
    D --> G{Policy Check}
    G -->|Threshold Exceeded| A[Alert Event]
    A --> WS[WebSocket Dashboard]
    A --> SL[Slack/Email Notification]
```

---

## 🛡️ Security & Authentication

ML Guard uses a robust **X-API-Key** based authentication system with SHA-256 hashing for all protected resources.

### Using the API Key
Every request to the backend must include the `X-API-Key` header.

```bash
curl -X POST "http://localhost:8000/api/v1/llm/evaluate" \
     -H "X-API-Key: YOUR_API_KEY_HERE" \
     -H "Content-Type: application/json" \
     -d '{...}'
```

> [!IMPORTANT]
> In local development mode, use the key: `mlg_K0njzcPf5hS7AKtccxdePVglpJMiZnZX`

---

## 🔧 Getting Started

### Quick Start (Windows)
We provide a helper script to launch all services simultaneously:

1. Open PowerShell and navigate to the project root.
2. Run the startup script:
   ```powershell
   .\ml_guard\start_services.bat
   ```
This will start the **Redis Server**, **FastAPI Backend**, **Celery Worker**, and **Next.js Frontend**.

### Manual Installation

#### 1. Backend Setup
```bash
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

#### 2. Celery Worker Setup
```bash
cd backend
.\venv\Scripts\activate
celery -A app.core.celery_app worker --loglevel=info -P solo
```

#### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

---

## 🛠️ Feature Matrix

| Module | Type | Status | Key Metrics |
| :--- | :--- | :--- | :--- |
| **Model Audit** | Offline | ✅ Production | Accuracy, F1, PSI/KS Drift, Calibration |
| **Streaming Drift** | Live | ✅ **Stable** | Rolling PSI, JSD, Stability Score, Brier Score |
| **LLM Guard** | GenAI | ✅ **Stable** | Toxicity, Injection, Hallucination, Stability |
| **Performance Probe**| Infra | ✅ **Stable** | p95 Latency, Error Rate, CPU/Memory Utilization |
| **Red Teaming** | Adversarial | ✅ **Stable** | Jailbreak Success, Vulnerability Mapping |
| **SHAP Explainer** | Transparency | ✅ Stable | Global/Local Feature Importance |

---

**ML Guard v8.5** — Technical AI Governance for the Modern Enterprise.
© 2026 FireFlink ML Research. Proprietary & Confidential.
