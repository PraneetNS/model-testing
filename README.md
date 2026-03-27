# FireFlink ML Guard: Enterprise AI Governance Platform (v7.2)

## The Gold Standard in Machine Learning Reliability & LLM Governance

ML Guard brings FireFlink's scriptless testing excellence to both Tabular Machine Learning and Generative AI (LLMs). It serves as a mathematically rigorous "Quality Gate" that ensures models are accurate, safe, stable, and ethically sound before reaching production.

![ML Guard Status](https://img.shields.io/badge/Status-v7.2_Enterprise_Auditor-orange?style=for-the-badge)
![Security](https://img.shields.io/badge/Security-Hardened_Middleware-green?style=flat-square)
![Async](https://img.shields.io/badge/Tier-Async_Worker_Ready-blue?style=flat-square)

---

## 🚀 What's New in v7.2?

The latest evolution of ML Guard introduces the **Enterprise Governance Suite**, providing deep visibility into the model lifecycle and compliance certificate generation.

### 🔬 Multi-Modal Governance
- **Tabular Models**: Deep audit of Data Quality, PSI Drift, Performance, Robustness, and Fairness.
- **Generative AI (LLMs)**: Local-first evaluation for Hallucinations, Toxicity, Bias, and Injection robustness.

### 🛡️ v7.2 Enterprise Features
- **Scan History & Compare**: Track every governance audit with side-by-side metric comparison and **Trajectory Sparklines** for model health trends.
- **Model Report Cards**: Professional, exportable, and printable compliance certificates consolidating scores, policy gates, and statistical proof.
- **Live Notifications**: Real-time alerting bell with severity-based flyout for critical governance events.

### 🌐 Modern Infrastructure
- **Serverless Data Tier**: Powered by **Neon PostgreSQL** and **Upstash Redis** for global scalability.
- **Object Storage**: **MinIO** integration for secure, S3-compatible model and dataset artifact management.

---

## 📋 Table of Contents

- [Core Capabilities](#-core-capabilities)
- [Generative AI (LLM) Engine](#-generative-ai-llm-engine)
- [Quick Start](#-quick-start)
- [Architecture](#-architecture)
- [Test Categories](#-test-categories)
- [CI/CD Integration](#-cicd-pipeline-integration)

---

## 💎 Core Capabilities

*   **⚡ Zero-Block Execution**: Asynchronous processing ensures your UI remains responsive while the worker tier handles complex benchmarking.
*   **📜 Compliance Certificates**: Generate "Report Cards" that serve as official proof-of-governance for risk management stakeholders.
*   **📈 Trajectory Analysis**: Monitor how your model's governance score evolves over time with built-in trend visualization.
*   **🚫 Automated Quality Gates**: Block CI/CD pipelines automatically if models fail any critical governance criteria.

---

## 🤖 Generative AI (LLM) Engine

The v7.2 engine provides a comprehensive suite for auditing Large Language Models (OpenAI, HuggingFace, Custom Inference) with deterministic, local checks.

| Domain | Metrics & Audits | Governance Goal |
| :--- | :--- | :--- |
| **Integrity** | Hallucination Rate, Knowledge Score | Truthfulness & Accuracy |
| **Safety** | Toxicity Score, Injection Robustness | Content Safety & Security |
| **Ethics** | Bias Score, Stereotype Detection | Fairness & Neutrality |
| **Reliability** | Consistency Index, Latency Benchmarking | Production Stability |

---

## 🏗️ System Architecture

```mermaid
graph TD
    A[Premium Dashboard / Next.js] -->|REST + WS| B[FastAPI Gateway]
    B -->|Rate Limited| C(API Service Layer)
    C -->|Dispatch| D{Worker Tier}
    D -->|Celery/Redis| E[LLM Evaluator]
    D -->|Celery/Redis| F[Tabular Test Suites]
    E & F -->|Artifacts| H[(MinIO Object Store)]
    E & F -->|Persistence| G[(Neon PostgreSQL)]
    G --> B
    CI[CI/CD Pipelines] -->|Sync Evaluating| B
```

### Tech Stack
- **Frontend**: Next.js 18, React, Tailwind CSS, Lucide
- **Backend**: FastAPI, Pydantic, SQLAlchemy, JWT
- **Orchestration**: Celery, Upstash Redis
- **Persistence**: Neon PostgreSQL (Serverless)
- **Object Store**: MinIO (S3-Compatible)
- **Math Engine**: scikit-learn, Pandas, SciPy, Custom PSI Engine

---

## 🧪 Test Categories

1. **Data Quality**: Schema validation, uniqueness, row density, missing entropy.
2. **Statistical Stability**: PSI Drift (Population Stability Index), KS Divergence.
3. **Model Performance**: Accuracy, F1-Score, ROC-AUC, Overfitting gap.
4. **Robustness**: Input perturbation, stress testing, adversarial resilience.
5. **Bias & Fairness**: Protected attribute parity, subgroup performance analysis.
6. **LLM Safety**: Hallucination risks, jailbreak resistance, toxicity audits.

---

## 🛡️ CI/CD Pipeline Integration (v7.2)

ML Guard v7.2 introduces a **Synchronous Governance Gate** for direct integration into DevOps pipelines.

### Policy-as-Code (`mlguard.yaml`)
Define your release thresholds in a simple declarative file kept alongside your model code:

```yaml
version: "1.0"
model_name: "CustomerChurnPredictor-V2"
max_psi: 0.15
min_accuracy: 0.88
max_hallucination_rate: 0.04
bias_parity_threshold: 0.1
```

### ML Guard CLI
The `mlguard` tool provides immediate governance feedback in the terminal:

```bash
# Check model artifact against policy
python ml_guard/sdk/python/mlguard_cli.py check --policy mlguard.yaml --artifact models/latest_model.pkl

# Probe a production endpoint for safety
python ml_guard/sdk/python/mlguard_cli.py check --policy mlguard.yaml --url https://api.prod.inference.internal
```

### GitHub Actions Integration
Automatically block PR merges if a model fails any governance signal:

```yaml
jobs:
  governance:
    steps:
      - uses: actions/checkout@v3
      - name: ML Guard Gate
        run: |
          mlguard check --policy mlguard.yaml --artifact model_v2.pkl
```

Exits with `code 1` on failure, ensuring no non-compliant model reaches production.

---

## 📄 License & Status

- **Status**: Production-Ready (v7.2.0)
- **License**: MIT
- **Support**: (c) 2026 Antigravity AI Systems

---

**ML Guard** - Bringing FireFlink's testing excellence to the frontiers of AI. 🚀
