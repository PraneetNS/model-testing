# ML Guard - Production-Grade Machine Learning Quality Gate

> *The CI/CD Quality Gate for Machine Learning Models*

ML Guard is a production-ready, open-source platform designed to validate, test, and gate machine learning models before they reach production. It combines statistical rigor with modern DevOps practices.

---

## 🏛 System Architecture

The system is designed as a modular, service-oriented architecture, separating concerns between test orchestration, data ingestion, and result visualization.

### 1. High-Level Components

*   **Backend (FastAPI):** The core engine. Exposes a REST API for model registration, test suite management, and execution.
*   **Frontend (React/Vite):** A modern dashboard for visualizing test results, drift, fairness metrics, and risk scores.
*   **CLI (`ml-guard`):** A developer-friendly command-line interface for local testing and CI/CD integration.
*   **Orchestrator:** Manages the execution flow of test suites.
*   **Validation Engine:** The heart of the system, running statistical tests (Drift, Bias, Performance, Robustness).

### 2. Backend Architecture (Python 3.10+)

The backend follows a strict Domain-Driven Design (DDD) approach.

#### Directory Structure:
```
backend/
├── app/
│   ├── api/            # API Routes (v1)
│   ├── core/           # Config, Logging, Security
│   ├── domain/         # Business Logic & Entities
│   │   ├── models/     # Pydantic Models (DTOs & DB Schemas)
│   │   ├── services/   # Orchestrator, Validation Engine
│   │   └── exceptions/ # Custom Exceptions
│   ├── infrastructure/ # External Services (DB, Storage, MLflow)
│   └── tests/          # Unit & Integration Tests
├── scripts/            # Database Migrations & seeds
└── main.py             # Entry Point
```

#### Key Modules:
*   **Test Orchestrator:** Handles the lifecycle of a test run. It parses the test suite definition (YAML/JSON or NLP-generated), loads the model and data, and dispatches jobs to the Validation Engine.
*   **Validation Engine:** Contains the logic for specific tests:
    *   `DriftDetector`: PSI, KS-Test, JS-Divergence.
    *   `BiasAnalyzer`: Disparate Impact, Equal Opportunity.
    *   `PerformanceEvaluator`: Accuracy, F1, ROC-AUC, Confidence calibration.
    *   `RobustnessTester`: Noise injection, Adversarial attacks (basic).
*   **Risk Scoring Engine:** Aggregates test results into a weighted score (0-100) and determines the `deployment_allowed` status.
*   **NLP Parser:** Uses lightweight NLP (KeyBERT/Transformers or rule-based) to convert natural language queries into structured test configurations.

### 3. Frontend Architecture (React + Vite)

The frontend is a Single Page Application (SPA) built for performance and user experience.

#### Tech Stack:
*   **Framework:** React (Vite)
*   **Styling:** TailwindCSS (Utility-first) + Shadcn UI (Component Library)
*   **State Management:** TanStack Query (React Query) for server state.
*   **Router:** React Router v6.
*   **Visualization:** Recharts (Responsive, composable charts).

#### Key Views:
*   **Dashboard:** High-level overview of model health and recent runs.
*   **Model Detail:** Deep dive into a specific model version's performance.
*   **Drift Analysis:** Visual comparison of training vs. production data distributions.
*   **Fairness Audit:** Bias metrics across protected attributes.
*   **Settings:** Team management, API keys, and project configuration.

### 4. CLI Architecture

The CLI is built with `Typer` for a robust developer experience.

#### Commands:
*   `ml-guard init`: Initialize a new ML Guard project.
*   `ml-guard scan <model_path>`: detailed analysis of a model artifact.
*   `ml-guard test --suite <suite_name>`: Run a specific test suite.
*   `ml-guard report --run-id <id>`: Generate a static report (HTML/PDF).

### 5. Data Flow

1.  **Ingestion:** User uploads model (`.pkl`, `.onnx`) and datasets (Reference & Current).
2.  **Profiling:** System automatically profiles the data (pandas-profiling/ydata-profiling logic).
3.  **Suggestion:** The `Auto Test Suggestion Engine` analyzes the profile and recommends a test suite.
4.  **Execution:** The `Orchestrator` runs the tests asynchronously.
5.  **Scoring:** Results are aggregated, risk score is calculated.
6.  **Gating:** Final status (`PASS`/`BLOCK`) is returned to the CI/CD pipeline.
7.  **Visualization:** Results are stored and viewable on the dashboard.

---

## 🛠 Technology Stack

| Component | Technology | Reasoning |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Standard for ML/Data Science. |
| **Web Framework** | FastAPI | High performance, async support, auto-docs. |
| **Data Processing** | Pandas, SciPy, NumPy | Robust numerical computing. |
| **ML Core** | Scikit-learn, XGBoost | Broad model support. |
| **Validation** | Pydantic v2 | Strict data validation and serialization. |
| **Frontend** | React + Vite + Tailwind | Modern, fast, and responsive UI. |
| **Charts** | Recharts | Composable and React-native charting. |
| **Task Queue** | BackgroundTasks (Simple) / Celery (Scale) | Asynchronous test execution. |
| **Containerization** | Docker | Consistent deployment environment. |

---

## 🚀 Roadmap

1.  **Phase 1 (MVP):** Core validation engine, basic CLI, and Streamlit-based prototype (Completed).
2.  **Phase 2 (Production):** Full React Frontend, FastAPI Backend, Robustness Tests, Risk Scoring (Current).
3.  **Phase 3 (Enterprise):** User Auth, RBAC, Persistent Storage (PostgreSQL), MLflow Integration.
4.  **Phase 4 (Scale):** Distributed execution, K8s Operator, Real-time monitoring.
