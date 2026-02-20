# 🛡️ ML Guard: The Strategic ML Quality Gate

**ML Guard** is a production-grade MLOps platform designed to provide a mathematically rigorous and user-friendly "Quality Gate" for Machine Learning models. It moves beyond simple validation by combining natural language processing (NLP), advanced statistical divergence metrics, and AI-driven remediation to ensure models are safe, stable, and ethically sound before deployment.

---

## 🚀 Key Value Propositions

*   **Natural Language Testing**: Describe your test intent in plain English (e.g., *"Check drift and class balance"*), and the engine dynamically builds a strategic scan.
*   **Statistical Rigor**: Implements industry-standard mathematical formulas like **Population Stability Index (PSI)** with Laplacian smoothing and **Kolmogorov-Smirnov (KS)** tests.
*   **AI Remediation Engine**: Failed tests aren't just red flags; they come with automated, context-aware remediation advice explaining *why* they failed and *how* to fix them.
*   **Weighted Quality Index**: A sophisticated scoring system that calculates a deployment score (0-100) based on the statistical impact and severity of each failure.
*   **Premium Visualization**: A high-fidelity, dark-mode dashboard providing real-time telemetry and deep-trench analysis.

---

## 🛠️ Technology Stack

### Backend (The "Engine Room")
- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Asynchronous, High-Performance)
- **ML Core**: [Scikit-learn](https://scikit-learn.org/), [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Serialization**: [Joblib](https://joblib.readthedocs.io/) (High-performance model persistence)
- **Logging**: [Structlog](https://www.structlog.org/) (Structured JSON logging for production observability)
- **Validation**: [Pydantic v2](https://docs.pydantic.dev/)

### Frontend (The "Control Tower")
- **Framework**: [React](https://reactjs.org/) + [Vite](https://vitejs.dev/)
- **Styling**: [Tailwind CSS](https://tailwindcss.com/) with custom premium glassmorphism aesthetics.
- **Interactions**: Framer Motion-inspired micro-animations.

---

## 🏗️ Architecture & How It Works

ML Guard follows a **Domain-Driven Design (DDD)** pattern, separating business logic into specialized services:

### 1. The NLP Parser (`nlp_parser.py`)
Maps natural language queries to granular test categories. It identifies keywords like "drift", "bias", "accuracy", or "fairness" to build a customized test suite on the fly.

### 2. The Test Orchestrator (`orchestrator.py`)
The "Brain" of the operation. It:
- Builds the dynamic test suite configuration.
- Triggers the **Validation Engine**.
- Aggregates results into a **QualityGateResult**.
- Calculates the **Weighted Quality Index**.
- Enforces the **Deployment Gate** (Boolean Pass/Fail).

### 3. The Validation Engine (`validation_engine.py`)
A standardized interface that executes individual tests from the `test_categories.py` library. It handles the translation between raw data distributions and domain-standard `TestResult` objects.

### 4. The Trainer Service (`trainer.py`)
Handles automated preprocessing (Median/Mode imputation), feature alignment, and model training. It produces a "Pipeline Artifact" containing not just the model, but also metadata about encoders and features to ensure evaluate-time consistency.

---

## 🔬 Mathematical Implementation Details

### Population Stability Index (PSI)
The engine calculates drift using the formal divergence formula:
$$PSI = \sum_{i=1}^B (Actual\%_i - Expected\%_i) \cdot \ln\left(\frac{Actual\%_i}{Expected\%_i}\right)$$
We incorporate Laplacian smoothing ($10^{-6}$) to prevent logarithmic singularities and support both **Quantile Binning** (for continuous variables) and **Frequency Alignment** (for categorical variables).

### Weighted Scoring
The **Quality Index** is not a simple average. It uses a severity-weighted model:
- **Critical Failure**: -10 Weight (Triggers immediate Gate Fail)
- **High Severity**: -5 Weight
- **Medium/Low**: Progressive deductions.
Deployment is only allowed if **No Critical Failures** exist **AND** the Quality Index is **≥ 70%**.

---

## 📁 Project Structure

```text
├── backend/
│   ├── app/
│   │   ├── api/v1/         # FastAPI endpoints (Evaluating, Training)
│   │   ├── domain/
│   │   │   ├── models/     # Pydantic Domain Models
│   │   │   └── services/   # Orchestrator, Trainer, Testing Engine
│   └── main.py             # Entry point
├── frontend/
│   ├── src/
│   │   ├── Dashboard.tsx   # Premium Unified UI
│   │   └── components/     # Specialized UI views
│   └── index.css           # Design System & Token styles
└── README.md               # You are here
```

---

## 🚦 Getting Started

### Prerequisites
- Python 3.9+
- Node.js 18+

### Setup & Run
1. **Initialize Backend**:
   ```powershell
   cd backend
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   uvicorn app.main:app --reload
   ```

2. **Initialize Frontend**:
   ```powershell
   cd frontend
   npm install
   npm run dev
   ```

3. **Usage**:
   - Access the dashboard at `http://localhost:5173`.
   - Upload your `.pkl` model and `.csv` datasets.
   - Type your intent (e.g., *"Perform a comprehensive audit"*) and hit **Analyze**.

---
*(c) 2026 Antigravity AI - Advanced Agentic Coding Project*
