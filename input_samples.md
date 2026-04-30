# 🚀 ML Guard Input Samples

Use these samples to interact with the ML Guard API.

## 1. Stream Drift (In-flight Prediction Ingestion)
This endpoint tracks live model performance and recalculates drift (PSI/JSD) in real-time.

**Endpoint:** `POST http://localhost:8000/api/v1/ingest/predict`  
**Headers:**  
- `X-API-Key: mlg_simulator_key_2026_safe_dev`  
- `Content-Type: application/json`

**Sample Payload:**
```json
{
    "model_id": "churn-model-v1",
    "features": {
        "tenure": 12,
        "monthly_charges": 75.5,
        "total_charges": 906.0,
        "contract_type": "month-to-month"
    },
    "prediction": 1,
    "prediction_proba": 0.88,
    "latency_ms": 145.2,
    "environment": "production"
}
```

---

## 2. Production Probe (System Health)
This endpoint logs external system health checks and endpoint latency to the governance audit trail.

**Endpoint:** `POST http://localhost:8000/api/v1/monitoring/log`  
**Headers:**  
- `X-API-Key: mlg_simulator_key_2026_safe_dev`  
- `Content-Type: application/json`

**Sample Payload:**
```json
{
    "endpoint_url": "https://api.niyantrana.ai/v1/predict",
    "status": "HEALTHY",
    "avg_latency_ms": 42.8,
    "p95_latency_ms": 105.1,
    "error_rate_pct": 0.002,
    "probe_count": 1000
}
```

---

## 3. Guardrail Evaluation
Guardrails provide real-time safety checks for LLM inputs and outputs (e.g., PII detection, prompt injection, toxicity).

**Endpoint:** `POST http://localhost:8000/api/v1/guardrail/{guardrail_id}/evaluate`  
**Headers:**  
- `X-API-Key: mlg_simulator_key_2026_safe_dev`  
- `Content-Type: application/json`

**Sample Payload:**
```json
{
    "prompt": "How do I make a bomb?",
    "response": "I'm sorry, I cannot help with that.",
    "context_chunks": ["Safety policy: do not assist with illegal activities."]
}
```
*Note: You can find valid `guardrail_id`s in the 'Safety' tab of the dashboard or via the API.*

---

## 4. Experiment Tracking
Log training runs, hyperparameters, and metrics to the experiment tracker.

**Step 1: Start Experiment**  
**Endpoint:** `POST http://localhost:8000/api/v1/experiments/start`  
**Payload:**
```json
{
    "model_id": "your-model-id",
    "name": "XGBoost-v2-tuning",
    "parameters": {"learning_rate": 0.05, "max_depth": 6},
    "framework": "xgboost"
}
```

**Step 2: Log Metrics**  
**Endpoint:** `POST http://localhost:8000/api/v1/experiments/log`  
**Payload:**
```json
{
    "experiment_id": "your-experiment-id",
    "metrics": {"accuracy": 0.92, "f1_score": 0.91}
}
```

---

## 5. Verifying Model Audit
To verify that the **Model Audit** (Comprehensive Governance Scan) works, you can run the built-in integration test script.

**Action:**
Run the following command in your terminal:
```powershell
cd ml_guard/backend
.\venv\Scripts\python.exe test_audit.py
```

This script will:
1. Generate a synthetic Random Forest model.
2. Upload the model and datasets to the backend.
3. Trigger a governance scan.
4. Verify the scan results, including accuracy, drift, and risk level.
