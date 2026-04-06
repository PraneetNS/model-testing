# ML Guard Python SDK

> The ML governance and observability SDK that goes beyond Evidently AI and WhyLabs.

[![PyPI version](https://badge.fury.io/py/mlguard.svg)](https://badge.fury.io/py/mlguard)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## Installation

```bash
pip install mlguard                    # core only
pip install "mlguard[sklearn]"         # + sklearn wrapper
pip install "mlguard[xgboost]"         # + xgboost wrapper
pip install "mlguard[all]"             # everything
```

---

## Why ML Guard SDK?

| Feature | Evidently | WhyLabs | **ML Guard** |
|---|---|---|---|
| Decorator instrumentation | ❌ | ❌ | ✅ `@monitor`, `@gate`, `@profile_input` |
| Governance score integration | ❌ | ❌ | ✅ Live score in every test |
| Privacy-preserving profiles | ❌ | ✅ (whylogs) | ✅ JSON-serializable, diff-able |
| Model object wrapping | ❌ | ❌ | ✅ `wrap_sklearn()`, `wrap_xgboost()` |
| CI/CD deploy blocker | ❌ | ❌ | ✅ `@gate(min_score=80)` |
| Compliance certificate | ❌ | ❌ | ✅ Verifiable cert hash |
| Test Suite → Governance score | ❌ | ❌ | ✅ Suites update live score |
| LLM Red-teaming | ❌ | ❌ | ✅ Built-in |
| Open-source + self-hosted | ✅ | ✅ | ✅ |

---

## Quick Start

### 1. Basic prediction logging

```python
import ml_guard

client = ml_guard.Client(
    host="http://localhost:8000",
    api_key="mlg_xxx"
)

# Log a single prediction (fire-and-forget, never raises)
client.log(
    model_id="churn-v2",
    features={"age": 34, "spend": 800, "tenure": 12},
    prediction=1,
    proba=0.87,
    latency_ms=4.2
)
```

### 2. Zero-code model wrapping (unique to ML Guard)

```python
from sklearn.ensemble import RandomForestClassifier
import ml_guard

model = RandomForestClassifier().fit(X_train, y_train)

# Wrap the model — same API, auto-instrumented
monitored = ml_guard.wrap_sklearn(model, model_id="churn-v2", client=client)

# All predictions auto-logged to ML Guard
predictions = monitored.predict(X_test)
probas = monitored.predict_proba(X_test)

# Input distribution profiled every 500 calls
# No code changes needed in your serving layer
```

### 3. Decorator instrumentation

```python
import ml_guard

@ml_guard.monitor(model_id="fraud-detector")
def predict(features: dict) -> float:
    return model.predict_proba([list(features.values())])[0][1]

# Every call to predict() is automatically logged
result = predict({"amount": 500, "merchant": "amazon"})
```

### 4. Governance-aware CI/CD gate

```python
import ml_guard

@ml_guard.gate(model_id="churn-v2", min_score=80.0)
def deploy_to_production():
    """This function is BLOCKED if governance score < 80."""
    kubernetes.deploy(image="churn-v2:latest")

# In your CI pipeline:
deploy_to_production()  # raises RuntimeError if score too low
```

### 5. Privacy-preserving data profiles

```python
import ml_guard

# Profile a DataFrame — sends only statistics, never raw data
profile = ml_guard.profile.from_dataframe(
    df=production_df,
    model_id="churn-v2",
    client=client
)

# Serialize to compact JSON (100 bytes vs megabytes of raw data)
json_str = profile.to_json()

# Compare two profiles without raw data
diff = profile.diff(reference_profile)
print(f"Drifted columns: {diff['drifted_columns']}")

# Flush to ML Guard backend
profile.flush()
```

### 6. Policy Test Suites (Evidently-style + governance)

```python
import ml_guard

suite = ml_guard.Suite(model_id="churn-v2", name="Production Quality Gate")

# Add tests
suite.add(ml_guard.tests.accuracy_above(0.85))
suite.add(ml_guard.tests.drift_psi_below(0.25))
suite.add(ml_guard.tests.null_rate_below(0.05))
suite.add(ml_guard.tests.governance_score_above(75.0, client=client))

# Add custom test
suite.add(ml_guard.tests.custom(
    name="revenue_feature_stable",
    fn=lambda ctx: ctx["df_current"]["revenue"].mean() > 100,
    message="Revenue feature mean dropped below 100"
))

# Run and print beautiful summary
results = suite.run(
    df_reference=train_df,
    df_current=production_df,
    model=model,
    label_col="target"
)
results.print_summary()

# Block CI pipeline on failure
results.assert_passed()
```

### 7. Profile input batches every N predictions

```python
import ml_guard

@ml_guard.profile_input(model_id="churn-v2", every_n=500, client=client)
def predict_batch(df: pd.DataFrame) -> np.ndarray:
    return model.predict(df)

# Automatically sends data profiles to ML Guard every 500 calls
results = predict_batch(production_df)
```

### 8. Compliance certificates

```python
# Generate a verifiable compliance certificate
cert = client.certify("churn-v2")
print(f"Cert hash: {cert['cert_hash']}")
print(f"Verify at: https://mlguard.io/verify/{cert['cert_hash']}")
```

---

## Full API Reference

### `ml_guard.Client`

| Method | Description |
|---|---|
| `log(model_id, features, prediction, ...)` | Log a single prediction |
| `log_batch(rows)` | Batch-log up to 10,000 predictions |
| `add_labels(log_ids, ground_truths)` | Add ground-truth labels |
| `get_score(model_id)` | Get governance score |
| `certify(model_id)` | Generate compliance certificate |
| `verify_cert(cert_hash)` | Verify a certificate (public) |
| `get_drift_report(model_id)` | Get drift analysis report |
| `get_performance(model_id)` | Get live performance metrics |
| `upload_profile(profile)` | Upload a DataProfile |
| `gate(model_id, policy_config)` | CI/CD gate check |
| `get_forecast(model_id)` | Get score forecast |

### `ml_guard.profile`

| Function | Description |
|---|---|
| `from_dataframe(df, model_id, ...)` | Build profile from DataFrame |
| `track(model_id, ...)` | Context manager for incremental tracking |
| `DataProfile.diff(reference)` | Compare two profiles |
| `DataProfile.quality_report()` | Data quality analysis |
| `DataProfile.to_json()` | Serialize to compact JSON |
| `DataProfile.flush()` | Send to backend |

### `ml_guard.tests` (Test Suite factories)

| Test | Description |
|---|---|
| `accuracy_above(threshold)` | Model accuracy >= threshold |
| `drift_psi_below(threshold, feature)` | PSI drift < threshold |
| `null_rate_below(threshold)` | Missing value rate < threshold |
| `governance_score_above(threshold)` | Live score >= threshold |
| `custom(name, fn, message)` | Any custom assertion |

### Decorators

| Decorator | Description |
|---|---|
| `@monitor(model_id)` | Auto-log every prediction |
| `@gate(model_id, min_score)` | Block execution if score too low |
| `@profile_input(model_id, every_n)` | Profile DataFrames every N calls |
| `@trace_prediction(model_id, ...)` | Advanced prediction tracing |

### Integrations

| Function | Description |
|---|---|
| `wrap_sklearn(model, model_id, client)` | Wrap any sklearn estimator |
| `wrap_xgboost(model, model_id, client)` | Wrap any XGBoost model |

---

## Environment Variables

```bash
export MLGUARD_HOST=http://localhost:8000
export MLGUARD_API_KEY=mlg_your_key_here
```

When set, all decorators work without passing `client` explicitly.

---

## CLI

```bash
# Check backend health
mlguard check --host http://localhost:8000

# Run a quick audit
mlguard audit --model-id churn-v2 --api-key mlg_xxx

# Get governance score
mlguard score --model-id churn-v2

# Generate certificate
mlguard certify --model-id churn-v2
```

---

## License

MIT — © ML Guard Team
