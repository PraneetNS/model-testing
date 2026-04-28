# Niyantrana SDK

Open-source AI governance SDK offering standalone capabilities for drift detection, fairness, explainability, and LLM guardrails. Connects seamlessly to the Niyantrana enterprise platform.

## Installation

```bash
pip install niyantrana
```

To install with advanced capabilities (Platform connection and Explainability):

```bash
pip install "niyantrana[all]"
```

## Quickstart

```python
import pandas as pd
from niyantrana.local.drift import detect_drift

# Drift Detection (Standalone)
ref_df = pd.DataFrame({"age": [25, 30, 35, 40]})
cur_df = pd.DataFrame({"age": [55, 60, 65, 70]})

report = detect_drift(ref_df, cur_df, method="psi")
print(f"Drift Detected: {report.overall_drift_detected}")
```

## Connecting to Niyantrana Platform

```python
from niyantrana.client import NiyantranaClient

client = NiyantranaClient(api_key="your-api-key")
client.log_prediction(
    model_id="my-model",
    features={"age": 30},
    prediction=1,
    probability=0.85,
    latency_ms=45
)
```

## License
Apache 2.0
