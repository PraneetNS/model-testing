# ML Guard Monitoring System: Production Architecture

This document outlines the architecture for high-scale model monitoring, drift detection, and alerting in ML Guard 2.0.

## 1. Architecture Overview

```text
[ ML MODEL (PROD) ] --( Batch Log )--> [ API SERVICE ] --( Persistence )--> [ POSTGRES (PredictionLogs) ]
                                             |
                                     ( Scheduled Trigger )
                                             |
                                     [ CELERY WORKER ]
                                             |
                        _____________________|_____________________
                       |                     |                     |
                [ BIAS SCAN ]          [ DRIFT ENGINE ]      [ STATS ENGINE ]
               (Fairness Audit)        (PSI, KS Test)        (Feature Stats)
                       |_____________________|_____________________|
                                             |
                                     [ ALERT PIPELINE ]
                                             |
                        _____________________|_____________________
                       |                     |                     |
                [ SLACK WEBHOOK ]      [ EMAIL SERVICE ]      [ CUSTOM HOOK ]
```

## 2. Drift Detection Engine (Worker Design)

The `MonitoringJob` triggers an async task that performs the following steps:
1.  **Reference Fetch**: Retrieve the "Reference Dataset" fingerprint and baseline distribution from the `TestRun` associated with the model version.
2.  **Current Fetch**: Pull the last $N$ predictions from `PredictionLog` for the specific project.
3.  **Statistical Comparison**:
    *   **PSI (Population Stability Index)**: Categorical and binned continuous drift.
    *   **KS Test**: Non-parametric test for continuous distribution shift.
    *   **JS Divergence**: Information-theory based distance between distributions.
4.  **Risk Escalation**: If drift exceeds `drift_threshold`, the Risk Level is escalated (e.g., from Low to High).

## 3. Alert Pipeline Design

Alerting is handled via an extensible **Observer Pattern**:
- **Critical Alerts**: Triggered when a `Critical` risk model drifts or bias is detected.
- **Degradation Alerts**: Triggered when the Quality Score (if ground truth is available) drops below a baseline.

**Example Alert Payload:**
```json
{
  "project": "ecommerce-recommendation",
  "event": "DRIFT_DETECTED",
  "severity": "High",
  "details": {
    "feature": "user_age_group",
    "psi": 0.24,
    "threshold": 0.10
  },
  "remediation": "Check for upstream data source changes or retrain model."
}
```

## 4. API Contract

### POST `/monitoring/predictions/log`
- **Request**:
  ```json
  {
    "project_id": "uuid",
    "model_version": "v2.1",
    "predictions": [
      {"feature_1": 10.5, "feature_2": "A", "prediction": 0.98},
      {"feature_1": 11.2, "feature_2": "B", "prediction": 0.12}
    ]
  }
  ```
- **Response**: `{"status": "success", "logged_count": 2}`

### GET `/monitoring/drift/history/{project_id}`
- **Response**:
  ```json
  [
    {
      "feature": "user_age",
      "metric": "PSI",
      "value": 0.12,
      "is_drifted": true,
      "timestamp": "2024-03-23T..."
    }
  ]
  ```

## 5. Storage Optimization (Long-term)
For high-volume production (~millions of predictions/day):
- **Partitioning**: Prediction tables are partitioned by `timestamp` (daily/weekly).
- **Archiving**: Older logs are compressed and moved to Cold Storage (S3) after 90 days.
- **Aggregation**: Drift is calculated on windows, and only aggregated metrics are kept in the primary SQL store.
