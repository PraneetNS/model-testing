# ML Guard Enterprise - Platform Input Guide

This guide provides sample inputs for every feature of the ML Guard platform to verify the newly integrated ML lifecycle governance.

## 1. Governance Audit (Core Lifecycle)
This is the **primary entry point** for automated discovery of models and datasets.
- **Model File**: `ml_guard/backend/fair_loan_model.pkl`
- **Training Data**: `ml_guard/backend/fair_loan_train.csv` (CSV with headers)
- **Validation Data**: `ml_guard/backend/fair_loan_test.csv`
- **Checks**: Select "Accuracy", "F1", "Security Audit", and "Explainability".
- **Effect**: Automatically registers the model in **Registry**, captures datasets in **Datasets**, creates **Lineage Links**, and logs an entry in **Experiments**.

## 2. Model Registry & Versioning
- **Action**: Go to **Registry** after running an Audit.
- **Input**: None (Auto-fetched).
- **Deployment**: Click the **"Deploy to DEV"** button on a specific model version to push it to the Deployments tab.

## 3. Dataset Inventory
- **Action**: Go to **Datasets**.
- **Input**: None (Automated sync with audits).
- **Manual (Optional)**: If you select "Register Dataset", provide:
  - **Name**: `Loan-Training-V1`
  - **Model**: Select from dropdown.
  - **Type**: Training.

## 4. Performance Monitoring (Real-time)
- **Action**: Go to **Performance**.
- **Input**: Prediction logs are captured via the API. To simulate real-time data:
  ```bash
  curl -X POST "http://localhost:8090/api/v1/monitoring/predictions" \
       -H "Content-Type: application/json" \
       -d '{"model_version_id": "YOUR_VERSION_ID", "latency_ms": 15, "confidence": 0.92, "prediction": {"label": 1}}'
  ```
- **Real-time**: The UI auto-refreshes every 5 seconds.

## 5. Model Security (Fixed)
- **Action**: Run a **Security Scan** in the Audit tab.
- **Expected Result**: Poisoning, Extraction, and Membership risks will show correctly (0-100% scale). The "10000 score" bug has been resolved.

## 6. Data Quality Validation
- **Action**: Upload a CSV in the **Data Quality** tab.
- **Input**: Use `ml_guard/backend/fair_loan_test.csv`.
- **Target Column**: `Approved` (if present) or leave blank.

## 7. Model Deployments
- **Action**: View the **Deployments** tab.
- **Verification**: Ensure you have pushed at least one model from the **Registry** using the "Deploy" button. Switch between **DEV**, **STAGING**, and **PRODUCTION** tabs.

## 8. CI/CD & Policy
- **Action**: Click **"Verify Compliance"** in the CI/CD tab.
- **Policy Input**: `gate_governance_score > 70`
- **Expected Result**: Pass/Fail based on the model's lineage.

---
**Note**: All features now fetch real data from the FastAPI backend. Mock data has been replaced with live database queries.
