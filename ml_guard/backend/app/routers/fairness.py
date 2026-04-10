"""
Fairness & Bias Detection Router.
Endpoints for analyzing model fairness with respect to sensitive features.
"""
import uuid
import pandas as pd
import numpy as np
import joblib
import tempfile
import os
import sys
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form

# ML Guard core path injection
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if _repo_root not in sys.path:
    sys.path.append(_repo_root)

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.session import get_db
from app.db.models import Job, FairnessResult, AuditLog, ScanRecord
from app.core.auth import AuthContext, get_auth_context, log_action
from ml_guard.core import ONNXModelWrapper

router = APIRouter()


@router.post("/fairness/analyze")
async def analyze_fairness(
    model_file: UploadFile = File(...),
    data_file: UploadFile = File(None),
    dataset_url: str = Form(None),
    sensitive_column: str = Form(...),
    label_col: str = Form("target"),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
):
    """
    Full fairness analysis endpoint.

    Expects:
      - model_file: Trained model (.pkl/.joblib/.onnx)
      - data_file: Evaluation dataset (CSV/Parquet)
      - dataset_url: Alternatively, a MinIO/HTTP URL for the dataset
      - sensitive_column: Name of the column containing the sensitive feature
      - label_col: Name of the target/label column
    """
    from ml_guard.core.fairness import compute_fairness
    from ml_guard.core.policy import evaluate_policy
    from app.services.storage_service import download_from_url

    # ─── Load model ───
    model_bytes = await model_file.read()
    suffix = ".onnx" if model_file.filename.lower().endswith(".onnx") else ".pkl"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(model_bytes)
        tmp_model = f.name
    try:
        if suffix == ".onnx":
            model = ONNXModelWrapper(tmp_model)
        else:
            model = joblib.load(tmp_model)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load model: {e}")
    finally:
        os.unlink(tmp_model)

    import io
    from sklearn.preprocessing import LabelEncoder

    # ─── Load data ───
    if data_file:
        data_bytes = await data_file.read()
        filename = data_file.filename
    elif dataset_url:
        data_bytes = download_from_url(dataset_url)
        filename = dataset_url
    else:
        raise HTTPException(400, "Provide either a data file or a dataset URL.")

    import io
    from sklearn.preprocessing import LabelEncoder

    try:
        if filename.lower().endswith(".parquet"):
            df = pd.read_parquet(io.BytesIO(data_bytes))
        else:
            try:
                # Try standard UTF-8 first
                df = pd.read_csv(io.BytesIO(data_bytes))
            except (UnicodeDecodeError, pd.errors.ParserError):
                # Fallback to Latin1
                df = pd.read_csv(io.BytesIO(data_bytes), encoding='latin-1')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read dataset: {e}")

    # ─── Validate columns ───
    if label_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Label column '{label_col}' not found. Available: {list(df.columns)}")
    if sensitive_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Sensitive column '{sensitive_column}' not found. Available: {list(df.columns)}")

    # ─── Prepare Features ───
    feature_cols = [c for c in df.columns if c not in [label_col, sensitive_column]]
    # Only keep numeric columns for the model if not specified otherwise
    X = df[feature_cols].select_dtypes(include=[np.number])
    if X.shape[1] == 0:
         # Fallback to all feature columns if no numeric ones found (model might handle strings)
         X = df[feature_cols]

    # ─── Map Labels to Numeric 0/1 for Fairness Logic ───
    y_true_raw = df[label_col].values
    sensitive = df[sensitive_column].values

    try:
        y_pred_raw = model.predict(X.values)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Model prediction failed: {e}")

    # Unified Label Encoding for both True and Pred
    # This ensures "Approved"/"Denied" becomes 1/0
    all_labels = np.unique(np.concatenate([y_true_raw.astype(str), y_pred_raw.astype(str)]))
    le = LabelEncoder()
    le.fit(all_labels)
    
    y_true = le.transform(y_true_raw.astype(str))
    y_pred = le.transform(y_pred_raw.astype(str))

    # ─── Compute fairness ───
    fairness_result = compute_fairness(y_true, y_pred, sensitive)


    # ─── Policy evaluation (fairness checks only) ───
    policy_result = evaluate_policy(fairness=fairness_result)

    # ─── Log to enterprise stream ───
    log_action(db, auth, "fairness.analyze", "fairness", None, {
        "sensitive_column": sensitive_column,
        "spd": fairness_result["statistical_parity_diff"],
        "dir": fairness_result["disparate_impact_ratio"],
        "eod": fairness_result["equal_opportunity_diff"],
        "fairness_flag": fairness_result["fairness_flag"],
    })

    return {
        "fairness": fairness_result,
        "policy": policy_result,
        "sensitive_column": sensitive_column,
        "n_samples": len(y_true),
        "n_groups": len(fairness_result.get("group_metrics", {})),
    }


@router.get("/fairness/{job_id}")
async def get_fairness_results(job_id: str, db: AsyncSession = Depends(get_db)):
    """Get fairness results for a specific job (legacy compatibility)."""
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    result = (await db.execute(select(FairnessResult).filter(FairnessResult.job_id == job_id))).scalars().first()
    if not result:
        return {"status": job.status, "error": job.error, "result": None}

    return {
        "status": job.status,
        "result": {
            "metrics": result.computed_metrics_json,
            "severity_counts": result.severity_counts,
            "module_status": result.status
        }
    }
