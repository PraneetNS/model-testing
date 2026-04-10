import io
import os
import sys
import uuid
import hashlib
import tempfile
import numpy as np
import pandas as pd
import joblib
import logging
import time
from datetime import datetime

# ML Guard core path injection
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if _repo_root not in sys.path:
    sys.path.append(_repo_root)

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Body, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.session import get_db
from app.db.models import (
    Job, PreflightResult, DriftResult, PerformanceResult, 
    GovernanceResult, ScanRecord, Model, AuditLog,
    ModelVersion, ExplainabilityResult, Dataset, DatasetVersion, LineageLink, Experiment
)
from app.core.auth import AuthContext, require_role, log_action

# New Lifecycle Modules
try:
    from ml_guard.core.explainability import run_explainability
    from ml_guard.core.model_security import run_security_checks
    _has_lifecycle_core = True
except ImportError:
    _has_lifecycle_core = False

from app.domain.services.risk_engine import RiskEngine
from app.domain.services.drift_engine import DriftEngine
from app.domain.services.governance_engine import GovernanceEngine
from app.core.config import settings

storage_service = None
try:
    from app.services.storage_service import (
        upload_model as minio_upload_model,
        upload_dataset as minio_upload_dataset,
        check_storage_health,
    )
    storage_service = True
except ImportError:
    storage_service = None

logger = logging.getLogger(__name__)

# Core imports
from ml_guard.core.calibration import compute_calibration
from ml_guard.core.leakage import detect_leakage
from ml_guard.core.advisory import generate_advisories
from ml_guard.core.governance_score import compute_governance_score, compute_model_fingerprint, compute_model_complexity
from ml_guard.core.policy import evaluate_policy
from ml_guard.core.metrics import compute_accuracy, compute_f1
from ml_guard.core import ONNXModelWrapper

router = APIRouter()


# ─────────────────────────────────────────────
# UTILITY: Load model with fingerprint
# ─────────────────────────────────────────────
def _load_model_with_fingerprint(model_file: UploadFile):
    fingerprint = compute_model_fingerprint(model_file.file)
    model_file.file.seek(0)
    
    suffix = ".pkl"
    if model_file.filename.endswith(".onnx"):
        suffix = ".onnx"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        chunk = model_file.file.read(8192)
        while chunk:
            tmp.write(chunk)
            chunk = model_file.file.read(8192)
        tmp_path = tmp.name
        
    try:
        if suffix == ".onnx":
            model = ONNXModelWrapper(tmp_path)
        else:
            model = joblib.load(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return model, fingerprint


# ─────────────────────────────────────────────
# UTILITY: Extract model metadata
# ─────────────────────────────────────────────
def _extract_metadata(model) -> dict:
    framework = type(model).__module__.split(".")[0]
    task = "unknown"
    if hasattr(model, "_estimator_type"):
        task = model._estimator_type
    elif hasattr(model, "predict_proba"):
        task = "classifier"
    elif hasattr(model, "predict"):
        task = "regressor"

    meta = {
        "model_class": type(model).__name__,
        "framework": framework,
        "task": task,
        "n_features_in": int(model.n_features_in_) if getattr(model, "n_features_in_", None) is not None else None,
        "n_estimators": int(model.n_estimators) if hasattr(model, "n_estimators") else None,
        "classes": model.classes_.tolist() if hasattr(model, "classes_") else None,
        "feature_importances": model.feature_importances_.tolist() if hasattr(model, "feature_importances_") else None,
        "is_onnx": isinstance(model, ONNXModelWrapper),
        "params": {},
    }
    try:
        params = model.get_params()
        meta["params"] = {k: v if isinstance(v, (int, float, str, bool, type(None))) else str(v) for k, v in params.items()}
    except Exception:
        pass
    return meta


# ─────────────────────────────────────────────
# UTILITY: Dataset summary
# ─────────────────────────────────────────────
def _dataset_summary(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(exclude=[np.number]).columns.tolist()
    total_cells = df.shape[0] * df.shape[1]
    missing = int(df.isnull().sum().sum())
    cols = {}
    for col in df.columns:
        s = df[col]
        entry = {"dtype": str(s.dtype), "missing_pct": round(float(s.isnull().mean() * 100), 2)}
        if col in numeric:
            entry.update({"min": float(s.min()), "max": float(s.max()), "mean": round(float(s.mean()), 4), "std": round(float(s.std()), 4)})
        else:
            entry.update({"unique": int(s.nunique()), "top": str(s.mode()[0]) if len(s.mode()) > 0 else None})
        cols[col] = entry
    return {
        "n_rows": df.shape[0], "n_cols": df.shape[1],
        "n_numeric": len(numeric), "n_categorical": len(categorical),
        "missing_pct_global": round(missing / total_cells * 100, 2) if total_cells > 0 else 0,
        "columns": cols,
    }


def _fingerprint_data(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ════════════════════════════════════════════
# ENDPOINT 1: Instant model metadata
# ════════════════════════════════════════════
@router.post("/audit/inspect-model")
async def inspect_model(model_file: UploadFile = File(...), auth: AuthContext = Depends(require_role("viewer"))):
    if not model_file.filename.lower().endswith((".pkl", ".joblib", ".onnx")):
        raise HTTPException(400, "Only .pkl, .joblib, or .onnx accepted.")
    model, fingerprint = _load_model_with_fingerprint(model_file)
    meta = _extract_metadata(model)
    complexity = compute_model_complexity(model)
    return {
        "status": "ok",
        "model_metadata": meta,
        "complexity": complexity,
        "fingerprint": fingerprint,
    }


# ════════════════════════════════════════════
# ENDPOINT 2: Instant dataset summary
# ════════════════════════════════════════════
@router.post("/audit/dataset-summary")
async def dataset_summary(csv_file: UploadFile = File(...), auth: AuthContext = Depends(require_role("viewer"))):
    if not csv_file.filename.endswith(".csv"):
        raise HTTPException(400, "Only .csv accepted.")
    try:
        # Read a portion of the CSV for summary or use streaming if possible
        # For simplicity and small datasets, we can still load, but let's be safe
        df = pd.read_csv(csv_file.file)
        csv_file.file.seek(0)
    except (UnicodeDecodeError, pd.errors.ParserError):
        try:
            csv_file.file.seek(0)
            df = pd.read_csv(csv_file.file, encoding='latin-1')
            csv_file.file.seek(0)
        except Exception as e:
            raise HTTPException(422, f"CSV parse failed (encoding issue): {e}")
    return {"status": "ok", "dataset_summary": _dataset_summary(df)}


# ════════════════════════════════════════════
# ENDPOINT 3: Full Audit
# ════════════════════════════════════════════
@router.post("/audit/run")
async def run_audit(
    model_name: str = Form("CreditRiskDetector"),
    label_col: str = Form("target"),
    model_file: UploadFile = File(...),
    train_file: UploadFile = File(None),
    val_file: UploadFile = File(None),
    train_dataset_url: str = Form(None),
    val_dataset_url: str = Form(None),
    selected: list = Form(["drift", "performance", "fairness", "security"]),
    policy_override: str = Form(None),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer"))
):
    from app.services.storage_service import download_from_url
    # ─── Resolve or Create Model for Audit History ───
    from app.db.models import Model, Project
    model = (await db.execute(select(Model).filter(Model.name == model_name))).scalars().first()
    
    if not model:
        # Check for a default CI/CD project or create one
        project = (await db.execute(select(Project).filter(Project.name == "CI/CD Audits"))).scalars().first()
        if not project:
            project = Project(name="CI/CD Audits", org_id=auth.org_id)
            db.add(project)
            db.flush()
            
        model = Model(name=model_name, project_id=project.id, created_by=auth.user_id)
        db.add(model)
        db.flush()

    import uuid
    submission_token = str(uuid.uuid4())
    # Create Job record
    job = Job(model_id=model.id, status="PENDING", submission_token=submission_token)
    db.add(job)
    await db.commit() # Must commit so worker and status API see it
    await db.refresh(job)
    job_id = str(job.id)

    # --- 4. Encode Data for direct transfer to Worker ---
    import base64
    
    m_b64 = base64.b64encode(model_file.file.read()).decode("utf-8")
    
    # Validation Data
    if val_file:
        v_b64 = base64.b64encode(val_file.file.read()).decode("utf-8")
    elif val_dataset_url:
        v_b64 = base64.b64encode(download_from_url(val_dataset_url)).decode("utf-8")
    else:
        raise HTTPException(400, "Provide either a validation file or a validation dataset URL.")

    # Training Data
    if train_file:
        t_b64 = base64.b64encode(train_file.file.read()).decode("utf-8")
    elif train_dataset_url:
        t_b64 = base64.b64encode(download_from_url(train_dataset_url)).decode("utf-8")
    else:
        t_b64 = v_b64 # Use validation as surrogate

    # Dispatch task
    from app.workers.tasks import run_governance_audit_task
    run_governance_audit_task.delay(
        job_id=job_id,
        model_id=str(model.id),
        checks=selected,
        model_b64=m_b64,
        train_b64=t_b64,
        val_b64=v_b64,
        model_filename=model_file.filename,
        train_filename=train_file.filename if train_file else (train_dataset_url if train_dataset_url else "train.csv"),
        val_filename=val_file.filename if val_file else (val_dataset_url if val_dataset_url else "val.csv"),
        label_col=label_col,
        user_id=auth.user_id if hasattr(auth, "user_id") else None,
        org_id=auth.org_id if hasattr(auth, "org_id") else None,
        policy_override=policy_override
    )

    return {
        "job_id": job_id,
        "submission_token": submission_token,
        "poll_url": f"/api/v1/gate/result/{submission_token}",
        "message": "Governance audit dispatched. Transferring data via secure message queue."
    }


# ════════════════════════════════════════════
# ENDPOINT 4: Policy config preview
# ════════════════════════════════════════════
@router.get("/audit/default-policy")
async def get_default_policy():
    from ml_guard.core.policy import DEFAULT_POLICY
    return {"policy": DEFAULT_POLICY}
# ════════════════════════════════════════════
# ENDPOINT 5: Latest Security Scans
# ════════════════════════════════════════════
@router.get("/security/scans")
async def get_security_scans(
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("viewer")),
):
    """Fetch latest scan records that contain security audit data."""
    scans = db.query(ScanRecord).filter(
        ScanRecord.security_checks.isnot(None)
    ).order_by(ScanRecord.created_at.desc()).limit(limit).all()
    
    def _extract_sec_metric(checks, key, field, default):
        if not checks: return default
        if isinstance(checks, dict):
            return checks.get(key, {}).get(field, default)
        if isinstance(checks, list):
            # Fallback for list format: find test by name or partial name
            for item in checks:
                name = item.get("test_name", "").lower()
                if key.replace("_", " ") in name:
                    return item.get(field, default)
        return default

    return [
        {
            "scan_id": str(s.id),
            "model_id": str(s.model_id),
            "created_at": str(s.created_at),
            "security_audit_results": {
                "results": [
                    {
                        "test_name": "Data Poisoning",
                        "score": _extract_sec_metric(s.security_checks, "data_poisoning", "score", 0),
                        "risk_level": _extract_sec_metric(s.security_checks, "data_poisoning", "risk", "LOW"),
                        "status": _extract_sec_metric(s.security_checks, "data_poisoning", "status", "PASS")
                    },
                    {
                        "test_name": "Extraction Vulnerability",
                        "score": _extract_sec_metric(s.security_checks, "extraction_vulnerability", "score", 0),
                        "risk_level": _extract_sec_metric(s.security_checks, "extraction_vulnerability", "risk", "LOW"),
                        "status": _extract_sec_metric(s.security_checks, "extraction_vulnerability", "status", "PASS")
                    },
                    {
                        "test_name": "Membership Inference",
                        "score": _extract_sec_metric(s.security_checks, "membership_inference", "score", 0),
                        "risk_level": _extract_sec_metric(s.security_checks, "membership_inference", "risk", "LOW"),
                        "status": _extract_sec_metric(s.security_checks, "membership_inference", "status", "PASS")
                    }
                ]
            }, 
            "governance_score": s.governance_score,
            "risk_level": s.risk_level,
        }
        for s in scans
    ]
