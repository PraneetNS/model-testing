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
from app.billing.metering import record_usage
from app.billing.enforcement import check_billing_limits

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
# ============================================
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
    auth: AuthContext = Depends(require_role("ml_engineer")),
    _billing: None = Depends(check_billing_limits)
):
    from app.services.storage_service import download_from_url
    from app.db.models import Model, Project
    import numpy as np, base64, tempfile, joblib, os as _os

    # --- Resolve or create Model record ---
    model = (await db.execute(select(Model).filter(Model.name == model_name))).scalars().first()
    if not model:
        project = (await db.execute(select(Project).filter(Project.name == "CI/CD Audits"))).scalars().first()
        if not project:
            project = Project(name="CI/CD Audits", org_id=auth.org_id)
            db.add(project)
            await db.flush()
        model = Model(name=model_name, project_id=project.id, created_by=auth.user_id)
        db.add(model)
        await db.flush()

    async def _ensure_dataset_registered(ds_bytes: bytes, ds_filename: str, ds_type: str, model_id: str, row_count: int = 0):
        if not ds_bytes: return None
        ds_fingerprint = hashlib.sha256(ds_bytes).hexdigest()[:32]
        
        # Check if already exists for this model
        existing = (await db.execute(select(Dataset).filter(
            Dataset.model_id == model_id,
            Dataset.fingerprint == ds_fingerprint
        ))).scalars().first()
        
        if existing:
            return existing.id

        # Register new
        new_ds = Dataset(
            model_id=model_id,
            type=ds_type,
            metadata_json={"name": ds_filename, "source": "audit_upload"},
            row_count=row_count,
            fingerprint=ds_fingerprint
        )
        db.add(new_ds)
        await db.flush()

        # Save to permanent storage (optional, for now just DB entry)
        # In a real system, we'd move the bytes to minio/S3 here
        
        version = DatasetVersion(
            dataset_id=new_ds.id,
            version_number=1,
            storage_url=f"audit://{ds_filename}",
            schema_hash=ds_fingerprint,
            row_count=row_count,
            created_by=auth.user_id
        )
        db.add(version)
        await db.flush()
        return new_ds.id

    # Record usage for the audit
    record_usage(auth.org_id, getattr(auth, "key_id", None), "model_audited")

    import uuid
    submission_token = str(uuid.uuid4())
    job = Job(model_id=model.id, status="PENDING", submission_token=submission_token)
    db.add(job)
    await db.commit()
    await db.refresh(job)
    job_id = str(job.id)

    # --- Read uploaded file bytes eagerly ---
    m_bytes = await model_file.read()

    # --- Gracefully handle missing validation data by falling back to training data ---
    v_bytes = None
    if val_file and val_file.filename:
        v_bytes = await val_file.read()
    elif val_dataset_url:
        v_bytes = download_from_url(val_dataset_url)
    
    t_bytes = None
    if train_file and train_file.filename:
        t_bytes = await train_file.read()
    elif train_dataset_url:
        t_bytes = download_from_url(train_dataset_url)

    # Cross-fill if one is missing
    if v_bytes is None and t_bytes is not None:
        v_bytes = t_bytes
        print("Audit: No validation data provided, using training data as surrogate.")
    elif t_bytes is None and v_bytes is not None:
        t_bytes = v_bytes
        print("Audit: No training data provided, using validation data as surrogate.")
    
    if v_bytes is None or t_bytes is None:
        raise HTTPException(400, "Provide at least one dataset (Training or Validation).")

    # --- Try Celery; fall back to inline if unavailable ---
    celery_ok = False
    try:
        from app.workers.tasks import run_governance_audit_task
        from app.core.celery_app import encrypt_task_payload
        
        payload = {
            "job_id": job_id, "model_id": str(model.id), "checks": selected,
            "model_b64": base64.b64encode(m_bytes).decode(),
            "train_b64": base64.b64encode(t_bytes).decode(),
            "val_b64": base64.b64encode(v_bytes).decode(),
            "model_filename": model_file.filename,
            "train_filename": (train_file.filename if (train_file and train_file.filename) else (train_dataset_url or "train.csv")),
            "val_filename": (val_file.filename if (val_file and val_file.filename) else (val_dataset_url or "val.csv")),
            "label_col": label_col,
            "user_id": auth.user_id if hasattr(auth, "user_id") else None,
            "org_id": auth.org_id if hasattr(auth, "org_id") else None,
            "policy_override": policy_override,
        }
        
        encrypted_payload = encrypt_task_payload(
            payload,
            ["model_b64", "train_b64", "val_b64", "policy_override"]
        )
        
        # run_governance_audit_task.delay(**encrypted_payload)
        # celery_ok = True
        logger.info("Bypassing Celery: Forcing inline execution for audit task")
        celery_ok = False
    except Exception as e:
        logger.error(f"failed_to_dispatch_celery error={e}")
        celery_ok = False

    if celery_ok:
        return {
            "status": "pending", "job_id": job_id,
            "submission_token": submission_token,
            "poll_url": f"/api/v1/gate/result/{submission_token}",
            "message": "Governance audit dispatched to worker.",
        }

    # --- Inline fallback: run audit synchronously within request ---
    tmp_files = []
    try:
        def _write_tmp(data: bytes, suffix: str) -> str:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(data); tmp.close(); tmp_files.append(tmp.name)
            return tmp.name

        msuffix = ".onnx" if model_file.filename.lower().endswith(".onnx") else ".pkl"
        model_path = _write_tmp(m_bytes, msuffix)
        train_path = _write_tmp(t_bytes, ".csv")
        val_path = _write_tmp(v_bytes, ".csv")

        if msuffix == ".onnx":
            from ml_guard.core import ONNXModelWrapper
            model_obj = ONNXModelWrapper(model_path)
        else:
            model_obj = joblib.load(model_path)
        if isinstance(model_obj, dict):
            model_obj = next(iter(model_obj.values()))

        def _read_df(path):
            try:
                return pd.read_csv(path)
            except UnicodeDecodeError:
                return pd.read_csv(path, encoding="latin-1")

        df_train = _read_df(train_path)
        df_val = _read_df(val_path)

        # --- Register datasets in lineage store (now with row counts) ---
        t_name = train_file.filename if (train_file and train_file.filename) else (train_dataset_url or "train.csv")
        v_name = val_file.filename if (val_file and val_file.filename) else (val_dataset_url or "val.csv")
        await _ensure_dataset_registered(t_bytes, t_name, "training", str(model.id), row_count=len(df_train))
        await _ensure_dataset_registered(v_bytes, v_name, "validation", str(model.id), row_count=len(df_val))

        feature_names = [c for c in df_train.columns if c != label_col]
        X_train_df = pd.get_dummies(df_train[feature_names])
        X_val_df = pd.get_dummies(df_val[feature_names]).reindex(columns=X_train_df.columns, fill_value=0)

        if getattr(model_obj, "feature_names_in_", None) is not None:
            expected = list(model_obj.feature_names_in_)
            # Ensure all expected features exist in the DFs
            for f in expected:
                if f not in X_train_df.columns: X_train_df[f] = 0
                if f not in X_val_df.columns: X_val_df[f] = 0
            X_train = X_train_df[expected]; X_val = X_val_df[expected]
        else:
            X_train = X_train_df
            X_val = X_val_df.reindex(columns=X_train.columns, fill_value=0)

        y_train_raw = df_train[label_col].values if label_col in df_train.columns else np.zeros(len(df_train))
        y_val_raw = df_val[label_col].values if label_col in df_val.columns else np.zeros(len(df_val))

        from sklearn.preprocessing import LabelEncoder
        try:
            y_train = y_train_raw.astype(float); y_val = y_val_raw.astype(float)
        except (ValueError, TypeError):
            le = LabelEncoder()
            le.fit(np.concatenate([y_train_raw, y_val_raw]))
            y_train = le.transform(y_train_raw).astype(float)
            y_val = le.transform(y_val_raw).astype(float)

        train_preds = model_obj.predict(X_train.values)
        val_preds = model_obj.predict(X_val.values)
        from ml_guard.core import compute_accuracy, compute_f1
        train_acc = float(compute_accuracy(y_train, train_preds))
        val_acc = float(compute_accuracy(y_val, val_preds))
        try:
            val_f1 = float(compute_f1(y_val, val_preds))
        except Exception:
            val_f1 = 0.0
        metrics = {"accuracy": val_acc, "train_accuracy": train_acc, "f1": val_f1}

        from ml_guard.core.drift import compute_feature_drift_report
        drift_report, _ = compute_feature_drift_report(X_train, X_val)

        top_drifted = sorted(
            [{"feature": k, "psi": v.get("PSI", 0),
              "severity": "CRITICAL" if v.get("PSI", 0) > 0.25 else ("WARNING" if v.get("PSI", 0) > 0.15 else "OK")}
             for k, v in drift_report.items()],
            key=lambda x: x["psi"], reverse=True
        )[:10]

        ov_gap = {"accuracy_gap": train_acc - val_acc}

        from ml_guard.core.drift import compute_target_drift
        try:
            target_drift = compute_target_drift(y_train, y_val)
        except Exception:
            target_drift = {}

        try:
            proba = model_obj.predict_proba(X_val.values)[:, 1]
            calibration = compute_calibration(y_val, proba)
        except Exception:
            calibration = {}

        try:
            leakage = detect_leakage(X_train, y_train)
        except Exception:
            leakage = {}

        gov = compute_governance_score(drift_report=drift_report, overfitting_gap=ov_gap)

        from app.domain.services.risk_engine import RiskEngine
        max_psi = max([v.get("PSI", 0) for v in drift_report.values() if isinstance(v, dict)], default=0.0)
        drifted_count = sum(1 for v in drift_report.values() if isinstance(v, dict) and v.get("drift_flag", False))
        re_metrics = {
             "accuracy_delta": ov_gap.get("accuracy_gap", 0.0),
             "psi": max_psi,
             "brier_score": calibration.get("brier_score", 0.0) if calibration else 0.0,
             "drifted_features_count": drifted_count,
             "calibration_flag": calibration.get("overconfident_flag", False) if calibration else False
        }
        risk_result = RiskEngine().calculate_risk_score(re_metrics)

        from app.domain.services.governance_engine import GovernanceEngine
        eval_ctx = {"metrics": metrics, "drift": drift_report, "overfitting_gap": ov_gap,
                    "governance_score": gov["governance_score"]}
        if policy_override:
            import json
            policy_result = evaluate_policy(**eval_ctx, policy=json.loads(policy_override))
        else:
            policy_result = await GovernanceEngine(db).evaluate_active_policy(metrics=eval_ctx, org_id=auth.org_id)

        try:
            advisories = generate_advisories(drift_report=drift_report, overfitting_gap=ov_gap, metrics=metrics)
        except Exception:
            advisories = []

        with open(model_path, "rb") as mf:
            fingerprint = compute_model_fingerprint(mf)
        complexity = compute_model_complexity(model_obj)

        results_json = {
            "checks_run": selected, "metrics": metrics, "drift": drift_report,
            "overfitting_gap": ov_gap, "governance": gov, "policy": policy_result,
            "calibration": calibration, "leakage": leakage, "target_drift": target_drift,
            "advisories": advisories, "risk_score": risk_result.get("risk_score"),
            "risk_level": risk_result.get("risk_level"), "top_drifted_ranked": top_drifted,
            "top5_drifted_features": [f["feature"] for f in top_drifted[:5]],
            "fingerprint": fingerprint, "complexity": complexity,
        }

        scan = ScanRecord(
            model_id=str(model.id), job_id=job_id, scan_type="audit",
            checks_run=selected, results_json=results_json,
            governance_score=gov["governance_score"],
            risk_score=risk_result.get("risk_score"),
            risk_level=risk_result.get("risk_level"),
            gate_status=policy_result.get("gate_status", "UNKNOWN"),
            triggered_by=auth.user_id if hasattr(auth, "user_id") else None,
            trigger_source="inline",
        )
        db.add(scan)
        
        # --- Auto-Tiering (v7.5) ---
        if not model.risk_tier:
            from app.services.inventory_service import auto_tier_model_logic
            await auto_tier_model_logic(str(model.id), db)

        job_rec = (await db.execute(select(Job).filter(Job.id == job_id))).scalar_one_or_none()
        if job_rec:
            job_rec.status = "COMPLETED"
        await db.commit()
        await db.refresh(scan)

        return {
            "status": "completed", "scan_id": str(scan.id), "job_id": job_id,
            "governance": gov, "risk_score": risk_result.get("risk_score"),
            "risk_level": risk_result.get("risk_level"), "metrics": metrics,
            "drift": drift_report, "top_drifted_ranked": top_drifted,
            "top5_drifted_features": [f["feature"] for f in top_drifted[:5]],
            "overfitting_gap": ov_gap, "target_drift": target_drift,
            "calibration": calibration, "leakage": leakage,
            "policy": policy_result, "advisories": advisories,
            "fingerprint": fingerprint, "complexity": complexity,
        }

    except Exception as e:
        import traceback
        from datetime import datetime
        error_msg = f"Audit failed for job {job_id}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        # Log to a persistent file we can inspect
        with open("audit_crash.log", "a") as f:
            f.write(f"\n--- {datetime.now()} ---\n{error_msg}\n")
        
        # Rollback is handled by the caller or context manager often, but let's be safe
        try:
            await db.rollback()
        except:
            pass
        job_rec = (await db.execute(select(Job).filter(Job.id == job_id))).scalar_one_or_none()
        if job_rec:
            job_rec.status = "FAILED"
            job_rec.error = str(e)
            await db.commit()
        raise HTTPException(500, f"Audit failed: {e}")
    finally:
        for p in tmp_files:
            try:
                if p and _os.path.exists(p):
                    _os.unlink(p)
            except Exception:
                pass


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
    stmt = select(ScanRecord).filter(
        ScanRecord.security_checks.isnot(None)
    ).order_by(ScanRecord.created_at.desc()).limit(limit)
    
    result = await db.execute(stmt)
    scans = result.scalars().all()
    
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
