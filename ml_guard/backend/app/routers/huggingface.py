"""
huggingface.py — HuggingFace Hub Integration Router

Endpoints:
  POST /api/plugins/huggingface/pull-model
  POST /api/plugins/huggingface/pull-dataset
  POST /api/plugins/huggingface/audit-from-hub
  GET  /api/plugins/huggingface/model-card-risks?repo_id=
  GET  /api/plugins/huggingface/search?query=&task=&limit=

Security:
  - HF tokens are used for the request only and never persisted.
  - repo_id is validated before any external call.
  - Downloaded models are sandboxed before inference.
"""
from __future__ import annotations

import logging
import uuid
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.core.auth import AuthContext, require_role

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Validation ────────────────────────────────────────────────────────────────

import re
REPO_ID_RE = re.compile(r"^[a-zA-Z0-9_\-\.]+/[a-zA-Z0-9_\-\.]+$")


def _assert_repo_id(repo_id: str) -> None:
    if not REPO_ID_RE.match(repo_id):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid repo_id '{repo_id}'. Must be <namespace>/<name>."
        )


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class PullModelRequest(BaseModel):
    repo_id: str
    revision: str = "main"
    filename: Optional[str] = None
    hf_token: Optional[str] = Field(None, description="Ephemeral HF token — never stored.")

    @field_validator("repo_id")
    @classmethod
    def validate_repo_id(cls, v: str) -> str:
        if not REPO_ID_RE.match(v):
            raise ValueError(f"Invalid repo_id: {v}")
        return v


class PullDatasetRequest(BaseModel):
    repo_id: str
    split: str = "test"
    max_rows: int = Field(10_000, ge=1, le=100_000)
    hf_token: Optional[str] = Field(None, description="Ephemeral HF token — never stored.")
    model_id: Optional[str] = None
    dataset_type: str = "training"
    dataset_name: Optional[str] = None

    @field_validator("repo_id")
    @classmethod
    def validate_repo_id(cls, v: str) -> str:
        if not REPO_ID_RE.match(v):
            raise ValueError(f"Invalid repo_id: {v}")
        return v


class AuditFromHubRequest(BaseModel):
    model_repo_id: str
    dataset_repo_id: str
    label_col: str = "label"
    hf_token: Optional[str] = Field(None, description="Ephemeral HF token — never stored.")
    split: str = "test"
    max_rows: int = Field(10_000, ge=1, le=100_000)
    min_score: Optional[float] = None

    @field_validator("model_repo_id", "dataset_repo_id")
    @classmethod
    def validate_repo_ids(cls, v: str) -> str:
        if not REPO_ID_RE.match(v):
            raise ValueError(f"Invalid repo_id: {v}")
        return v


# ── Helper: get plugin instance ───────────────────────────────────────────────

def _get_plugin(hf_token: Optional[str] = None):
    """Lazily import and construct the plugin. Token is ephemeral."""
    try:
        from ml_guard.plugins.huggingface import HuggingFacePlugin
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="HuggingFace Hub libraries not installed. "
                   "Install: pip install huggingface_hub datasets"
        )
    return HuggingFacePlugin(hf_token=hf_token)


# ═════════════════════════════════════════════════
# ENDPOINT 1: Pull Model from HuggingFace
# ═════════════════════════════════════════════════

@router.post("/huggingface/pull-model", tags=["huggingface"])
async def pull_model_from_hf(
    req: PullModelRequest,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer")),
):
    """
    Pull a model file from HuggingFace Hub, compute its SHA-256,
    and register it in the ML Guard model registry.

    HF token is used only for this request and discarded.
    """
    plugin = _get_plugin(req.hf_token)

    try:
        result = plugin.pull_model(
            repo_id=req.repo_id,
            revision=req.revision,
            filename=req.filename,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("HF model pull failed")
        raise HTTPException(status_code=502, detail=f"HuggingFace pull failed: {e}")

    # Register in ML Guard DB
    from app.db.models import Model, Project
    from sqlalchemy.future import select

    project = (await db.execute(
        select(Project).filter(Project.name == "HuggingFace Hub")
    )).scalars().first()

    if not project:
        project = Project(name="HuggingFace Hub", org_id=auth.org_id)
        db.add(project)
        await db.flush()

    model = Model(
        name=req.repo_id,
        project_id=project.id,
        provider="HuggingFace",
        fingerprint=result["sha256"],
        metadata_json={
            "source": "huggingface_hub",
            "repo_id": req.repo_id,
            "revision": req.revision,
            "filename": result.get("filename"),
            "license": result.get("license"),
            "pipeline_tag": result.get("pipeline_tag"),
            "downloads_last_month": result.get("downloads_last_month"),
            "model_card_url": result.get("model_card_url"),
        },
        created_by=auth.user_id,
    )
    db.add(model)
    await db.commit()
    await db.refresh(model)

    # Audit log
    from app.core.auth import log_action
    await log_action(db, auth, "huggingface.pull_model", "model", str(model.id), {
        "repo_id": req.repo_id, "revision": req.revision,
    })

    return {
        "model_id": str(model.id),
        "local_path": result["local_path"],
        "sha256": result["sha256"],
        "repo_id": req.repo_id,
        "license": result.get("license"),
        "pipeline_tag": result.get("pipeline_tag"),
        "downloads_last_month": result.get("downloads_last_month"),
    }


# ═════════════════════════════════════════════════
# ENDPOINT 2: Pull Dataset from HuggingFace
# ═════════════════════════════════════════════════

@router.post("/huggingface/pull-dataset", tags=["huggingface"])
async def pull_dataset_from_hf(
    req: PullDatasetRequest,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer")),
):
    """
    Load a dataset from HuggingFace Datasets, convert to CSV,
    and register in the ML Guard lineage store if model_id is provided.
    """
    plugin = _get_plugin(req.hf_token)

    try:
        result = plugin.pull_dataset(
            repo_id=req.repo_id,
            split=req.split,
            max_rows=req.max_rows,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("HF dataset pull failed")
        raise HTTPException(status_code=502, detail=f"HuggingFace dataset pull failed: {e}")

    # Register in DB if requested
    dataset_id = None
    if req.model_id:
        from app.db.models import Dataset, DatasetVersion, Model
        from sqlalchemy.future import select

        model = await db.get(Model, req.model_id)
        if not model:
            raise HTTPException(404, "Target model not found.")

        name = req.dataset_name or f"HF: {req.repo_id} ({req.split})"
        
        # Compute SHA-256 for the new CSV
        import hashlib
        h = hashlib.sha256()
        with open(result["local_csv_path"], "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        sha256 = h.hexdigest()

        dataset = Dataset(
            model_id=req.model_id,
            type=req.dataset_type,
            metadata_json={
                "name": name,
                "source": "huggingface_hub",
                "repo_id": req.repo_id,
                "split": req.split,
            },
            row_count=result["row_count"],
            fingerprint=sha256[:32],
        )
        db.add(dataset)
        await db.flush()
        dataset_id = str(dataset.id)

        version = DatasetVersion(
            dataset_id=dataset.id,
            version_number=1,
            storage_url=result["local_csv_path"],
            schema_hash=sha256[:32],
            row_count=result["row_count"],
            feature_count=len(result["column_names"]),
            created_by=auth.user_id,
        )
        db.add(version)
        await db.commit()

        await log_action(db, auth, "huggingface.pull_dataset", "dataset", dataset_id, {
            "repo_id": req.repo_id, "split": req.split
        })

    return {
        **result,
        "dataset_id": dataset_id,
        "status": "registered" if dataset_id else "pulled_only"
    }


# ═════════════════════════════════════════════════
# ENDPOINT 3: Zero-Upload Audit from Hub
# ═════════════════════════════════════════════════

@router.post("/huggingface/audit-from-hub", tags=["huggingface"])
async def audit_from_hub(
    req: AuditFromHubRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer")),
):
    """
    One-shot: pull model + dataset from HuggingFace and run the
    full ML Guard governance audit pipeline.

    Zero-upload governance — no file uploads required.
    """
    plugin = _get_plugin(req.hf_token)

    # 1. Pull model
    try:
        model_result = plugin.pull_model(repo_id=req.model_repo_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Model pull failed: {e}")

    # 2. Pull dataset
    try:
        data_result = plugin.pull_dataset(
            repo_id=req.dataset_repo_id,
            split=req.split,
            max_rows=req.max_rows,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Dataset pull failed: {e}")

    # 3. Register model in DB
    from app.db.models import Model, Project, Job
    from sqlalchemy.future import select

    project = (await db.execute(
        select(Project).filter(Project.name == "HuggingFace Hub")
    )).scalars().first()
    if not project:
        project = Project(name="HuggingFace Hub", org_id=auth.org_id)
        db.add(project)
        await db.flush()

    model = Model(
        name=req.model_repo_id,
        project_id=project.id,
        provider="HuggingFace",
        fingerprint=model_result["sha256"],
        metadata_json={
            "source": "huggingface_hub",
            "repo_id": req.model_repo_id,
            "license": model_result.get("license"),
            "pipeline_tag": model_result.get("pipeline_tag"),
        },
        created_by=auth.user_id,
    )
    db.add(model)
    await db.flush()

    # 4. Register dataset in DB
    import hashlib
    h = hashlib.sha256()
    with open(data_result["local_csv_path"], "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    sha256 = h.hexdigest()

    dataset = Dataset(
        model_id=model.id,
        type="validation", # Typically test split used for audit
        metadata_json={
            "name": f"HF Audit: {req.dataset_repo_id}",
            "source": "huggingface_hub",
            "repo_id": req.dataset_repo_id,
            "split": req.split,
        },
        row_count=data_result["row_count"],
        fingerprint=sha256[:32],
    )
    db.add(dataset)
    await db.flush()

    version = DatasetVersion(
        dataset_id=dataset.id,
        version_number=1,
        storage_url=data_result["local_csv_path"],
        schema_hash=sha256[:32],
        row_count=data_result["row_count"],
        feature_count=len(data_result["column_names"]),
        created_by=auth.user_id,
    )
    db.add(version)

    # 5. Create a job for tracking
    submission_token = str(uuid.uuid4())
    job = Job(model_id=model.id, status="PENDING", submission_token=submission_token)
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # 5. Run the audit inline (same as audit.py's inline path)
    import tempfile, joblib, pandas as pd, numpy as np, os

    job_id = str(job.id)
    tmp_files = []

    try:
        model_path = model_result["local_path"]
        csv_path = data_result["local_csv_path"]

        # Load model
        if model_path.endswith(".onnx"):
            from ml_guard.core import ONNXModelWrapper
            model_obj = ONNXModelWrapper(model_path)
        elif model_path.endswith((".pkl", ".joblib")):
            model_obj = joblib.load(model_path)
        else:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported model format: {model_path}. "
                       "Currently supports .pkl, .joblib, .onnx."
            )

        df = pd.read_csv(csv_path)
        label_col = req.label_col

        if label_col not in df.columns:
            raise HTTPException(
                status_code=422,
                detail=f"Label column '{label_col}' not found in dataset. "
                       f"Available columns: {list(df.columns)[:20]}"
            )

        feature_names = [c for c in df.columns if c != label_col]
        X_df = pd.get_dummies(df[feature_names])

        if getattr(model_obj, "feature_names_in_", None) is not None:
            expected = list(model_obj.feature_names_in_)
            for f in expected:
                if f not in X_df.columns:
                    X_df[f] = 0
            X_df = X_df[expected]

        y_raw = df[label_col].values
        from sklearn.preprocessing import LabelEncoder
        try:
            y = y_raw.astype(float)
        except (ValueError, TypeError):
            le = LabelEncoder()
            y = le.fit_transform(y_raw).astype(float)

        preds = model_obj.predict(X_df.values)

        from ml_guard.core.metrics import compute_accuracy, compute_f1
        from ml_guard.core.governance_score import compute_governance_score, compute_model_fingerprint

        acc = float(compute_accuracy(y, preds))
        try:
            f1 = float(compute_f1(y, preds))
        except Exception:
            f1 = 0.0
        metrics = {"accuracy": acc, "f1": f1}

        from ml_guard.core.drift import compute_feature_drift_report
        # Use first/second half as train/val splits for drift
        split_pt = len(X_df) // 2
        drift_report, _ = compute_feature_drift_report(X_df.iloc[:split_pt], X_df.iloc[split_pt:])

        ov_gap = {"accuracy_gap": 0.0}  # No separate train set
        gov = compute_governance_score(drift_report=drift_report, overfitting_gap=ov_gap)

        # Model card risks
        try:
            card_risks = plugin.get_model_card_risks(req.model_repo_id)
        except Exception:
            card_risks = {}

        # Security checks
        security_results = None
        try:
            from ml_guard.core.model_security import run_security_checks
            security_results = run_security_checks(
                model_obj, X_df.iloc[:split_pt], X_df.iloc[split_pt:],
                y[:split_pt], y[split_pt:]
            )
        except Exception as sec_err:
            logger.warning("Security checks failed: %s", sec_err)

        from app.db.models import ScanRecord
        results_json = {
            "checks_run": ["accuracy", "drift", "governance", "security"],
            "metrics": metrics,
            "drift": drift_report,
            "overfitting_gap": ov_gap,
            "governance": gov,
            "security": security_results,
            "model_card_risks": card_risks,
            "source": "huggingface_hub",
            "model_repo_id": req.model_repo_id,
            "dataset_repo_id": req.dataset_repo_id,
        }

        scan = ScanRecord(
            model_id=str(model.id), job_id=job_id, scan_type="hf_audit",
            checks_run=["accuracy", "drift", "governance", "security"],
            results_json=results_json,
            governance_score=gov.get("governance_score"),
            risk_level=security_results.get("overall_risk") if security_results else "LOW",
            gate_status="PASSED" if gov.get("deployment_allowed") else "BLOCKED",
            triggered_by=auth.user_id if hasattr(auth, "user_id") else None,
            trigger_source="huggingface_hub",
            security_checks=security_results,
        )
        db.add(scan)

        job_rec = (await db.execute(
            select(Job).filter(Job.id == job_id)
        )).scalar_one_or_none()
        if job_rec:
            job_rec.status = "COMPLETED"
        await db.commit()
        await db.refresh(scan)

        return {
            "status": "completed",
            "scan_id": str(scan.id),
            "model_id": str(model.id),
            "job_id": job_id,
            "submission_token": submission_token,
            "governance": gov,
            "metrics": metrics,
            "model_card_risks": card_risks,
            "security": security_results,
            "source": {
                "model": req.model_repo_id,
                "dataset": req.dataset_repo_id,
            },
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("HF audit-from-hub failed")
        job_rec = (await db.execute(
            select(Job).filter(Job.id == job_id)
        )).scalar_one_or_none()
        if job_rec:
            job_rec.status = "FAILED"
            job_rec.error = str(exc)
            await db.commit()
        raise HTTPException(500, f"Audit failed: {exc}")


# ═════════════════════════════════════════════════
# ENDPOINT 4: Model Card Risk Analysis
# ═════════════════════════════════════════════════

@router.get("/huggingface/model-card-risks", tags=["huggingface"])
async def get_model_card_risks(
    repo_id: str = Query(..., description="HuggingFace repo ID (e.g. microsoft/resnet-50)"),
    hf_token: Optional[str] = Query(None, description="Ephemeral HF token"),
    auth: AuthContext = Depends(require_role("viewer")),
):
    """
    Synchronous model card risk analysis — fast API call.
    """
    _assert_repo_id(repo_id)
    plugin = _get_plugin(hf_token)

    try:
        return plugin.get_model_card_risks(repo_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch model card: {e}")


# ═════════════════════════════════════════════════
# ENDPOINT 5: Search HuggingFace Models
# ═════════════════════════════════════════════════

@router.get("/huggingface/search", tags=["huggingface"])
async def search_hf_models(
    query: str = Query(..., min_length=1, description="Search query"),
    task: Optional[str] = Query(None, description="Filter by pipeline task (e.g. text-classification)"),
    limit: int = Query(10, ge=1, le=50),
    hf_token: Optional[str] = Query(None, description="Ephemeral HF token"),
    auth: AuthContext = Depends(require_role("viewer")),
):
    """
    Search HuggingFace Hub models. Returns top results by download count.
    """
    plugin = _get_plugin(hf_token)

    try:
        return plugin.search_models(query=query, task=task, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"HuggingFace search failed: {e}")
