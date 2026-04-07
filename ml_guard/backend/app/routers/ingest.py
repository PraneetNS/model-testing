"""
ingest.py — ML Guard Prediction Ingestion Router

Provides endpoints for:
- Single prediction log (non-blocking background write)
- Batch prediction ingestion (Celery dispatched)
- Ground truth label stitching
- Recent prediction retrieval
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.models import PredictionLog
from app.db.session import get_db
from app.services.ingestion_service import ingest_single, stitch_labels

router = APIRouter()


# ─── Pydantic Schemas ────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    model_id: str
    features: Dict[str, Any] = Field(default_factory=dict)
    prediction: Any
    prediction_proba: Optional[float] = None
    latency_ms: Optional[float] = None
    data_source: str = "api"
    environment: str = "production"
    tags: Optional[Dict[str, Any]] = None


class BatchPredictRequest(BaseModel):
    rows: List[PredictRequest]


class LabelRequest(BaseModel):
    log_ids: List[str]
    ground_truths: List[Any]


# ─── Internal background write ────────────────────────────────────────────────

def _background_ingest(req: PredictRequest, log_id: str) -> None:
    """Non-blocking write executed in FastAPI BackgroundTasks."""
    from app.db.session import SessionLocal
    db = SessionLocal()
    try:
        ingest_single(
            db=db,
            model_id=req.model_id,
            features=req.features,
            prediction=req.prediction,
            prediction_proba=req.prediction_proba,
            latency_ms=req.latency_ms,
            data_source=req.data_source,
            environment=req.environment,
            tags=req.tags,
        )
    except Exception:
        pass  # Fire-and-forget; log handled inside service
    finally:
        db.close()


# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/predict", status_code=202)
async def ingest_single_prediction(
    req: PredictRequest,
    background_tasks: BackgroundTasks,
):
    """
    Accepts a single prediction and writes it asynchronously.
    Returns immediately with a log_id for tracking.
    """
    log_id = str(uuid.uuid4())
    background_tasks.add_task(_background_ingest, req, log_id)
    return {"log_id": log_id, "status": "accepted"}


@router.post("/batch", status_code=202)
async def ingest_batch_predictions(req: BatchPredictRequest):
    """
    Accepts a batch of predictions (max 10,000) and dispatches
    a Celery task for DB bulk-insert.
    """
    if len(req.rows) > 10_000:
        raise HTTPException(status_code=400, detail="Batch max is 10,000 rows.")

    rows = [r.model_dump() for r in req.rows]

    try:
        from app.core.celery_app import celery_app
        task = celery_app.send_task(
            "app.tasks.ingest.ingest_batch_task",
            args=[rows],
        )
        task_id = task.id
    except Exception:
        # Fallback: synchronous if Celery unavailable
        from app.services.ingestion_service import ingest_batch
        ingest_batch(rows)
        task_id = "sync-fallback"

    return {"task_id": task_id, "count": len(rows), "status": "dispatched"}


@router.post("/label", status_code=200)
async def add_ground_truth_labels(
    req: LabelRequest,
    db: Session = Depends(get_db),
):
    """
    Stitch ground truth labels onto existing PredictionLog rows.
    Unlocks performance metric computation for those predictions.
    """
    if len(req.log_ids) != len(req.ground_truths):
        raise HTTPException(
            status_code=400,
            detail="log_ids and ground_truths must be equal length."
        )
    updated = stitch_labels(db, req.log_ids, req.ground_truths)
    return {"updated": updated, "requested": len(req.log_ids)}


@router.get("/{model_id}/recent")
async def get_recent_predictions(
    model_id: str,
    limit: int = Query(default=50, le=500),
    environment: Optional[str] = Query(default=None),
    start: Optional[datetime] = Query(default=None),
    end: Optional[datetime] = Query(default=None),
    labeled_only: bool = Query(default=False),
    db: Session = Depends(get_db),
):
    """
    Retrieve the most recent prediction logs for a model.
    Supports filtering by environment, date range, and label status.
    """
    q = db.query(PredictionLog).filter(PredictionLog.model_id == model_id)

    if environment:
        q = q.filter(PredictionLog.environment == environment)
    if start:
        q = q.filter(PredictionLog.timestamp >= start)
    if end:
        q = q.filter(PredictionLog.timestamp <= end)
    if labeled_only:
        q = q.filter(PredictionLog.ground_truth.isnot(None))

    rows = q.order_by(PredictionLog.timestamp.desc()).limit(limit).all()

    return [
        {
            "log_id": str(r.id),
            "model_id": r.model_id,
            "timestamp": r.timestamp.isoformat(),
            "features": r.features,
            "prediction": r.prediction,
            "prediction_proba": r.prediction_proba,
            "ground_truth": r.ground_truth,
            "latency_ms": r.latency_ms,
            "data_source": r.data_source,
            "environment": r.environment,
            "tags": r.tags,
        }
        for r in rows
    ]


@router.get("/{model_id}/stats")
async def get_ingest_stats(
    model_id: str,
    window_hours: int = Query(default=24, le=168),
    db: Session = Depends(get_db),
):
    """
    Summary statistics for ingested predictions:
    count, labeled count, avg latency, environment breakdown.
    """
    from datetime import timedelta
    cutoff = datetime.utcnow() - timedelta(hours=window_hours)
    rows = (
        db.query(PredictionLog)
        .filter(PredictionLog.model_id == model_id, PredictionLog.timestamp >= cutoff)
        .all()
    )

    total = len(rows)
    labeled = sum(1 for r in rows if r.ground_truth is not None)
    latencies = [r.latency_ms for r in rows if r.latency_ms is not None]
    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else None

    env_counts: Dict[str, int] = {}
    for r in rows:
        env_counts[r.environment] = env_counts.get(r.environment, 0) + 1

    return {
        "model_id": model_id,
        "window_hours": window_hours,
        "total_predictions": total,
        "labeled_count": labeled,
        "label_coverage_pct": round(labeled / total * 100, 2) if total > 0 else 0.0,
        "avg_latency_ms": avg_latency,
        "environment_breakdown": env_counts,
    }


# ── Profile Ingestion ─────────────────────────────────────────────────────────

class ColumnProfileSchema(BaseModel):
    name: str
    dtype: str = "unknown"
    count: int = 0
    null_count: int = 0
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    p50: Optional[float] = None
    cardinality: Optional[int] = None
    top_values: Optional[Dict[str, float]] = None


class DataProfileRequest(BaseModel):
    profile_id: str
    model_id: str
    dataset_name: str = "production"
    tags: Optional[Dict[str, Any]] = None
    row_count: int = 0
    created_at: Optional[str] = None
    columns: Dict[str, Any] = Field(default_factory=dict)


@router.post("/profile", status_code=202)
async def ingest_data_profile(
    req: DataProfileRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Accept a privacy-preserving data profile from the ML Guard SDK.

    Profiles contain statistical summaries (mean, std, percentiles,
    cardinality) — never raw feature values.

    Profile data is persisted and used to:
    - Track data distribution drift over time
    - Trigger governance alerts on schema violations
    - Feed the governance scoring engine
    """
    # Store in a lightweight JSON column in DriftReport or new table
    # For now: persist as metadata in DriftReport scaffold
    def _store_profile():
        try:
            from app.db.models import DriftReport
            # Store columns summary as a compact feature_results payload
            col_summaries = []
            for col_name, col_data in req.columns.items():
                col_summaries.append({
                    "feature_name": col_name,
                    "type": col_data.get("dtype", "unknown"),
                    "null_pct": round(
                        col_data.get("null_count", 0) /
                        max(col_data.get("count", 1), 1) * 100, 2
                    ),
                    "drift_detected": False,
                    "drift_score": 0.0,
                    "source": "sdk_profile",
                    "profile_id": req.profile_id,
                    "stats": {
                        k: v for k, v in col_data.items()
                        if k in ("mean", "std", "min", "max", "p50",
                                 "p95", "cardinality", "top_values")
                    },
                })

            record = DriftReport(
                model_id=req.model_id,
                feature_results=col_summaries,
                overall_drift_score=0.0,
                drift_detected=False,
                method="sdk_profile",
                sample_count=req.row_count,
                alert_triggered=False,
            )
            db.add(record)
            db.commit()
        except Exception:
            pass  # fire-and-forget

    background_tasks.add_task(_store_profile)
    return {
        "profile_id": req.profile_id,
        "model_id": req.model_id,
        "status": "accepted",
        "columns_received": len(req.columns),
        "rows": req.row_count,
    }
