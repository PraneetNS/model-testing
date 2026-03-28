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

def _background_ingest(req: PredictRequest, db: Session, log_id: str) -> None:
    """Non-blocking write executed in FastAPI BackgroundTasks."""
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


# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/predict", status_code=202)
async def ingest_single_prediction(
    req: PredictRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Accepts a single prediction and writes it asynchronously.
    Returns immediately with a log_id for tracking.
    """
    log_id = str(uuid.uuid4())
    background_tasks.add_task(_background_ingest, req, db, log_id)
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
