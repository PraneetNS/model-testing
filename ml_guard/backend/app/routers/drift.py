"""
drift.py — Real-time drift ingestion and analysis router.

Endpoints:
  POST /drift/{model_id}/ingest     — ingest a prediction batch and compute drift
  GET  /drift/{model_id}/report     — retrieve the latest drift report
  GET  /drift/{model_id}/history    — paginated drift report history
  POST /drift/{model_id}/trigger    — manually trigger a drift scan
  GET  /drift/{job_id}              — get drift job result (legacy)
  POST /drift/{model_id}/embedding-ingest — ingest embedding batch
  GET  /drift/{model_id}/embedding-report — embedding-based drift report
  GET  /drift/health                — module health
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import structlog
from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Path, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import desc

from app.db.session import get_db
from app.db.models import (
    DriftReport, DriftResult, EmbeddingBatch, Job, Model, PredictionLog, utcnow,
)
from app.core.auth import AuthContext, get_auth_context

router = APIRouter()
logger = structlog.get_logger(__name__)


# ─── Schemas ─────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    """Prediction batch to ingest and diff against reference window."""
    predictions: List[Dict[str, Any]]           # list of feature dicts
    reference_window_hours: int = 24            # hours back for reference
    current_window_hours: int = 1               # hours of current window
    method: str = "ks"                          # ks | psi | chi2
    alert_threshold: float = 0.20               # PSI / KS threshold


class EmbeddingIngestRequest(BaseModel):
    batch_id: str
    embeddings: List[List[float]]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _compute_psi(ref: list, cur: list, n_bins: int = 10) -> float:
    """Population Stability Index between two numeric lists."""
    import numpy as np
    eps = 1e-6
    if len(ref) < 5 or len(cur) < 5:
        return 0.0
    ref_arr = np.array(ref, dtype=float)
    cur_arr = np.array(cur, dtype=float)
    bins = np.histogram_bin_edges(ref_arr, bins=n_bins)
    bins[0] -= eps
    bins[-1] += eps
    ref_counts = np.histogram(ref_arr, bins=bins)[0]
    cur_counts = np.histogram(cur_arr, bins=bins)[0]
    ref_pct = (ref_counts + eps) / (len(ref_arr) + eps * n_bins)
    cur_pct = (cur_counts + eps) / (len(cur_arr) + eps * n_bins)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _compute_ks(ref: list, cur: list) -> float:
    """KS statistic (max absolute CDF difference)."""
    try:
        from scipy import stats
        stat, _ = stats.ks_2samp(ref, cur)
        return float(stat)
    except ImportError:
        # Pure-Python fallback
        import numpy as np
        ref_s = sorted(ref)
        cur_s = sorted(cur)
        all_v = sorted(set(ref_s + cur_s))
        ref_cdf = [sum(1 for x in ref_s if x <= v) / len(ref_s) for v in all_v]
        cur_cdf = [sum(1 for x in cur_s if x <= v) / len(cur_s) for v in all_v]
        return max(abs(r - c) for r, c in zip(ref_cdf, cur_cdf))


def _severity(score: float, method: str = "ks") -> str:
    if method == "ks":
        if score >= 0.30:
            return "CRITICAL"
        if score >= 0.20:
            return "WARNING"
        return "OK"
    else:  # PSI
        if score >= 0.25:
            return "CRITICAL"
        if score >= 0.10:
            return "WARNING"
        return "OK"


async def _run_drift_analysis(
    model_id: str,
    method: str,
    reference_window_hours: int,
    current_window_hours: int,
    alert_threshold: float,
    db: AsyncSession,
) -> Optional[DriftReport]:
    """
    Core drift analysis: pulls PredictionLogs for reference and current windows,
    computes KS or PSI per numeric feature, and writes a DriftReport row.
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    current_start = now - timedelta(hours=current_window_hours)
    reference_start = now - timedelta(hours=reference_window_hours + current_window_hours)
    reference_end = now - timedelta(hours=current_window_hours)

    # Load reference predictions
    ref_stmt = (
        select(PredictionLog)
        .where(PredictionLog.model_id == model_id)
        .where(PredictionLog.timestamp >= reference_start)
        .where(PredictionLog.timestamp < reference_end)
        .limit(5_000)
    )
    ref_rows = (await db.execute(ref_stmt)).scalars().all()

    # Load current predictions
    cur_stmt = (
        select(PredictionLog)
        .where(PredictionLog.model_id == model_id)
        .where(PredictionLog.timestamp >= current_start)
        .limit(2_000)
    )
    cur_rows = (await db.execute(cur_stmt)).scalars().all()

    if len(ref_rows) < 5 or len(cur_rows) < 5:
        logger.info(
            "drift_skipped_insufficient_data",
            model_id=model_id,
            ref_count=len(ref_rows),
            cur_count=len(cur_rows),
        )
        return None

    # Identify numeric features across reference window
    sample_features = next((r.features for r in ref_rows if r.features), {})
    numeric_keys = [
        k for k, v in sample_features.items()
        if isinstance(v, (int, float))
    ]

    feature_results = []
    drift_scores = []

    for feat in numeric_keys:
        ref_vals = [r.features.get(feat) for r in ref_rows if r.features and r.features.get(feat) is not None]
        cur_vals = [r.features.get(feat) for r in cur_rows if r.features and r.features.get(feat) is not None]

        if len(ref_vals) < 5 or len(cur_vals) < 5:
            continue

        if method == "psi":
            score = _compute_psi(ref_vals, cur_vals)
        else:
            score = _compute_ks(ref_vals, cur_vals)

        sev = _severity(score, method)
        drift_scores.append(score)

        feature_results.append({
            "feature": feat,
            "method": method,
            "score": round(score, 6),
            "severity": sev,
            "drifted": score >= alert_threshold,
            "ref_count": len(ref_vals),
            "cur_count": len(cur_vals),
        })

    overall_score = max(drift_scores) if drift_scores else 0.0
    drift_detected = overall_score >= alert_threshold
    alert_triggered = overall_score >= alert_threshold

    report = DriftReport(
        model_id=model_id,
        reference_window_start=reference_start,
        reference_window_end=reference_end,
        current_window_start=current_start,
        current_window_end=now,
        feature_results=feature_results,
        overall_drift_score=round(overall_score, 6),
        drift_detected=drift_detected,
        method=method,
        sample_count=len(ref_rows) + len(cur_rows),
        alert_triggered=alert_triggered,
    )
    db.add(report)
    await db.commit()
    await db.refresh(report)

    logger.info(
        "drift_report_created",
        model_id=model_id,
        drift_detected=drift_detected,
        overall_score=overall_score,
        features_analyzed=len(feature_results),
    )

    # ── Publish internal alert if drift detected ─────────────────────────────
    if alert_triggered:
        try:
            from app.db.models import AlertEvent, AlertRule
            active_rules = (
                await db.execute(
                    select(AlertRule)
                    .where(AlertRule.is_active == True)
                )
            ).scalars().all()
            for rule in active_rules:
                cond = rule.condition or {}
                if cond.get("metric") == "drift_score":
                    threshold = cond.get("value", 0.20)
                    if overall_score >= threshold:
                        event = AlertEvent(
                            rule_id=rule.id,
                            severity="HIGH" if overall_score >= 0.30 else "MEDIUM",
                            message=(
                                f"Drift detected on model {model_id}: "
                                f"score={overall_score:.4f} (method={method})"
                            ),
                        )
                        db.add(event)
            await db.commit()
        except Exception as e:
            logger.error("drift_alert_failed", error=str(e))

    return report


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/drift/health")
async def drift_health():
    return {"module": "drift", "status": "active", "version": "7.2.0"}


@router.post("/drift/{model_id}/ingest")
async def ingest_predictions_and_drift(
    model_id: str = Path(...),
    req: IngestRequest = Body(...),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
):
    """
    Ingest a batch of predictions into PredictionLog and trigger drift analysis.
    Returns the computed DriftReport immediately (synchronous for small batches).
    """
    # Verify model exists
    model = await db.get(Model, model_id)
    if not model:
        # Allow string model_id that doesn't link to models table (SDK usage)
        pass

    # Persist predictions
    inserted = 0
    for pred in req.predictions:
        log = PredictionLog(
            model_id=model_id,
            timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
            features=pred.get("features"),
            prediction=str(pred.get("prediction", "")),
            prediction_proba=pred.get("prediction_proba"),
            confidence=pred.get("confidence"),
            latency_ms=pred.get("latency_ms"),
            data_source=pred.get("data_source", "api"),
            environment=pred.get("environment", "production"),
        )
        db.add(log)
        inserted += 1

    await db.commit()

    # Run drift analysis
    report = await _run_drift_analysis(
        model_id=model_id,
        method=req.method,
        reference_window_hours=req.reference_window_hours,
        current_window_hours=req.current_window_hours,
        alert_threshold=req.alert_threshold,
        db=db,
    )

    if report is None:
        return {
            "model_id": model_id,
            "predictions_ingested": inserted,
            "drift_report": None,
            "message": "Not enough historical data for drift analysis yet.",
        }

    return {
        "model_id": model_id,
        "predictions_ingested": inserted,
        "drift_report": {
            "id": str(report.id),
            "overall_drift_score": report.overall_drift_score,
            "drift_detected": report.drift_detected,
            "method": report.method,
            "features_analyzed": len(report.feature_results),
            "feature_results": report.feature_results,
            "sample_count": report.sample_count,
            "created_at": report.created_at.isoformat(),
            "alert_triggered": report.alert_triggered,
        },
    }


@router.post("/drift/{model_id}/trigger")
async def trigger_drift_scan(
    model_id: str = Path(...),
    method: str = Query("ks"),
    reference_hours: int = Query(24),
    current_hours: int = Query(1),
    threshold: float = Query(0.20),
    background_tasks: BackgroundTasks = None,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
):
    """Manually trigger a drift scan for a model. Runs synchronously."""
    report = await _run_drift_analysis(
        model_id=model_id,
        method=method,
        reference_window_hours=reference_hours,
        current_window_hours=current_hours,
        alert_threshold=threshold,
        db=db,
    )
    if report is None:
        return {
            "model_id": model_id,
            "status": "insufficient_data",
            "message": "Not enough prediction data in the reference window.",
        }
    return {
        "model_id": model_id,
        "status": "completed",
        "report_id": str(report.id),
        "overall_drift_score": report.overall_drift_score,
        "drift_detected": report.drift_detected,
        "alert_triggered": report.alert_triggered,
        "computed_at": report.created_at.isoformat(),
    }


@router.get("/drift/{model_id}/report")
async def get_latest_drift_report(
    model_id: str = Path(...),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
):
    """Return the most recent DriftReport for a model."""
    stmt = (
        select(DriftReport)
        .where(DriftReport.model_id == model_id)
        .order_by(desc(DriftReport.created_at))
        .limit(1)
    )
    report = (await db.execute(stmt)).scalars().first()
    if not report:
        raise HTTPException(status_code=404, detail="No drift reports found for this model.")

    return {
        "id": str(report.id),
        "model_id": report.model_id,
        "overall_drift_score": report.overall_drift_score,
        "drift_detected": report.drift_detected,
        "alert_triggered": report.alert_triggered,
        "method": report.method,
        "sample_count": report.sample_count,
        "feature_results": report.feature_results,
        "reference_window_start": report.reference_window_start.isoformat() if report.reference_window_start else None,
        "reference_window_end": report.reference_window_end.isoformat() if report.reference_window_end else None,
        "current_window_start": report.current_window_start.isoformat() if report.current_window_start else None,
        "current_window_end": report.current_window_end.isoformat() if report.current_window_end else None,
        "created_at": report.created_at.isoformat(),
    }


@router.get("/drift/{model_id}/history")
async def get_drift_history(
    model_id: str = Path(...),
    limit: int = Query(30, le=200),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
):
    """Return paginated drift report history for a model."""
    stmt = (
        select(DriftReport)
        .where(DriftReport.model_id == model_id)
        .order_by(desc(DriftReport.created_at))
        .limit(limit)
    )
    reports = (await db.execute(stmt)).scalars().all()
    return [
        {
            "id": str(r.id),
            "overall_drift_score": r.overall_drift_score,
            "drift_detected": r.drift_detected,
            "method": r.method,
            "sample_count": r.sample_count,
            "created_at": r.created_at.isoformat(),
        }
        for r in reports
    ]


# ─── Legacy Endpoints (kept for backward compatibility) ───────────────────────

@router.get("/drift/{job_id}/result")
async def get_drift_job_result(job_id: str, db: AsyncSession = Depends(get_db)):
    """Legacy: get drift result from job ID."""
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    result = (
        await db.execute(select(DriftResult).where(DriftResult.job_id == job.id))
    ).scalars().first()

    return {
        "job_id": str(job.id),
        "status": job.status,
        "error": job.error,
        "result": result.computed_metrics_json if result else None,
    }


@router.post("/drift/{model_id}/embedding-ingest")
async def ingest_embedding_batch(
    model_id: str = Path(...),
    req: EmbeddingIngestRequest = Body(...),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
):
    """Store a batch of embeddings for later drift comparison."""
    batch = EmbeddingBatch(
        model_id=model_id,
        batch_id=req.batch_id,
        embeddings=req.embeddings,
    )
    db.add(batch)
    await db.commit()
    return {
        "model_id": model_id,
        "batch_id": req.batch_id,
        "embedding_count": len(req.embeddings),
        "status": "stored",
    }


@router.get("/drift/{model_id}/embedding-report")
async def get_embedding_drift_report(
    model_id: str = Path(...),
    batch_limit: int = Query(2, ge=2, le=10),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
):
    """Compute drift between the last N embedding batches using cosine distance."""
    stmt = (
        select(EmbeddingBatch)
        .where(EmbeddingBatch.model_id == model_id)
        .order_by(desc(EmbeddingBatch.timestamp))
        .limit(batch_limit)
    )
    batches = (await db.execute(stmt)).scalars().all()

    if len(batches) < 2:
        raise HTTPException(
            status_code=404,
            detail="At least 2 embedding batches required for drift comparison.",
        )

    try:
        import numpy as np

        def centroid(embeds: list) -> np.ndarray:
            return np.mean(np.array(embeds), axis=0)

        ref_batch = batches[-1]
        cur_batch = batches[0]

        ref_c = centroid(ref_batch.embeddings)
        cur_c = centroid(cur_batch.embeddings)

        cos_sim = float(
            np.dot(ref_c, cur_c) / (np.linalg.norm(ref_c) * np.linalg.norm(cur_c) + 1e-9)
        )
        drift_score = 1.0 - cos_sim
        drift_detected = drift_score > 0.15

        return {
            "model_id": model_id,
            "reference_batch_id": ref_batch.batch_id,
            "current_batch_id": cur_batch.batch_id,
            "cosine_distance": round(drift_score, 6),
            "drift_detected": drift_detected,
            "batches_compared": len(batches),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding drift computation failed: {e}")
