"""
observe.py — ML Guard Unified Observability Router

Exposes endpoints for:
- Global observability feed (all models command center)
- Live drift analysis and history per model
- Live performance snapshots and timeline
- Baseline management
- Governance health score (live decay)
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.models import DriftReport, PerformanceSnapshot, PredictionLog
from app.db.session import get_db
from app.services.drift_analyzer import DriftAnalyzer
from app.services.performance_tracker import PerformanceTracker

router = APIRouter()
logger = logging.getLogger(__name__)


# ─── Schemas ─────────────────────────────────────────────────────────────────

class SliceRequest(BaseModel):
    slice_feature: str
    window_hours: int = 24


class TaskTypeRequest(BaseModel):
    task_type: str  # classification | regression | ranking


# ─── MODULE 5: Live Governance Score Computation ─────────────────────────────

def compute_live_governance_score(
    model_id: str,
    db: Session,
) -> Dict[str, Any]:
    """
    Decays the last audit governance score based on observed drift
    and performance degradation.
    
    live_score = base_score × (1 - drift_penalty) × (1 - perf_penalty)
    """
    # Get last drift report
    last_drift = (
        db.query(DriftReport)
        .filter(DriftReport.model_id == model_id)
        .order_by(DriftReport.created_at.desc())
        .first()
    )

    # Get last performance snapshot
    last_perf = (
        db.query(PerformanceSnapshot)
        .filter(PerformanceSnapshot.model_id == model_id)
        .order_by(PerformanceSnapshot.computed_at.desc())
        .first()
    )

    # Try to get base audit score from scan records
    base_score = 75.0
    try:
        from app.db.models import ScanRecord
        last_scan = (
            db.query(ScanRecord)
            .filter(ScanRecord.model_id == model_id)
            .order_by(ScanRecord.created_at.desc())
            .first()
        )
        if last_scan and last_scan.governance_score:
            base_score = last_scan.governance_score
    except Exception:
        pass

    drift_penalty = 0.0
    perf_penalty = 0.0
    drift_status = "NONE"
    perf_delta = 0.0

    if last_drift:
        drift_score = last_drift.overall_drift_score or 0.0
        drift_penalty = min(0.30, drift_score)
        drift_status = "CRITICAL" if drift_score > 0.30 else "HIGH" if drift_score > 0.20 else "LOW" if drift_score > 0.10 else "NONE"

    if last_perf and last_perf.degradation_report:
        acc_delta = last_perf.degradation_report.get("accuracy", {}).get("delta", 0) or 0
        perf_penalty = max(0, -acc_delta) * 2

    live_score = max(0, base_score * (1 - drift_penalty) * (1 - perf_penalty))

    return {
        "base_audit_score": base_score,
        "drift_penalty": round(drift_penalty, 4),
        "perf_penalty": round(perf_penalty, 4),
        "live_governance_score": round(live_score, 1),
        "drift_status": drift_status,
        "perf_delta": round(perf_delta, 4),
        "last_drift_at": last_drift.created_at.isoformat() if last_drift else None,
        "last_perf_at": last_perf.computed_at.isoformat() if last_perf else None,
    }


# ─── MODULE 4: Global Observability Feed ─────────────────────────────────────

@router.get("/feed")
async def global_observability_feed(
    db: Session = Depends(get_db),
    sort_by: str = Query(default="most_degraded"),
):
    """
    Command center view: all models with health score, drift status,
    prediction volume, and last alert. Equivalent to WhyLabs home dashboard.
    """
    # Get all unique model IDs from prediction logs
    model_ids_result = db.query(PredictionLog.model_id).distinct().all()
    model_ids = [r[0] for r in model_ids_result]

    # Also pull from scan records for models without logs yet
    try:
        from app.db.models import ScanRecord, Model
        scan_model_ids = [str(r[0]) for r in db.query(ScanRecord.model_id).distinct().all()]
        model_ids = list(set(model_ids + scan_model_ids))
    except Exception:
        pass

    cutoff_24h = datetime.utcnow() - timedelta(hours=24)
    cutoff_30d = datetime.utcnow() - timedelta(days=30)

    feed = []
    for mid in model_ids:
        if not mid:
            continue
        try:
            # Prediction volume
            pred_count = (
                db.query(PredictionLog)
                .filter(PredictionLog.model_id == mid, PredictionLog.timestamp >= cutoff_24h)
                .count()
            )

            # Last drift
            last_drift = (
                db.query(DriftReport)
                .filter(DriftReport.model_id == mid)
                .order_by(DriftReport.created_at.desc())
                .first()
            )

            # Last perf
            last_perf = (
                db.query(PerformanceSnapshot)
                .filter(PerformanceSnapshot.model_id == mid)
                .order_by(PerformanceSnapshot.computed_at.desc())
                .first()
            )

            # Live score
            live_gov = compute_live_governance_score(mid, db)

            # Last alert (could be drift breach)
            alert_triggered = last_drift.alert_triggered if last_drift else False

            # Days since last audit
            days_since_audit = None
            try:
                from app.db.models import ScanRecord
                last_scan = (
                    db.query(ScanRecord)
                    .filter(ScanRecord.model_id == mid)
                    .order_by(ScanRecord.created_at.desc())
                    .first()
                )
                if last_scan:
                    days_since_audit = (datetime.utcnow() - last_scan.created_at).days
            except Exception:
                pass

            feed.append({
                "model_id": mid,
                "live_governance_score": live_gov["live_governance_score"],
                "drift_status": live_gov["drift_status"],
                "predictions_24h": pred_count,
                "last_drift_score": last_drift.overall_drift_score if last_drift else None,
                "last_drift_at": last_drift.created_at.isoformat() if last_drift else None,
                "last_perf_snapshot": last_perf.metrics if last_perf else None,
                "alert_triggered": alert_triggered,
                "days_since_last_audit": days_since_audit,
            })
        except Exception as e:
            logger.warning("feed_model_error", model_id=mid, error=str(e))
            continue

    # Sort
    if sort_by == "most_degraded":
        feed.sort(key=lambda x: x["live_governance_score"])
    elif sort_by == "highest_drift":
        feed.sort(key=lambda x: x["last_drift_score"] or 0, reverse=True)
    elif sort_by == "prediction_volume":
        feed.sort(key=lambda x: x["predictions_24h"], reverse=True)
    elif sort_by == "last_updated":
        feed.sort(key=lambda x: x["last_drift_at"] or "", reverse=True)

    return {"models": feed, "total": len(feed), "generated_at": datetime.utcnow().isoformat()}


# ─── Drift Endpoints ─────────────────────────────────────────────────────────

@router.get("/drift/{model_id}/report")
async def get_drift_report(
    model_id: str,
    window_hours: int = Query(default=24),
    method: str = Query(default="ks"),
    db: Session = Depends(get_db),
):
    """
    Run live drift analysis for the model. Returns per-feature breakdown.
    Compares current window against MinIO-stored baseline.
    """
    analyzer = DriftAnalyzer(db, model_id, method=method)
    result = analyzer.analyze(window_hours=window_hours)
    if result is None:
        return {
            "model_id": model_id,
            "status": "insufficient_data",
            "message": "Not enough predictions or baseline not yet set. Ingest more data.",
        }
    return result


@router.get("/drift/{model_id}/history")
async def get_drift_history(
    model_id: str,
    limit: int = Query(default=30, le=100),
    db: Session = Depends(get_db),
):
    """Last N drift reports with trend per feature."""
    analyzer = DriftAnalyzer(db, model_id)
    return {"model_id": model_id, "history": analyzer.get_history(limit=limit)}


@router.get("/drift/{model_id}/features")
async def get_feature_drift_timeline(
    model_id: str,
    feature: str = Query(..., description="Feature name to fetch timeline for"),
    limit: int = Query(default=30, le=100),
    db: Session = Depends(get_db),
):
    """Per-feature drift score timeline for sparkline charts."""
    analyzer = DriftAnalyzer(db, model_id)
    timeline = analyzer.get_feature_timeline(feature_name=feature, limit=limit)
    return {"model_id": model_id, "feature": feature, "timeline": timeline}


@router.post("/drift/{model_id}/set-baseline")
async def set_drift_baseline(
    model_id: str,
    window_hours: int = Query(default=168, description="Hours of data to use as new baseline"),
    db: Session = Depends(get_db),
):
    """
    Manually set the current prediction window as the new reference baseline.
    Stores as parquet in MinIO: baselines/{model_id}/reference.parquet
    """
    from app.services.ingestion_service import get_recent_predictions_df, store_baseline_to_minio
    df = get_recent_predictions_df(db, model_id, hours=window_hours)
    if df.empty:
        raise HTTPException(status_code=404, detail="No prediction data found to set as baseline.")

    skip = {"log_id", "timestamp", "prediction", "prediction_proba", "ground_truth", "latency_ms", "environment"}
    feature_cols = [c for c in df.columns if c not in skip]
    key = store_baseline_to_minio(model_id, df[feature_cols])

    return {
        "model_id": model_id,
        "status": "baseline_updated",
        "rows_used": len(df),
        "features": feature_cols,
        "storage_key": key,
    }


# ─── Performance Endpoints ────────────────────────────────────────────────────

@router.get("/performance/{model_id}/live")
async def get_live_performance(
    model_id: str,
    window_hours: int = Query(default=24),
    db: Session = Depends(get_db),
):
    """
    Compute current performance metrics from labeled predictions.
    Returns degradation delta vs baseline.
    """
    tracker = PerformanceTracker(db, model_id)

    # Load baseline from last stored snapshot
    prev_snap = (
        db.query(PerformanceSnapshot)
        .filter(PerformanceSnapshot.model_id == model_id)
        .order_by(PerformanceSnapshot.computed_at.desc())
        .offset(1)  # second-to-last as baseline reference
        .first()
    )
    baseline_metrics = prev_snap.metrics if prev_snap else None

    result = tracker.compute_snapshot(window_hours=window_hours, baseline_metrics=baseline_metrics)
    if result is None:
        return {
            "model_id": model_id,
            "status": "insufficient_labeled_data",
            "message": "Fewer than 10 labeled predictions found. Use /ingest/label to add ground truth.",
        }
    return result


@router.get("/performance/{model_id}/timeline")
async def get_performance_timeline(
    model_id: str,
    limit: int = Query(default=48, le=200),
    db: Session = Depends(get_db),
):
    """Hourly performance snapshots for timeline charts."""
    tracker = PerformanceTracker(db, model_id)
    return {"model_id": model_id, "timeline": tracker.get_timeline(limit=limit)}


@router.post("/performance/{model_id}/task-type")
async def set_task_type(
    model_id: str,
    req: TaskTypeRequest,
    db: Session = Depends(get_db),
):
    """Set classification vs regression for model performance tracking."""
    if req.task_type not in ("classification", "regression", "ranking"):
        raise HTTPException(status_code=400, detail="Must be one of: classification, regression, ranking")
    return {"model_id": model_id, "task_type": req.task_type, "status": "acknowledged"}


@router.post("/performance/{model_id}/slice")
async def get_performance_slice(
    model_id: str,
    req: SliceRequest,
    db: Session = Depends(get_db),
):
    """
    Slice analysis: compute metrics per-value of a feature.
    Reveals if model fails for specific subgroups (Arize heatmap equivalent).
    """
    tracker = PerformanceTracker(db, model_id)
    result = tracker.analyze_slice(req.slice_feature, window_hours=req.window_hours)
    return result


# ─── Model Overview Card ─────────────────────────────────────────────────────

@router.get("/{model_id}/overview")
async def get_model_overview(
    model_id: str,
    db: Session = Depends(get_db),
):
    """
    Unified overview for model observe page:
    - prediction volume 24h + sparkline
    - avg latency
    - drift status badge
    - performance delta
    - live governance score
    """
    cutoff = datetime.utcnow() - timedelta(hours=24)
    preds_24h = (
        db.query(PredictionLog)
        .filter(PredictionLog.model_id == model_id, PredictionLog.timestamp >= cutoff)
        .all()
    )

    latencies = [p.latency_ms for p in preds_24h if p.latency_ms is not None]
    avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else None

    # Prediction sparkline (hourly counts for last 24h)
    sparkline = []
    for h in range(24, 0, -1):
        h_start = datetime.utcnow() - timedelta(hours=h)
        h_end = datetime.utcnow() - timedelta(hours=h - 1)
        count = sum(1 for p in preds_24h if h_start <= p.timestamp < h_end)
        sparkline.append({"hour": h, "count": count})

    live_gov = compute_live_governance_score(model_id, db)

    # Last drift per-feature summary
    last_drift_report = (
        db.query(DriftReport)
        .filter(DriftReport.model_id == model_id)
        .order_by(DriftReport.created_at.desc())
        .first()
    )

    feature_summary = []
    if last_drift_report and last_drift_report.feature_results:
        for fr in last_drift_report.feature_results[:10]:
            feature_summary.append({
                "feature": fr.get("feature_name"),
                "type": fr.get("type"),
                "drift_score": fr.get("drift_score"),
                "severity": fr.get("severity"),
                "drift_detected": fr.get("drift_detected"),
            })

    return {
        "model_id": model_id,
        "predictions_24h": len(preds_24h),
        "avg_latency_ms": avg_latency,
        "sparkline": sparkline,
        "drift_status": live_gov["drift_status"],
        "live_governance_score": live_gov["live_governance_score"],
        "base_audit_score": live_gov["base_audit_score"],
        "drift_penalty": live_gov["drift_penalty"],
        "feature_drift_summary": feature_summary,
    }
