"""
app/tasks/observability.py — Celery beat tasks for scheduled observability scans.

drift_scan:             every 1 hour  — for all active models
performance_snapshot:   every 6 hours — for all active models
"""
from celery import shared_task
from app.core.celery_app import celery_app
from app.db.session import SessionLocal
from app.db.models import PredictionLog, PerformanceSnapshot
import structlog

logger = structlog.get_logger()


def _get_active_model_ids() -> list:
    """Fetch distinct model_ids from prediction_logs (active models)."""
    db = SessionLocal()
    try:
        rows = db.query(PredictionLog.model_id).distinct().all()
        return [r[0] for r in rows if r[0]]
    finally:
        db.close()


@celery_app.task(name="app.tasks.observability.run_hourly_drift_scan")
def run_hourly_drift_scan():
    """
    Beat job: run drift analysis for all active models every hour.
    Triggers governance audit if HIGH/CRITICAL drift detected.
    """
    model_ids = _get_active_model_ids()
    logger.info("drift_scan_started", model_count=len(model_ids))

    results = []
    for model_id in model_ids:
        db = SessionLocal()
        try:
            from app.services.drift_analyzer import DriftAnalyzer
            analyzer = DriftAnalyzer(db, model_id, method="ks")
            result = analyzer.analyze(window_hours=1)
            if result:
                results.append({
                    "model_id": model_id,
                    "drift_detected": result.get("drift_detected"),
                    "drift_score": result.get("overall_drift_score"),
                })
                logger.info("drift_scan_complete", model_id=model_id,
                            drift=result.get("drift_detected"),
                            score=result.get("overall_drift_score"))
        except Exception as e:
            logger.error("drift_scan_failed", model_id=model_id, error=str(e))
        finally:
            db.close()

    return {"scanned": len(model_ids), "results": results}


@celery_app.task(name="app.tasks.observability.run_performance_snapshot")
def run_performance_snapshot():
    """
    Beat job: compute performance snapshots for all active models every 6 hours.
    """
    model_ids = _get_active_model_ids()
    logger.info("perf_snapshot_started", model_count=len(model_ids))

    results = []
    for model_id in model_ids:
        db = SessionLocal()
        try:
            from app.services.performance_tracker import PerformanceTracker

            # Use the penultimate snapshot as baseline
            prev_snap = (
                db.query(PerformanceSnapshot)
                .filter(PerformanceSnapshot.model_id == model_id)
                .order_by(PerformanceSnapshot.computed_at.desc())
                .offset(1)
                .first()
            )
            baseline = prev_snap.metrics if prev_snap else None

            tracker = PerformanceTracker(db, model_id)
            result = tracker.compute_snapshot(window_hours=6, baseline_metrics=baseline)
            if result:
                results.append({"model_id": model_id, "metrics": result.get("metrics")})
                logger.info("perf_snapshot_complete", model_id=model_id)
        except Exception as e:
            logger.error("perf_snapshot_failed", model_id=model_id, error=str(e))
        finally:
            db.close()

    return {"snapshotted": len(model_ids), "results": results}
