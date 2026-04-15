"""
ingestion_service.py — ML Guard Production Observability Layer

Handles prediction log ingestion, batch writes, label stitching,
and baseline reference distribution storage via MinIO.
"""
from __future__ import annotations

import io
import uuid
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db.models import PredictionLog
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Single prediction write (called from background task in endpoint)
# ─────────────────────────────────────────────────────────────────────

async def ingest_single(
    db: AsyncSession,
    model_id: str,
    features: Dict[str, Any],
    prediction: Any,
    prediction_proba: Optional[float] = None,
    latency_ms: Optional[float] = None,
    data_source: str = "api",
    environment: str = "production",
    tags: Optional[Dict[str, Any]] = None,
) -> str:
    """Write a single prediction record and return its UUID."""
    log = PredictionLog(
        id=uuid.uuid4(),
        model_id=str(model_id),
        features=features,
        prediction=str(prediction),
        prediction_proba=prediction_proba,
        latency_ms=latency_ms,
        data_source=data_source,
        environment=environment,
        tags=tags or {},
    )
    db.add(log)
    await db.commit()
    await db.refresh(log)
    logger.info("ingested_prediction", model_id=model_id, log_id=str(log.id))

    # ── Real-time Security Scan (v7.2) ────────────────────────────────────────
    try:
        from app.routers.security import run_realtime_scan
        # We run this synchronously here for the demo/stability
        sec_res = await run_realtime_scan(
            model_id=model_id, 
            features=features or {}
        )
        if sec_res.get("risk_level") in ("HIGH", "MEDIUM"):
            logger.warning(
                "realtime_security_threat_detected",
                model_id=model_id,
                log_id=str(log.id),
                risk=sec_res.get("risk_level"),
                anomalies=sec_res.get("anomalies")
            )
            # Store in PredictionLog if we had a security_flag column, 
            # for now just log it.
    except Exception as _sec_err:
        logger.debug(f"realtime_security_scan_failed error={_sec_err}")

    # ── Contract enforcement (fire-and-forget safe) ───────────────────────────
    # Runs synchronously but NEVER raises. Any failure is silently swallowed
    # so the ingest pipeline is always safe.
    try:
        from app.services.contract_engine import ContractEngine
        _ce = ContractEngine()
        breaches = await _ce.check_prediction(
            db=db,
            model_id=model_id,
            prediction=prediction,
            prediction_proba=prediction_proba,
            features=features or {},
            latency_ms=latency_ms,
            log_id=str(log.id),
        )
        if breaches:
            logger.warning(
                "contract_breaches_detected",
                model_id=model_id,
                log_id=str(log.id),
                count=len(breaches),
                breaches=[
                    {"promise": b.get("promise"), "actual": b.get("actual")}
                    for b in breaches
                ],
            )
    except Exception as _contract_err:
        # Contract checking must NEVER crash the ingest pipeline
        logger.debug(f"contract_check_suppressed model_id={model_id} error={_contract_err}")

    return str(log.id)


# ─────────────────────────────────────────────────────────────────────
# Batch write (via Celery task, see tasks/ingest.py)
# ─────────────────────────────────────────────────────────────────────

async def ingest_batch(rows: List[Dict[str, Any]]) -> int:
    """Bulk-insert prediction rows. Uses a fresh DB session (Celery-safe)."""
    db = SessionLocal()
    try:
        logs = []
        for row in rows:
            logs.append(PredictionLog(
                id=uuid.uuid4(),
                model_id=str(row.get("model_id", "")),
                features=row.get("features", {}),
                prediction=str(row.get("prediction", "")),
                prediction_proba=row.get("prediction_proba"),
                latency_ms=row.get("latency_ms"),
                data_source=row.get("data_source", "batch"),
                environment=row.get("environment", "production"),
                tags=row.get("tags", {}),
            ))
        db.bulk_save_objects(logs)
        await db.commit()
        logger.info("batch_ingested", count=len(logs))
        return len(logs)
    except Exception as e:
        db.rollback()
        logger.error("batch_ingest_failed", error=str(e))
        raise
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────
# Label stitching — fills ground_truth after the fact
# ─────────────────────────────────────────────────────────────────────

async def stitch_labels(
    db: AsyncSession,
    log_ids: List[str],
    ground_truths: List[Any],
) -> int:
    """Update PredictionLog rows with ground truth labels."""
    if len(log_ids) != len(ground_truths):
        raise ValueError("log_ids and ground_truths must have equal length")

    updated = 0
    for lid, gt in zip(log_ids, ground_truths):
        row = (await db.execute(select(PredictionLog).filter(PredictionLog.id == lid))).scalars().first()
        if row:
            row.ground_truth = str(gt)
            updated += 1

    await db.commit()
    logger.info("labels_stitched", updated=updated, requested=len(log_ids))
    return updated


# ─────────────────────────────────────────────────────────────────────
# Reference baseline storage in MinIO
# ─────────────────────────────────────────────────────────────────────

def store_baseline_to_minio(model_id: str, df: pd.DataFrame) -> str:
    """Persist the reference feature distribution as parquet in MinIO."""
    try:
        from app.services.storage_service import _get_s3_client
        from app.core.config import settings

        client = _get_s3_client()
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)

        key = f"baselines/{model_id}/reference.parquet"
        client.put_object(
            Bucket=settings.MINIO_BUCKET,
            Key=key,
            Body=buf,
            ContentLength=buf.getbuffer().nbytes,
            ContentType="application/octet-stream",
        )
        logger.info("baseline_stored", model_id=model_id, key=key)
        return key
    except Exception as e:
        logger.warning("baseline_store_failed", model_id=model_id, error=str(e))
        return ""


def load_baseline_from_minio(model_id: str) -> Optional[pd.DataFrame]:
    """Load reference distribution parquet from MinIO, returns None if missing."""
    try:
        from app.services.storage_service import _get_s3_client
        from app.core.config import settings

        client = _get_s3_client()
        key = f"baselines/{model_id}/reference.parquet"
        obj = client.get_object(Bucket=settings.MINIO_BUCKET, Key=key)
        buf = io.BytesIO(obj["Body"].read())
        return pd.read_parquet(buf)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────
# Fetch recent predictions as DataFrame (for drift / perf analysis)
# ─────────────────────────────────────────────────────────────────────

async def get_recent_predictions_df(
    db: AsyncSession,
    model_id: str,
    hours: int = 24,
    environment: Optional[str] = None,
) -> pd.DataFrame:
    """Return recent PredictionLog rows as a pandas DataFrame."""
    from datetime import timedelta
    cutoff = datetime.utcnow() - timedelta(hours=hours)

    from sqlalchemy import select
    stmt = select(PredictionLog).filter(
        PredictionLog.model_id == model_id,
        PredictionLog.timestamp >= cutoff,
    )
    if environment:
        stmt = stmt.filter(PredictionLog.environment == environment)

    result = await db.execute(stmt.order_by(PredictionLog.timestamp.desc()))
    rows = result.scalars().all()
    if not rows:
        return pd.DataFrame()

    records = []
    for r in rows:
        rec = {"log_id": str(r.id), "timestamp": r.timestamp,
               "prediction": r.prediction, "prediction_proba": r.prediction_proba,
               "ground_truth": r.ground_truth, "latency_ms": r.latency_ms,
               "environment": r.environment}
        if r.features:
            rec.update(r.features)
        records.append(rec)

    return pd.DataFrame(records)
