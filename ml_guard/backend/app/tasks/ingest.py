"""
app/tasks/ingest.py — Celery tasks for batch prediction ingestion
"""
from app.core.celery_app import celery_app
from app.services.ingestion_service import ingest_batch
import structlog

logger = structlog.get_logger()


@celery_app.task(name="app.tasks.ingest.ingest_batch_task", bind=True, max_retries=3)
def ingest_batch_task(self, rows: list) -> dict:
    """Bulk-insert prediction rows into DB. Retryable on DB error."""
    try:
        count = ingest_batch(rows)
        return {"status": "success", "rows_written": count}
    except Exception as exc:
        logger.error("batch_ingest_task_failed", error=str(exc))
        raise self.retry(exc=exc, countdown=10)
