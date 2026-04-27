import asyncio
from app.db.session import SessionLocal
from app.db.models import UsageEvent
from app.core.celery_app import celery_app
import structlog

logger = structlog.get_logger()

@celery_app.task(name="app.tasks.billing.record_usage_task", bind=True, max_retries=5, default_retry_delay=5)
async def record_usage_task(self, org_id: str, key_id: str, event_type: str, quantity: int, metadata: dict):
    """
    Asynchronous task to persist a usage event to the database.
    """
    try:
        async with SessionLocal() as db:
            event = UsageEvent(
                org_id=org_id,
                api_key_id=key_id,
                event_type=event_type,
                quantity=quantity,
                metadata_json=metadata or {}
            )
            db.add(event)
            await db.commit()
            logger.info("Usage event recorded", org_id=org_id, event_type=event_type, quantity=quantity)
    except Exception as exc:
        logger.error("Failed to record usage event", error=str(exc))
        raise self.retry(exc=exc)
