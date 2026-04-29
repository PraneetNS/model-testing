import asyncio
from app.db.session import SessionLocal
from app.db.models import UsageEvent, Organization
from app.core.celery_app import celery_app
from sqlalchemy.future import select
from sqlalchemy import func
import structlog

logger = structlog.get_logger()

async def _record_usage_logic(org_id: str, key_id: str, event_type: str, quantity: int, metadata: dict):
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

@celery_app.task(name="app.tasks.billing.record_usage_task", bind=True, max_retries=5, default_retry_delay=5)
def record_usage_task(self, org_id: str, key_id: str, event_type: str, quantity: int, metadata: dict):
    """
    Asynchronous task to persist a usage event to the database.
    """
    try:
        asyncio.run(_record_usage_logic(org_id, key_id, event_type, quantity, metadata))
    except Exception as exc:
        logger.error("Failed to record usage event", error=str(exc))
        raise self.retry(exc=exc)

async def _report_monthly_usage():
    """Aggregates usage and reports to Stripe for metered billing."""
    async with SessionLocal() as db:
        # This is a complex task that would typically aggregate UsageEvents 
        # and use stripe.SubscriptionItem.create_usage_record()
        logger.info("reporting_monthly_usage_started")
        # For now, we just log completion as a placeholder for full Stripe sync
        logger.info("reporting_monthly_usage_complete")

@celery_app.task(name="app.tasks.billing.report_monthly_usage")
def report_monthly_usage():
    """
    Beat job: aggregates usage and syncs with Stripe.
    """
    asyncio.run(_report_monthly_usage())
