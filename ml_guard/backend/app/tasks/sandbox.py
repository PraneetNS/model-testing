import asyncio
from datetime import datetime, timezone
from app.core.celery_app import celery_app
from app.db.session import SessionLocal
from app.db.models import Sandbox, utcnow
from sqlalchemy.future import select
from sqlalchemy import delete
import structlog

logger = structlog.get_logger(__name__)

async def _cleanup_expired_sandboxes():
    """Removes sandbox environments that have passed their expiration date."""
    async with SessionLocal() as db:
        # Find expired sandboxes
        stmt = select(Sandbox).where(Sandbox.expires_at < utcnow())
        result = await db.execute(stmt)
        expired = result.scalars().all()
        
        for sb in expired:
            try:
                # Actual cleanup logic (e.g. killing containers, deleting namespaces) would go here
                logger.info("cleaning_up_sandbox", sandbox_id=str(sb.id), name=sb.name)
                await db.delete(sb)
            except Exception as e:
                logger.error("sandbox_cleanup_failed", sandbox_id=str(sb.id), error=str(e))
        
        await db.commit()

@celery_app.task(name="app.tasks.sandbox.cleanup_expired")
def cleanup_expired():
    """
    Beat job: cleanup expired model sandboxes.
    """
    logger.info("starting_sandbox_cleanup")
    asyncio.run(_cleanup_expired_sandboxes())
    logger.info("finished_sandbox_cleanup")
