import asyncio
from datetime import datetime, timezone
from app.core.celery_app import celery_app
from app.db.session import SessionLocal
from app.db.models import Model, utcnow
from sqlalchemy.future import select
import structlog

logger = structlog.get_logger(__name__)

async def _check_validation_due_dates():
    """Identifies models overdue for validation and triggers alerts."""
    async with SessionLocal() as db:
        stmt = select(Model).where(Model.next_validation_due_at < utcnow())
        result = await db.execute(stmt)
        overdue_models = result.scalars().all()
        
        from app.routers.alerts import InternalAlertCreate, create_internal_alert
        
        for model in overdue_models:
            try:
                alert = InternalAlertCreate(
                    severity="MEDIUM",
                    message=f"Model '{model.name}' is overdue for governance validation (Due: {model.next_validation_due_at})",
                    source="inventory",
                    model_id=str(model.id)
                )
                await create_internal_alert(alert, db)
                logger.info("validation_overdue_alert_triggered", model_id=str(model.id))
            except Exception as e:
                logger.error("validation_overdue_alert_failed", model_id=str(model.id), error=str(e))

@celery_app.task(name="app.tasks.inventory.check_validation_due_dates")
def check_validation_due_dates():
    """
    Beat job: check for models with overdue validation dates.
    """
    logger.info("starting_inventory_check")
    asyncio.run(_check_validation_due_dates())
    logger.info("finished_inventory_check")
