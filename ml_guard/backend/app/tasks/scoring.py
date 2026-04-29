import asyncio
from app.core.celery_app import celery_app
from app.db.session import SessionLocal
from app.db.models import Model
from sqlalchemy.future import select
import structlog

logger = structlog.get_logger(__name__)

async def _refresh_all_scores():
    """Logic to refresh governance scores for all models."""
    async with SessionLocal() as db:
        result = await db.execute(select(Model))
        models = result.scalars().all()
        
        from app.services.governance_engine import GovernanceEngine
        
        for model in models:
            try:
                engine = GovernanceEngine(db, str(model.id))
                await engine.compute_score()
                logger.info("governance_score_refreshed", model_id=str(model.id))
            except Exception as e:
                logger.error("governance_score_refresh_failed", model_id=str(model.id), error=str(e))

@celery_app.task(name="app.tasks.scoring.refresh_all_scores")
def refresh_all_scores():
    """
    Beat job: refresh governance scores for all models periodically.
    """
    logger.info("starting_score_refresh")
    asyncio.run(_refresh_all_scores())
    logger.info("finished_score_refresh")
