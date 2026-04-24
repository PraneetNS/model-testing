from app.core.celery_app import celery_app
from app.infrastructure.database import SessionLocal
from app.db.models import RetrainingPolicy
from ml_guard.core.retraining import evaluate_retrain_trigger, execute_retrain_action
from sqlalchemy import select
import asyncio
import structlog

logger = structlog.get_logger(__name__)

async def _process_retraining_policies():
    async with SessionLocal() as db:
        # Get all enabled policies
        policies = (await db.execute(select(RetrainingPolicy).filter(RetrainingPolicy.enabled == True))).scalars().all()
        
        for policy in policies:
            try:
                result = await evaluate_retrain_trigger(policy.model_id, db)
                if result.get("should_trigger") and not result.get("suppressed"):
                    logger.info("retraining_trigger_fired", model_id=policy.model_id, conditions=result["triggered_conditions"])
                    await execute_retrain_action(policy, result, db)
            except Exception as e:
                logger.error("retraining_trigger_failed", model_id=policy.model_id, error=str(e))


@celery_app.task(name="app.tasks.retraining.evaluate_all_retraining_policies")
def evaluate_all_retraining_policies():
    """
    Celery beat task that evaluates all enabled retraining policies.
    """
    logger.info("starting_retraining_evaluator")
    asyncio.run(_process_retraining_policies())
    logger.info("finished_retraining_evaluator")
