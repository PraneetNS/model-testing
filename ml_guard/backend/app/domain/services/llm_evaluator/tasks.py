import asyncio
from datetime import datetime
import structlog
from app.core.celery_app import celery_app
from app.infrastructure.database import SessionLocal
from app.infrastructure.persistence import models as sql_models
from app.domain.services.llm_evaluator.engine import LLMEvaluationEngine

logger = structlog.get_logger(__name__)

@celery_app.task(name="app.domain.services.llm_evaluator.tasks.run_llm_evaluation_task")
def run_llm_evaluation_task(job_id: str, provider_config: dict, eval_config: dict):
    """
    Celery background worker task for LLM governance audit.
    """
    db = SessionLocal()
    engine = LLMEvaluationEngine()
    
    try:
        logger.info("Starting LLM evaluation task", job_id=job_id)
        
        # Initialize record
        eval_record = db.query(sql_models.LLMEvaluation).filter(sql_models.LLMEvaluation.id == job_id).first()
        if not eval_record:
            eval_record = sql_models.LLMEvaluation(
                id=job_id,
                model_name=provider_config["model_name"],
                provider=provider_config["provider"],
                status="IN_PROGRESS"
            )
            db.add(eval_record)
            db.commit()

        # Run async engine using asyncio.run
        report = asyncio.run(engine.run_evaluation(provider_config, eval_config))
        
        # Update record with results
        eval_record.status = "COMPLETED"
        eval_record.metrics = report.metrics.model_dump()
        eval_record.detailed_results = report.detailed_results
        eval_record.completed_at = datetime.utcnow()
        db.commit()
        
        logger.info("LLM evaluation task completed", job_id=job_id)
        return {"status": "success", "job_id": job_id}
        
    except Exception as e:
        logger.error("LLM evaluation task failed", job_id=job_id, error=str(e))
        eval_record = db.query(sql_models.LLMEvaluation).filter(sql_models.LLMEvaluation.id == job_id).first()
        if eval_record:
            eval_record.status = "FAILED"
            eval_record.error = str(e)
            db.commit()
        return {"status": "error", "message": str(e)}
    finally:
        db.close()
