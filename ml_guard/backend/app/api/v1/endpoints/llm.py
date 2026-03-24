from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
import structlog
import uuid

from app.api.v1 import deps
from app.infrastructure.persistence import models as sql_models
from app.domain.models.llm import LLMModelPullRequest, LLMEvaluationConfig, LLMEvalJobResponse, LLMFullReport
from app.domain.services.llm_evaluator.tasks import run_llm_evaluation_task

logger = structlog.get_logger(__name__)
router = APIRouter()

@router.post("/pull", response_model=LLMEvalJobResponse)
async def pull_and_eval_llm(
    request: LLMModelPullRequest, 
    db: Session = Depends(deps.get_db),
    config: LLMEvaluationConfig = Depends()
):
    """
    Enterprise LLM Governance: Initiate a pull and evaluation session.
    Triggers an async Celery worker for heavy inference.
    """
    job_id = f"llm_{str(uuid.uuid4())[:12]}"
    
    # Check for existing job for this model to avoid redundancy
    # existing = db.query(sql_models.LLMEvaluation).filter(sql_models.LLMEvaluation.model_name == request.model_name).first()
    
    # 1. Register job in database
    eval_record = sql_models.LLMEvaluation(
        id=job_id,
        model_name=request.model_name,
        provider=request.provider,
        status="IN_PROGRESS"
    )
    db.add(eval_record)
    db.commit()
    
    # 2. Dispatch to Celery worker (Redis queue)
    provider_config = {
        "provider": request.provider,
        "model_name": request.model_name,
        "api_key": request.api_key
    }
    
    run_llm_evaluation_task.delay(
        job_id=job_id, 
        provider_config=provider_config, 
        eval_config=config.model_dump()
    )
    
    return LLMEvalJobResponse(
        job_id=job_id,
        status="IN_PROGRESS",
        model_name=request.model_name
    )

@router.get("/status/{job_id}")
async def get_job_status(job_id: str, db: Session = Depends(deps.get_db)):
    """
    Poll the enterprise ledger for the status and results of an LLM audit.
    """
    eval_record = db.query(sql_models.LLMEvaluation).filter(sql_models.LLMEvaluation.id == job_id).first()
    if not eval_record:
        raise HTTPException(status_code=404, detail="Governance job not found")
        
    response = {
        "status": eval_record.status,
        "job_id": eval_record.id,
        "model": eval_record.model_name
    }
    
    if eval_record.status == "COMPLETED":
        response["report"] = {
            "job_id": eval_record.id,
            "model_name": eval_record.model_name,
            "provider": eval_record.provider,
            "metrics": eval_record.metrics,
            "detailed_results": eval_record.detailed_results,
            "completed_at": eval_record.completed_at
        }
    elif eval_record.status == "FAILED":
        response["error"] = eval_record.error
        
    return response

@router.get("/history", response_model=List[Dict[str, Any]])
async def get_eval_history(db: Session = Depends(deps.get_db)):
    """Retrieve history of all LLM governance audits."""
    evals = db.query(sql_models.LLMEvaluation).order_by(sql_models.LLMEvaluation.created_at.desc()).all()
    return evals
