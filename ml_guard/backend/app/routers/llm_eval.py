"""
LLM Governance Router.
Endpoints for evaluating LLM prompt/response pairs for safety and quality.
"""
import uuid
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from pydantic import BaseModel, Field
from typing import List, Optional
from app.db.session import get_db
from app.db.models import Job, LLMResult, LLMScanRecord, AuditLog
from app.core.auth import AuthContext, get_auth_context, log_action

router = APIRouter()


class LLMEvalRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="The input prompt to evaluate")
    response: str = Field(..., min_length=1, description="The model's response")
    additional_responses: Optional[List[str]] = Field(None, description="Additional responses for stability analysis")
    reference_facts: Optional[List[str]] = Field(None, description="Known facts for hallucination checking")
    model_name: Optional[str] = Field("unknown", description="Name or identifier of the LLM")


@router.post("/llm/evaluate")
async def evaluate_llm_endpoint(
    req: LLMEvalRequest,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
):
    """
    Full LLM governance evaluation.

    Analyzes a prompt/response pair for:
    - Prompt injection attempts
    - Toxicity scoring
    - Hallucination risk
    - Response stability (if multiple responses provided)

    Returns risk score, risk level, and detailed per-check breakdown.
    """
    from ml_guard.core.llm_guard import evaluate_llm
    from ml_guard.core.policy import evaluate_policy

    # ─── Run evaluation ───
    result = evaluate_llm(
        prompt=req.prompt,
        response=req.response,
        additional_responses=req.additional_responses,
        reference_facts=req.reference_facts,
    )

    # ─── Policy evaluation (LLM checks only) ───
    policy_result = evaluate_policy(llm_evaluation=result)

    # ─── Persist to DB ───
    try:
        llm_scan = LLMScanRecord(
            prompt_hash=result["prompt_hash"],
            response_hash=result["response_hash"],
            prompt_text=req.prompt[:2000],  # truncate for storage
            response_text=req.response[:5000],
            results_json=result,
            llm_risk_score=result["llm_risk_score"],
            llm_risk_level=result["llm_risk_level"],
            toxicity_score=result["toxicity_score"],
            hallucination_risk=result["hallucination_risk"],
            injection_flag=result["prompt_injection_flag"],
            stability_score=result["stability_score"],
        )
        db.add(llm_scan)
        await db.commit()
        await db.refresh(llm_scan)
        scan_id = str(llm_scan.id)
    except Exception:
        scan_id = None

    # ─── Log to enterprise stream ───
    log_action(db, auth, "llm.evaluate", "llm", scan_id, {
        "model_name": req.model_name,
        "risk_score": result["llm_risk_score"],
        "risk_level": result["llm_risk_level"],
        "injection_flag": result["prompt_injection_flag"],
        "toxicity": result["toxicity_score"],
    })

    return {
        "evaluation": result,
        "policy": policy_result,
        "scan_id": scan_id,
        "model_name": req.model_name,
    }


@router.get("/llm/history")
async def get_llm_history(
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
):
    """Get recent LLM scan history."""
    from sqlalchemy import desc
    records = (await db.execute(select(LLMScanRecord).order_by(desc(LLMScanRecord.created_at)).limit(limit))).scalars().all()
    return [
        {
            "id": str(r.id),
            "prompt_hash": r.prompt_hash,
            "response_hash": r.response_hash,
            "llm_risk_score": r.llm_risk_score,
            "llm_risk_level": r.llm_risk_level,
            "toxicity_score": r.toxicity_score,
            "hallucination_risk": r.hallucination_risk,
            "injection_flag": r.injection_flag,
            "stability_score": r.stability_score,
            "created_at": str(r.created_at) if r.created_at else None,
        }
        for r in records
    ]


@router.get("/llm/{job_id}")
async def get_llm_results(job_id: str, db: AsyncSession = Depends(get_db)):
    """Get LLM results for a specific job (legacy compatibility)."""
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    result = (await db.execute(select(LLMResult).filter(LLMResult.job_id == job_id))).scalars().first()
    if not result:
        return {"status": job.status, "error": job.error, "result": None}

    return {
        "status": job.status,
        "result": {
            "metrics": result.computed_metrics_json,
            "severity_counts": result.severity_counts,
            "module_status": result.status
        }
    }
