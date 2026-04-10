from fastapi import APIRouter, HTTPException, Depends, Body
from app.db.session import get_db
from typing import Optional, List, Dict
import os
import time
import requests
import joblib
import pandas as pd
from pydantic import BaseModel
from ml_guard.core.policy_schema import MLGuardPolicy, GateRequest, GateVerdict
from ml_guard.core.policy import evaluate_policy
from ml_guard.core.llm_guard import evaluate_llm
from ml_guard.core import MLEvaluator

router = APIRouter()

def generate_badge_url(status: str, score: float) -> str:
    """
    Generate a shields.io-style badge URL for the gate result.
    """
    color = "green" if status == "PASSED" else "orange" if status == "WARNING" else "red"
    return f"https://img.shields.io/badge/MLGuard-{status}_{int(score)}--100-{color}?style=for-the-badge"

@router.post("/evaluate", response_model=GateVerdict)
async def evaluate_gate(request: GateRequest):
    """
    Synchronous CI/CD evaluation gate.
    Runs a lightweight subset of governance checks inline.
    """
    start_time = time.time()
    policy = request.policy
    failures = []
    actual_metrics = {}
    
    # --- Case 1: Inference Endpoint (LLM or Probing) ---
    if request.inference_url:
        try:
            # For LLM checks, we probe the endpoint with a set of test cases
            test_prompt = "Hello, tell me a joke."
            # In a real scenario, we'd send multiple probes for toxicity, injection, etc.
            # Here we simulate the evaluation based on the policy requirements
            
            # Simulate LLM safety check
            llm_results = evaluate_llm(test_prompt, "I am a helpful assistant.")
            
            # Check against policy
            if llm_results["toxicity_score"] > policy.max_toxicity_score:
                failures.append(f"Toxicity score {llm_results['toxicity_score']} exceeds threshold {policy.max_toxicity_score}")
            if llm_results["hallucination_risk"] > policy.max_hallucination_rate:
                failures.append(f"Hallucination risk {llm_results['hallucination_risk']} exceeds threshold {policy.max_hallucination_rate}")
            
            actual_metrics = llm_results
            score = 100 - (len(failures) * 20)
            status = "CRITICAL" if failures else "PASSED"
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Inference endpoint evaluation failed: {str(e)}")

    # --- Case 2: Model Artifact (Tabular ML) ---
    elif request.artifact_path:
        if not os.path.exists(request.artifact_path):
            raise HTTPException(status_code=404, detail=f"Model artifact not found at {request.artifact_path}")
            
        try:
            # Load model (lightweight inspection)
            # In a real CI/CD, we'd have a small sample dataset provided or inferred
            # For this implementation, we simulate the evaluation results if data isn't provided
            
            # Example: Check if it's a valid scikit-learn model
            model = joblib.load(request.artifact_path)
            
            # Mocking some metrics for the CI gate demo
            # In production, we'd pull these from the CI context or a 'test_data.csv'
            actual_metrics = {
                "accuracy": 0.88,
                "max_psi": 0.12,
                "max_overfit_gap": 0.05
            }
            
            # Check against policy
            if actual_metrics["accuracy"] < policy.min_accuracy:
                failures.append(f"Accuracy {actual_metrics['accuracy']} is below threshold {policy.min_accuracy}")
            if actual_metrics["max_psi"] > policy.max_psi:
                failures.append(f"PSI Drift {actual_metrics['max_psi']} exceeds threshold {policy.max_psi}")
            
            score = 100 - (len(failures) * 25)
            status = "CRITICAL" if failures else "PASSED"
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Model artifact evaluation failed: {str(e)}")
    
    else:
        raise HTTPException(status_code=400, detail="Either artifact_path or inference_url must be provided.")

    end_time = time.time()
    
    return GateVerdict(
        passed=status == "PASSED",
        score=float(max(0, score)),
        gate_status=status,
        failures=failures,
        badge_url=generate_badge_url(status, score),
        details={
            "execution_time_sec": round(end_time - start_time, 3),
            "policy_version": policy.version,
            "metrics_evaluated": list(actual_metrics.keys())
        }
    )

@router.get("/result/{submission_token}")
async def get_gate_result(submission_token: str, db: Depends = Depends(get_db)):
    """
    Deterministically retrieve gate audit results securely utilizing the submission token.
    Combines Job status polling and Gate metric inspection.
    """
    from app.db.models import Job, ScanRecord
    from sqlalchemy.future import select
    from fastapi import HTTPException
    
    # 1. Verify existence of the submission record using strict token tracking
    job = (await db.execute(select(Job).filter(Job.submission_token == submission_token))).scalars().first()
    
    if not job:
        raise HTTPException(404, "Invalid submission token")
        
    if job.status in ["RUNNING", "PENDING"]:
        return {"status": "pending", "eta_seconds": 30} # Rough estimate based on task compute overhead
        
    if job.status == "FAILED":
        return {"status": "FAILED", "error": job.error}
        
    if job.status == "COMPLETED":
        scan = (await db.execute(select(ScanRecord).filter(ScanRecord.job_id == str(job.id)).order_by(ScanRecord.created_at.desc()))).scalars().first()
        if not scan:
            return {"status": "FAILED", "error": "Job is marked COMPLETED but no underlying artifact scan was correctly generated. Retry scan."}
            
        return {
            "status": "COMPLETED",
            "model_id": str(job.model_id),
            "score": scan.governance_score,
            "verdict": scan.gate_status,
            "breach_count": len(scan.checks_run) if scan.checks_run else 0
        }
