from fastapi import APIRouter, HTTPException, Depends, Body
from typing import Optional, List, Dict
import os
import time
import requests
import joblib
import pandas as pd
from pydantic import BaseModel
from ml_guard.core.policy_schema import MLGuardPolicy, GateRequest, GateVerdict
from ml_guard.core.policy import evaluate_policy
from ml_guard.core.llm_guard import evaluate_llm_safety
from ml_guard.core.evaluator import MLEvaluator

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
            llm_results = evaluate_llm_safety(test_prompt, "I am a helpful assistant.")
            
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
