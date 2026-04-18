import uuid
import datetime
import json
import os
from typing import List, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.session import get_db
from app.db.models import Sandbox, Model
from app.core.auth import AuthContext, require_role
from ml_guard.sandbox.sandbox_runner import ModelSandbox, SandboxHandle
from ml_guard.sandbox.attacks import fgsm, square_attack, prompt_injection_suite

router = APIRouter()
sandbox_manager = ModelSandbox()

@router.post("/sandbox/create")
async def create_sandbox(
    body: Dict = Body(...),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer"))
):
    model_id = body.get("model_id")
    if not model_id:
        raise HTTPException(400, "model_id required")
        
    result = await db.execute(select(Model).filter(Model.id == model_id))
    model = result.scalars().first()
    if not model:
        raise HTTPException(404, "Model not found")

    # Resolve model path. 
    # In production, we'd download from model.artifact_url.
    # For the sandbox architecture demo, we'll look for a local file.
    model_path = f"model_{model_id}.pkl"
    if not os.path.exists(model_path):
        # Fallback to test_model if exists, or create dummy
        model_path = "test_model.pkl"
        if not os.path.exists(model_path):
            with open(model_path, "wb") as f:
                import joblib
                from sklearn.ensemble import RandomForestClassifier
                import numpy as np
                m = RandomForestClassifier().fit(np.random.rand(10,2), [0,1]*5)
                joblib.dump(m, f)

    handle = sandbox_manager.create_sandbox(model_path)
    if not handle:
        raise HTTPException(500, "Failed to create sandbox (Docker error)")
    
    sandbox_id = str(uuid.uuid4())
    expires_at = datetime.datetime.utcnow() + datetime.timedelta(minutes=30)
    
    new_sandbox = Sandbox(
        id=sandbox_id,
        model_id=model_id,
        container_id=handle.container_id,
        port=handle.port,
        expires_at=expires_at
    )
    db.add(new_sandbox)
    await db.commit()
    
    return {
        "sandbox_id": sandbox_id, 
        "port": handle.port, 
        "expires_at": expires_at.isoformat()
    }

@router.post("/sandbox/{sandbox_id}/predict")
async def sandbox_predict(
    sandbox_id: str,
    features: Dict = Body(...),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("viewer"))
):
    result = await db.execute(select(Sandbox).filter(Sandbox.id == sandbox_id))
    sandbox = result.scalars().first()
    if not sandbox:
        raise HTTPException(404, "Sandbox session not found")
        
    handle = SandboxHandle(sandbox.container_id, sandbox.port, sandbox_manager.client)
    return handle.predict(features)

@router.post("/sandbox/{sandbox_id}/red-team")
async def sandbox_red_team(
    sandbox_id: str,
    attack_type: str = Body(..., embed=True),
    n_samples: int = Body(1, embed=True),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer"))
):
    result = await db.execute(select(Sandbox).filter(Sandbox.id == sandbox_id))
    sandbox = result.scalars().first()
    if not sandbox:
        raise HTTPException(404, "Sandbox session not found")
        
    handle = SandboxHandle(sandbox.container_id, sandbox.port, sandbox_manager.client)
    
    if attack_type == "prompt_injection":
        results = prompt_injection_suite(handle.predict)
        success_count = sum(1 for r in results if r["violated"])
        return {
            "attack_type": attack_type, 
            "success_rate": success_count / len(results) if results else 0, 
            "examples": results
        }
    
    # For Square Attack (Numerical)
    # Generate mock numerical input if not provided
    import numpy as np
    X_orig = np.random.rand(1, 10) 
    
    examples = []
    if attack_type == "square":
        # Square attack uses the predict_fn directly
        adv_X = square_attack(handle.predict, X_orig, n_queries=n_samples * 50)
        orig_y = handle.predict({"features": X_orig[0].tolist()})
        adv_y = handle.predict({"features": adv_X[0].tolist()})
        examples.append({
            "input": X_orig[0].tolist(),
            "output": orig_y.get("output"),
            "adversarial_input": adv_X[0].tolist(),
            "adversarial_output": adv_y.get("output")
        })

    return {
        "attack_type": attack_type, 
        "success_rate": 1.0 if examples and examples[0]["output"] != examples[0]["adversarial_output"] else 0, 
        "examples": examples
    }

@router.delete("/sandbox/{sandbox_id}")
async def delete_sandbox(
    sandbox_id: str,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer"))
):
    result = await db.execute(select(Sandbox).filter(Sandbox.id == sandbox_id))
    sandbox = result.scalars().first()
    if not sandbox:
        raise HTTPException(404, "Sandbox not found")
        
    handle = SandboxHandle(sandbox.container_id, sandbox.port, sandbox_manager.client)
    handle.shutdown()
    
    await db.delete(sandbox)
    await db.commit()
    return {"status": "deleted"}
