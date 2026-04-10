"""
Explainability Router.
Endpoints for computing and retrieving model explanations.
"""
import io
import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.session import get_db
from app.db.models import Model, ExplainabilityResult, ScanRecord
from app.core.auth import AuthContext, require_role

router = APIRouter()


@router.post("/explainability/compute")
async def compute_explainability(
    model_file: UploadFile = File(...),
    dataset_file: UploadFile = File(...),
    model_id: str = Form(""),
    max_samples: int = Form(100),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer")),
):
    """Compute explainability metrics for a model + dataset pair."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    try:
        from ml_guard.core.explainability import run_explainability
    except ImportError:
        # Fallback inline implementation
        run_explainability = None

    import uuid
    from app.db.models import Model

    # 1. Sanitize model_id for Postgres UUID constraints
    try:
        valid_model_id = str(uuid.UUID(model_id))
    except (ValueError, TypeError, AttributeError):
        valid_model_id = str(uuid.uuid4())

    # 2. Ensure model exists to satisfy ExplainabilityResult ForeignKey
    model_record = (await db.execute(select(Model).filter(Model.id == valid_model_id))).scalars().first()
    if not model_record:
        dummy_model = Model(id=valid_model_id, name=f"Adhoc Explainer {valid_model_id[:6]}")
        db.add(dummy_model)
        await db.commit()

    model_id = valid_model_id

    # Encode Data for Worker Transfer (No Shared Storage Needed)
    import base64
    m_b64 = base64.b64encode(await model_file.read()).decode("utf-8")
    d_b64 = base64.b64encode(await dataset_file.read()).decode("utf-8")

    # Dispatch task
    from app.workers.tasks import run_explainability_task
    task = run_explainability_task.delay(
        model_id=model_id,
        model_b64=m_b64,
        data_b64=d_b64,
        max_samples=max_samples
    )

    return {
        "status": "pending",
        "task_id": task.id,
        "model_id": model_id,
        "message": "Explainability computation started via direct data transfer."
    }

@router.get("/explainability/{model_id}")
async def get_explainability(
    model_id: str,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("viewer")),
):
    import uuid
    try:
        valid_model_id = str(uuid.UUID(model_id))
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(400, "Invalid explainability model_id format.")

    """Get stored explainability results for a model."""
    results = db.query(ExplainabilityResult).filter(
        ExplainabilityResult.model_id == valid_model_id
    ).order_by(ExplainabilityResult.created_at.desc()).all()

    if not results:
        raise HTTPException(404, "No explainability results found for this model.")

    return {
        "model_id": model_id,
        "results": [
            {
                "id": str(r.id),
                "method": r.method,
                "global_importance": r.global_importance,
                "summary_metrics": r.summary_metrics,
                "created_at": str(r.created_at),
            }
            for r in results
        ],
    }
