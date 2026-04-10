"""
Experiment Tracking Router.
Endpoints for tracking ML training runs, hyperparameters, and metrics.
"""
import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func
from app.db.session import get_db
from app.db.models import Experiment, Model, DatasetVersion, utcnow
from app.core.auth import AuthContext, require_role, log_action

router = APIRouter()


# ═══════════════════════════════════════════════
# START EXPERIMENT
# ═══════════════════════════════════════════════
@router.post("/experiments/start")
async def start_experiment(
    model_id: str,
    name: str = "",
    dataset_version_id: str = None,
    parameters: dict = Body(default={}),
    framework: str = "",
    tags: dict = Body(default={}),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer")),
):
    """Start a new experiment/training run."""
    model = db.get(Model, model_id)
    if not model:
        raise HTTPException(404, "Model not found.")

    experiment = Experiment(
        model_id=model_id,
        dataset_version_id=dataset_version_id,
        name=name or f"{model.name}_experiment",
        parameters=parameters,
        metrics={},
        framework=framework,
        tags=tags,
        status="RUNNING",
        created_by=auth.user_id,
    )
    db.add(experiment)
    await db.commit()
    await db.refresh(experiment)
    log_action(db, auth, "experiment.start", "experiment", str(experiment.id), {
        "model_id": model_id, "name": name
    })

    return {
        "experiment_id": str(experiment.id),
        "model_id": model_id,
        "name": experiment.name,
        "status": "RUNNING",
        "started_at": str(experiment.started_at),
    }


# ═══════════════════════════════════════════════
# LOG EXPERIMENT METRICS / PARAMS
# ═══════════════════════════════════════════════
@router.post("/experiments/log")
async def log_experiment(
    experiment_id: str,
    metrics: dict = Body(default={}),
    parameters: dict = Body(default={}),
    artifact_url: str = "",
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer")),
):
    """Log metrics and parameters to a running experiment."""
    experiment = db.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(404, "Experiment not found.")
    if experiment.status != "RUNNING":
        raise HTTPException(400, f"Experiment is {experiment.status}, cannot log to it.")

    # Merge metrics and parameters
    existing_metrics = experiment.metrics or {}
    existing_metrics.update(metrics)
    experiment.metrics = existing_metrics

    existing_params = experiment.parameters or {}
    existing_params.update(parameters)
    experiment.parameters = existing_params

    if artifact_url:
        experiment.artifact_url = artifact_url

    await db.commit()

    return {
        "experiment_id": experiment_id,
        "metrics": experiment.metrics,
        "parameters": experiment.parameters,
        "status": experiment.status,
    }


# ═══════════════════════════════════════════════
# END EXPERIMENT
# ═══════════════════════════════════════════════
@router.post("/experiments/end")
async def end_experiment(
    experiment_id: str,
    status: str = "COMPLETED",
    final_metrics: dict = Body(default={}),
    training_time_ms: int = None,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer")),
):
    """End an experiment with final metrics."""
    experiment = db.get(Experiment, experiment_id)
    if not experiment:
        raise HTTPException(404, "Experiment not found.")

    if status not in ("COMPLETED", "FAILED"):
        raise HTTPException(400, "Status must be COMPLETED or FAILED.")

    # Merge final metrics
    existing_metrics = experiment.metrics or {}
    existing_metrics.update(final_metrics)
    experiment.metrics = existing_metrics

    experiment.status = status
    experiment.completed_at = datetime.now(timezone.utc).replace(tzinfo=None)
    if training_time_ms:
        experiment.training_time_ms = training_time_ms

    await db.commit()
    log_action(db, auth, "experiment.end", "experiment", str(experiment.id), {
        "status": status, "metrics": final_metrics
    })

    return {
        "experiment_id": experiment_id,
        "status": status,
        "metrics": experiment.metrics,
        "completed_at": str(experiment.completed_at),
    }


# ═══════════════════════════════════════════════
# LIST EXPERIMENTS
# ═══════════════════════════════════════════════
@router.get("/experiments")
async def list_experiments(
    model_id: str = None,
    status: str = None,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("viewer")),
):
    """List experiments with optional filtering."""
    q = db.query(Experiment)
    if model_id:
        q = q.filter(Experiment.model_id == model_id)
    if status:
        q = q.filter(Experiment.status == status)

    total = q.count()
    offset = (page - 1) * per_page
    experiments = q.order_by(Experiment.created_at.desc()).offset(offset).limit(per_page).all()

    items = []
    for e in experiments:
        model = db.get(Model, str(e.model_id))
        items.append({
            "experiment_id": str(e.id),
            "name": e.name,
            "model_id": str(e.model_id),
            "model_name": model.name if model else None,
            "parameters": e.parameters,
            "metrics": e.metrics,
            "status": e.status,
            "framework": e.framework,
            "training_time_ms": e.training_time_ms,
            "started_at": str(e.started_at),
            "completed_at": str(e.completed_at) if e.completed_at else None,
        })

    return {"total": total, "page": page, "per_page": per_page, "items": items}
