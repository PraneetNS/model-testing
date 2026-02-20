from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.encoders import jsonable_encoder
import json
from typing import List, Optional
import pandas as pd
import io
import joblib
import os
import uuid
import structlog
from datetime import datetime
from sqlalchemy.orm import Session

from app.api.v1 import deps
from app.infrastructure.persistence import models as sql_models
from app.domain.services.orchestrator import TestOrchestrator
from app.domain.services.nlp_parser import NLPParser
from app.domain.services.trainer import Trainer
from app.domain.services.governance_engine import run_async_evaluation

logger = structlog.get_logger(__name__)
router = APIRouter()
orchestrator = TestOrchestrator()
nlp_parser = NLPParser()
trainer_service = Trainer()

# Temporary storage for async jobs (in production, use S3/Minio)
UPLOAD_DIR = "temp_uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

@router.post("/evaluate")
async def evaluate_model(
    project_id: str = Form(...),
    model_file: UploadFile = File(...),
    train_file: UploadFile = File(...),
    val_file: UploadFile = File(...),
    target_column: str = Form("churn"),
    query: Optional[str] = Form(None),
    db: Session = Depends(deps.get_db),
    current_user: sql_models.User = Depends(deps.get_current_active_user)
):
    """
    Evaluate an uploaded model and persist the results in the governance database.
    Now tied to the logged-in user and their tenant.
    """
    try:
        # 1. Load Data
        model = joblib.load(io.BytesIO(await model_file.read()))
        train_df = pd.read_csv(io.BytesIO(await train_file.read()))
        val_df = pd.read_csv(io.BytesIO(await val_file.read()))
        
        datasets = {"training": train_df, "validation": val_df}

        # 2. Parse Intent
        categories = nlp_parser.parse_query(query) if query else ["accuracy", "data_quality", "drift"]

        # Regression Support - Fetch baseline if needed
        baseline_model = None
        if 'regression' in categories:
            last_run = db.query(sql_models.TestRun)\
                .filter(sql_models.TestRun.project_id == project_id)\
                .filter(sql_models.TestRun.deployment_allowed == True)\
                .order_by(sql_models.TestRun.created_at.desc()).first()
            
            if last_run:
                # In a real system, we'd load the binary from a registry
                logger.info("Found baseline for regression", baseline_id=last_run.id)
                # baseline_model = joblib.load(last_run.model_path) 

        # 3. Synchronous Orchestration (Quick feedback)
        result = await orchestrator.run_test_suite(
            project_id=project_id,
            model_version="v1.0.0",
            test_suite_name="Active Governance Scan",
            model_artifact=model,
            datasets=datasets,
            categories=categories,
            target_column=target_column,
            baseline_model=baseline_model
        )

        # 4. Persistence
        # Ensure project exists
        project = db.query(sql_models.Project).filter(sql_models.Project.id == project_id).first()
        if not project:
            project = sql_models.Project(id=project_id, name=f"Project {project_id[:8]}", tenant_id=current_user.tenant_id)
            db.add(project)
            db.commit()

        # Save results
        # Use Pydantic's model_dump(mode='json') to ensure all types are JSON-serializable
        result_data = result.model_dump(mode='json')
        
        test_run = sql_models.TestRun(
            project_id=project_id,
            model_version="v1.0.0",
            suite_name=result.test_suite,
            score=result.score,
            deployment_allowed=result.deployment_allowed,
            summary_metrics={k: v for k, v in result_data.items() if k != 'results'},
            results_raw=result_data.get('results', [])
        )
        db.add(test_run)
        
        # Save Drift Metrics for monitoring
        for r in result.results:
            # Check for high drift for reporting if needed
            if "psi" in r.test_name.lower() and r.actual_value:
                 db.add(sql_models.DriftLog(test_run_id=test_run.id, feature_name="combined_drift", psi_score=r.actual_value))

        db.commit()
        return result

    except Exception as e:
        logger.error("Evaluation error", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/evaluate/async")
async def evaluate_model_async(
    project_id: str = Form(...),
    model_file: UploadFile = File(...),
    train_file: UploadFile = File(...),
    val_file: UploadFile = File(...),
    target_column: str = Form("churn"),
    query: Optional[str] = Form(None),
    current_user: sql_models.User = Depends(deps.get_current_active_user)
):
    """
    Trigger a background evaluation job. Suitable for massive datasets.
    """
    run_id = str(uuid.uuid4())
    
    # Save files to disk for worker to pick up
    # In a real enterprise app, these go to S3
    m_path = os.path.join(UPLOAD_DIR, f"{run_id}_model.pkl")
    t_path = os.path.join(UPLOAD_DIR, f"{run_id}_train.csv")
    v_path = os.path.join(UPLOAD_DIR, f"{run_id}_val.csv")
    
    with open(m_path, "wb") as f: f.write(await model_file.read())
    with open(t_path, "wb") as f: f.write(await train_file.read())
    with open(v_path, "wb") as f: f.write(await val_file.read())
    
    run_async_evaluation.delay(
        run_id=run_id,
        project_id=project_id,
        model_version="v1.0.0-async",
        intent=query or "Default robust scan",
        model_path=m_path,
        train_data_path=t_path,
        val_data_path=v_path,
        target_column=target_column
    )
    
    return {"status": "accepted", "job_id": run_id, "message": "Evaluation started in background."}

@router.post("/train")
async def train_new_model(
    dataset: UploadFile = File(...),
    target_column: str = Form(...),
    model_type: str = Form("random_forest"),
    test_size: float = Form(0.2),
    do_cv: bool = Form(True),
    current_user: sql_models.User = Depends(deps.get_current_active_user)
):
    """
    Train a model with identity tracking.
    """
    df = pd.read_csv(io.BytesIO(await dataset.read()))
    result = trainer_service.train_model(df=df, target_column=target_column, model_type=model_type, test_size=test_size, do_cv=do_cv)
    return result
