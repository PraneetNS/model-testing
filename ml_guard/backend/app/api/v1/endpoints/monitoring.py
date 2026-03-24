from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import uuid
import structlog
from datetime import datetime

from app.api.v1 import deps
from app.infrastructure.persistence import models as sql_models
from pydantic import BaseModel

logger = structlog.get_logger(__name__)
router = APIRouter()

class PredictionBatch(BaseModel):
    project_id: str
    model_version: str
    predictions: List[Dict[str, Any]] # List of features + prediction

@router.post("/predictions/log")
async def log_predictions(
    batch: PredictionBatch,
    db: Session = Depends(deps.get_db),
    current_user: sql_models.User = Depends(deps.get_current_active_user)
):
    """
    Live prediction logging endpoint.
    Optimized for batch ingestion to minimize DB overhead.
    """
    try:
        logs = []
        for p in batch.predictions:
            # Separate features from prediction if possible, or store as provided
            prediction_val = p.pop("prediction", None)
            
            logs.append(sql_models.PredictionLog(
                project_id=batch.project_id,
                model_version=batch.model_version,
                features=p,
                prediction={"value": prediction_val}
            ))
        
        db.add_all(logs)
        db.commit()
        
        logger.info("Batch predictions logged", project_id=batch.project_id, count=len(logs))
        return {"status": "success", "logged_count": len(logs)}
    except Exception as e:
        logger.error("Logging failed", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to log predictions")

@router.get("/drift/history/{project_id}")
async def get_drift_history(
    project_id: str,
    feature: Optional[str] = None,
    limit: int = 100,
    db: Session = Depends(deps.get_db),
    current_user: sql_models.User = Depends(deps.get_current_active_user)
):
    """
    Fetches historical drift metrics for the dashboard.
    """
    query = db.query(sql_models.DriftLog).join(sql_models.MonitoringJob)\
              .filter(sql_models.MonitoringJob.project_id == project_id)
    
    if feature:
        query = query.filter(sql_models.DriftLog.feature_name == feature)
    
    results = query.order_by(sql_models.DriftLog.timestamp.desc()).limit(limit).all()
    
    return [
        {
            "feature": r.feature_name,
            "metric": r.metric_type,
            "value": r.metric_value,
            "is_drifted": r.is_drifted,
            "timestamp": r.timestamp
        }
        for r in results
    ]

from fastapi import UploadFile, File, Form
import pandas as pd
import io
from app.domain.services.drift_engine import DriftEngine

@router.post("/batch")
async def monitor_batch(
    project_id: str = Form(...),
    reference_file: UploadFile = File(...),
    production_file: UploadFile = File(...),
    target_column: str = Form("churn"),
    db: Session = Depends(deps.get_db),
    current_user: sql_models.User = Depends(deps.get_current_active_user)
):
    """
    Live batch monitoring.
    Computes PSI, KS test, Chi-square, Correlation Shift, and Target Drift.
    """
    try:
        ref_df = pd.read_csv(io.BytesIO(await reference_file.read()))
        prod_df = pd.read_csv(io.BytesIO(await production_file.read()))

        drift_engine = DriftEngine()
        drift_report = drift_engine.detect_drift(ref_df, prod_df, target_column)

        # Map to requested response format
        return {
            "status": "success",
            "project_id": project_id,
            "global_risk_score": drift_report.get("risk_score"),
            "severity": drift_report.get("severity").upper(),
            "feature_breakdown": drift_report.get("feature_drift", {}),
            "correlation_shift": drift_report.get("correlation_shift_score"),
            "target_drift": drift_report.get("target_drift"),
            "summary": drift_report.get("summary")
        }
    except Exception as e:
        logger.error("Drift Monitoring failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
