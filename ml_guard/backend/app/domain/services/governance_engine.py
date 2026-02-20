import asyncio
import json
from fastapi.encoders import jsonable_encoder
import joblib
import pandas as pd
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
import structlog
from uuid import UUID

from app.core.celery_app import celery_app
from app.domain.services.orchestrator import TestOrchestrator
from app.infrastructure.persistence import models as sql_models
from app.infrastructure.database import SessionLocal
from app.domain.services.nlp_parser import NLPParser

logger = structlog.get_logger(__name__)

@celery_app.task(name="app.domain.services.governance_engine.run_async_evaluation")
def run_async_evaluation(
    run_id: str,
    project_id: str,
    model_version: str,
    intent: str,
    model_path: str,
    train_data_path: str,
    val_data_path: str,
    target_column: str
):
    """
    Celery background task for full model evaluation.
    """
    db = SessionLocal()
    orchestrator = TestOrchestrator()
    parser = NLPParser()
    
    try:
        # 1. Load Artifacts
        model = joblib.load(model_path)
        train_df = pd.read_csv(train_data_path)
        val_df = pd.read_csv(val_data_path)
        
        datasets = {
            "training": train_df,
            "validation": val_df
        }
        
        # 2. Parse Intent
        categories = parser.parse_query(intent)
        
        # Regression Support - Fetch baseline if needed
        baseline_model = None
        if 'regression' in categories:
            last_run = db.query(sql_models.TestRun)\
                .filter(sql_models.TestRun.project_id == project_id)\
                .filter(sql_models.TestRun.deployment_allowed == True)\
                .order_by(sql_models.TestRun.created_at.desc()).first()
            
            if last_run:
                # In a real system, we'd load the binary from a registry
                # For this demo, we assume the baseline is available or just use local cache
                logger.info("Found baseline for regression", run_id=run_id, baseline_id=last_run.id)
                # baseline_model = joblib.load(last_run.model_path) 
        
        # 3. Use an event loop to run the async orchestrator in a sync Celery worker
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(orchestrator.run_test_suite(
            project_id=project_id,
            model_version=model_version,
            test_suite_name=f"Governance Scan: {intent[:20]}...",
            model_artifact=model,
            datasets=datasets,
            categories=categories,
            target_column=target_column,
            baseline_model=baseline_model
        ))
        
        # 4. Save to Database
        # First find or create project
        project = db.query(sql_models.Project).filter(sql_models.Project.id == project_id).first()
        
        # Save results
        # Use Pydantic's model_dump(mode='json') to ensure all types are JSON-serializable
        result_data = result.model_dump(mode='json')
        
        test_run = sql_models.TestRun(
            id=run_id,
            project_id=project_id,
            model_version=model_version,
            suite_name=result.test_suite,
            score=result.score,
            deployment_allowed=result.deployment_allowed,
            summary_metrics={k: v for k, v in result_data.items() if k != 'results'},
            results_raw=result_data.get('results', [])
        )
        db.add(test_run)
        
        # Save Drift Logs specifically for time-series monitoring
        for r in result.results:
            if r.category == "statistical_stability" and r.test_id == "psi_drift":
                # Assuming details contains feature-level PSI
                for feature, score in r.details.get("psi_scores", {}).items():
                    drift_log = sql_models.DriftLog(
                        test_run_id=run_id,
                        feature_name=feature,
                        psi_score=score
                    )
                    db.add(drift_log)

        db.commit()
        logger.info("Async Evaluation Complete", run_id=run_id, score=result.score)
        
    except Exception as e:
        logger.error("Async Evaluation Failed", run_id=run_id, error=str(e))
        db.rollback()
    finally:
        db.close()

class GovernanceEngine:
    """
    Management layer for Governance platform actions.
    """
    def __init__(self, db: Session):
        self.db = db

    def list_projects(self, tenant_id: UUID) -> List[sql_models.Project]:
        return self.db.query(sql_models.Project).filter(sql_models.Project.tenant_id == tenant_id).all()

    def get_project_history(self, project_id: UUID) -> List[sql_models.TestRun]:
        return self.db.query(sql_models.TestRun).filter(sql_models.TestRun.project_id == project_id).order_by(sql_models.TestRun.created_at.desc()).all()

    def get_drift_trends(self, project_id: UUID, feature_name: Optional[str] = None):
        query = self.db.query(sql_models.DriftLog).join(sql_models.TestRun).filter(sql_models.TestRun.project_id == project_id)
        if feature_name:
            query = query.filter(sql_models.DriftLog.feature_name == feature_name)
        return query.order_by(sql_models.DriftLog.timestamp.asc()).all()

    def check_persistent_drift(self, project_id: str, threshold: float = 0.2, window: int = 3) -> bool:
        """
        FIREFLINK PHILOSOPHY: Automatic failure if drift persists.
        Returns True if PSI > threshold for 'window' consecutive runs.
        """
        # Get last 3 test runs for this project
        last_runs = self.db.query(sql_models.TestRun)\
            .filter(sql_models.TestRun.project_id == project_id)\
            .order_by(sql_models.TestRun.created_at.desc())\
            .limit(window).all()
        
        if len(last_runs) < window:
            return False
            
        failure_count = 0
        for run in last_runs:
            # Check if any feature in this run had high PSI
            high_drift = self.db.query(sql_models.DriftLog)\
                .filter(sql_models.DriftLog.test_run_id == run.id)\
                .filter(sql_models.DriftLog.psi_score > threshold).first()
            if high_drift:
                failure_count += 1
        
        return failure_count >= window
