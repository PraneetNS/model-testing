import asyncio
import json
import os
from fastapi.encoders import jsonable_encoder
import joblib
import pandas as pd
from typing import Dict, Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
import structlog
from uuid import UUID
from datetime import datetime

from app.core.celery_app import celery_app
from app.domain.services.orchestrator import TestOrchestrator
from app.db import models as sql_models
from app.db.session import SessionLocal
from app.domain.services.nlp_parser import NLPParser

logger = structlog.get_logger(__name__)

from app.domain.services.trainer import Trainer

@celery_app.task(name="app.domain.services.governance_engine.run_async_training", bind=True, max_retries=3, default_retry_delay=10)
def run_async_training(
    job_id: str,
    data_path: str,
    target_column: str,
    model_type: str,
    test_size: float,
    do_cv: bool
):
    """
    Background task for model training.
    """
    trainer = Trainer()
    try:
        df = pd.read_csv(data_path)
        result = trainer.train_model(
            df=df,
            target_column=target_column,
            model_type=model_type,
            test_size=test_size,
            do_cv=do_cv
        )
        # In a real app, we might save this result to a Job table or Redis
        logger.info("Async Training Complete", job_id=job_id, status=result.get("status"))
        return result
    except Exception as e:
        logger.error("Async Training Failed", job_id=job_id, error=str(e))
        return {"status": "error", "message": str(e)}
    finally:
        # Cleanup temp file
        if os.path.exists(data_path):
            os.remove(data_path)

@celery_app.task(name="app.domain.services.governance_engine.run_async_evaluation", bind=True, max_retries=3, default_retry_delay=10)
async def run_async_evaluation(
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
        
        # 3. Run async orchestrator in sync Celery worker using asyncio.run()
        result = asyncio.run(orchestrator.run_test_suite(
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
        project = (await db.execute(select(sql_models.Project).filter(sql_models.Project.id == project_id))).scalars().first()
        
        # Save results
        # Use Pydantic's model_dump(mode='json') to ensure all types are JSON-serializable
        result_data = result.model_dump(mode='json')
        
        test_run = sql_models.TestRun(
            id=run_id,
            project_id=project_id,
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
                        metric_type="PSI",
                        metric_value=score,
                        is_drifted=score > 0.1
                    )
                    db.add(drift_log)

        await db.commit()
        logger.info("Async Evaluation Complete", run_id=run_id, score=result.score)
        
    except Exception as e:
        logger.error("Async Evaluation Failed", run_id=run_id, error=str(e))
        db.rollback()
    finally:
        db.close()

@celery_app.task(name="app.domain.services.governance_engine.run_scheduled_monitoring", bind=True, max_retries=3, default_retry_delay=10)
async def run_scheduled_monitoring(job_id: str):
    """
    Background worker task for scheduled drift detection.
    Analyzes PredictionLogs against historical baselines.
    """
    db = SessionLocal()
    try:
        job = (await db.execute(select(sql_models.MonitoringJob).filter(sql_models.MonitoringJob.id == job_id))).scalars().first()
        if not job or not job.is_active:
            return
            
        logger.info("Running scheduled monitoring scan", job_id=job_id, project=job.project_id)
        
        # 1. Fetch Reference Data (from successful TestRuns)
        # 2. Fetch Recent Prediction Logs (PredictionLog)
        # 3. Calculate PSI/KS using the Framework Engines
        # 4. Save results to DriftLog (with monitoring_job_id)
        # 5. Trigger alerting hooks if drift detected
        
        job.last_run = datetime.now()
        await db.commit()
    except Exception as e:
        logger.error("Monitoring job failed", job_id=job_id, error=str(e))
    finally:
        db.close()

class GovernanceEngine:
    """
    Management layer for Governance platform actions.
    """
    def __init__(self, db: AsyncSession):
        self.db = db

    async def list_projects(self, tenant_id: UUID) -> List[sql_models.Project]:
        return (await db.execute(select(sql_models.Project).filter(sql_models.Project.tenant_id == tenant_id))).scalars().all()

    async def get_project_history(self, project_id: UUID) -> List[sql_models.TestRun]:
        return (await db.execute(select(sql_models.TestRun).filter(sql_models.TestRun.project_id == project_id).order_by(sql_models.TestRun.created_at.desc()))).scalars().all()

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
                .filter(sql_models.DriftLog.metric_value > threshold).first()
            if high_drift:
                failure_count += 1
        
        return failure_count >= window

    def evaluate_active_policy(self, metrics: dict, org_id: Optional[str] = None) -> dict:
        """
        Fetches the active PolicyRule for the given org (or global default)
        and evaluates the provided metrics against it.
        """
        from ml_guard.core.policy import evaluate_policy

        # 1. Fetch active policy rule
        # Use our new PolicyRule model
        policy_rule = self.db.query(sql_models.PolicyRule)\
            .filter(sql_models.PolicyRule.is_active == True)
        
        if org_id:
            policy_rule = policy_rule.filter(sql_models.PolicyRule.org_id == org_id)
        
        active_policy = policy_rule.order_by(sql_models.PolicyRule.created_at.desc()).first()

        # 2. Extract rules_json or use default
        rules = active_policy.rules_json if active_policy else None
        
        # 3. Use core policy engine for evaluation
        # mapping metrics to what evaluate_policy expects
        # evaluate_policy expects: metrics, drift_report, overfitting_gap, calibration, stability_score, governance_score, policy
        
        result = evaluate_policy(
            metrics=metrics.get("metrics"),
            drift_report=metrics.get("drift"),
            overfitting_gap=metrics.get("overfitting_gap"),
            calibration=metrics.get("calibration"),
            stability_score=metrics.get("stability_score"),
            governance_score=metrics.get("governance_score"),
            policy=rules
        )
        
        if active_policy:
            result["policy_name"] = active_policy.name
            result["policy_id"] = str(active_policy.id)
        else:
            result["policy_name"] = "Default (No Active Policy Found)"
            
        return result

