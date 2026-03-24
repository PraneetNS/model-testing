
import asyncio
from typing import List, Dict, Any, Optional
from app.domain.models.test_suite import TestConfig, TestResult, QualityGateResult, TestSuite
from app.domain.services.validation_engine import ValidationEngine
import structlog
import numpy as np
import uuid
from datetime import datetime

logger = structlog.get_logger(__name__)

from app.domain.services.risk_engine import RiskEngine
from app.domain.services.fingerprinting import FingerprintingService
from app.domain.services.profiler import ModelProfiler
from app.domain.services.notifications import NotificationService
from app.domain.services.ml_testing.framework.runner import MLTestRunner
from app.domain.services.explainability import ExplainabilityEngine
import pandas as pd
import hashlib
import platform
import sys

class TestOrchestrator:
    """
    Orchestrates the execution of a test suite.
    Upgraded for Tier 1: Risk Scoring, Fingerprinting, and Intelligent Profiling.
    Includes Tier 2: Real-time Alerting.
    """
    def __init__(self):
        self.runner = MLTestRunner()
        self.explainer = ExplainabilityEngine()
        self.risk_engine = RiskEngine()
        self.profiler = ModelProfiler()
        self.notifier = NotificationService()

    async def run_test_suite(
        self, 
        # ... existing parameters ...
        project_id: str, 
        model_version: str, 
        test_suite_name: str,
        model_artifact: Any = None,
        datasets: Dict[str, Any] = None,
        test_suite_config: Optional[Dict] = None,
        categories: Optional[List[str]] = None,
        target_column: str = "target",
        baseline_model: Any = None,
        baseline_datasets: Dict[str, Any] = None
    ) -> QualityGateResult:
        
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        repro_token = hashlib.sha256(f"{run_id}-{datetime.now().isoformat()}".encode()).hexdigest()[:16]
        
        logger.info("Starting production-grade test orchestration", run_id=run_id, repro_token=repro_token)
        
        # Tier 1: Intelligent Model Profiling
        model_profile = {}
        if model_artifact is not None and datasets and "validation" in datasets:
            model_profile = self.profiler.profile_artifacts(model_artifact, datasets["validation"], target_column)
            logger.info("Model profiling complete", risk=model_profile.get("suggested_risk"))

        # Tier 1: Dataset Fingerprinting
        dataset_metadata = {}
        # ... rest of the logic ...
        if datasets:
            for ds_name, df in datasets.items():
                if isinstance(df, pd.DataFrame):
                    fingerprint = FingerprintingService.generate_fingerprint(df)
                    schema = FingerprintingService.extract_schema(df)
                    dataset_metadata[ds_name] = {
                        "fingerprint": fingerprint,
                        "schema": schema,
                        "rows": len(df)
                    }
                    logger.info("Dataset fingerprinted", dataset=ds_name, fingerprint=fingerprint)

        if not test_suite_config:
            if categories:
                test_suite_config = self._build_suite_from_categories(categories, target_column)
            else:
                test_suite_config = self._build_default_suite(target_column)

        # Execute suite using the Framework Runner
        start_time = datetime.now()
        report = await self.runner.run_suite(
            suite_config=test_suite_config,
            model=model_artifact,
            datasets=datasets,
            baseline_model=baseline_model,
            baseline_datasets=baseline_datasets
        )
        end_time = datetime.now()

        results = []
        raw_test_results = []
        for r in report.results:
            results.append(TestResult(
                test_id=str(uuid.uuid4())[:8],
                test_name=r.name,
                category=r.name.lower(),
                status=r.status.value,
                severity=r.severity.value,
                message=r.explanation,
                execution_time_seconds=r.execution_time,
                actual_value=r.metric_value,
                threshold=r.threshold,
                details=r.details,
                description=r.description,
                remediation=r.remediation,
                explanation=r.explanation
            ))
            raw_test_results.append({
                "status": r.status.value,
                "severity": r.severity.value,
                "name": r.name
            })

        # Tier 1: Weighted Quality Scoring
        risk_evaluation = self.risk_engine.calculate_score(raw_test_results)
        score = risk_evaluation["score"]
        deployment_allowed = risk_evaluation["deployment_allowed"]
        risk_level = risk_evaluation["risk_level"]

        # Tier 1: Reproducibility Metadata
        environment_config = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "library_versions": {
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "sklearn": "1.4+" # Or use pkg_resources
            }
        }
        
        execution_metadata = {
            "duration_seconds": (end_time - start_time).total_seconds(),
            "dataset_fingerprints": dataset_metadata,
            "worker_node": platform.node()
        }

        # 2. Explainability (Feature Importance) - Tier 2
        feature_importance = []
        if model_artifact is not None and datasets and "validation" in datasets:
            try:
                val_df = datasets["validation"]
                X = val_df.drop(columns=[target_column]) if target_column in val_df.columns else val_df
                X_numeric = X.select_dtypes(include=[np.number])
                if not X_numeric.empty:
                    feature_importance = await self.explainer.get_feature_importance(model_artifact, X_numeric)
            except Exception as e:
                logger.warning("Explainability skipped", error=str(e))

        # Tier 2: Real-time Alerting
        self.notifier.send_alert({
            "event": "QUALITY_GATE_COMPLETED",
            "project": project_id,
            "severity": risk_level,
            "details": {
                "score": score,
                "allowed": deployment_allowed,
                "run_id": run_id
            }
        })

        logger.info("Orchestration complete", run_id=run_id, score=score, risk=risk_level)

        return QualityGateResult(
            run_id=run_id,
            project_id=project_id,
            model_version=model_version,
            test_suite=test_suite_name,
            score=score,
            deployment_allowed=deployment_allowed,
            results=results,
            scoring_breakdown=risk_evaluation.get("breakdown", {}),
            feature_importance=feature_importance,
            reproducibility_token=repro_token,
            risk_level=risk_level,
            environment_config=environment_config,
            execution_metadata=execution_metadata,
            model_profile=model_profile
        )

    def _build_suite_from_categories(self, categories: List[str], target_column: str) -> Dict:
        """Create comprehensive test suite configuration based on selected categories."""
        
        full_library = {
            'accuracy': [
                {
                    "name": "Accuracy Threshold (Critical)",
                    "category": "model_performance",
                    "type": "accuracy_threshold",
                    "severity": "critical",
                    "description": "Verifies that model accuracy meets the minimum acceptable standard (80%).",
                    "config": {"threshold": 0.80, "operator": "gte", "dataset": "validation", "target_column": target_column}
                },
                {
                    "name": "Precision (Weighted)",
                    "category": "model_performance",
                    "type": "precision_threshold",
                    "severity": "high",
                    "description": "Checks weighted precision to ensure false positives are minimized.",
                    "config": {"threshold": 0.75, "operator": "gte", "dataset": "validation", "target_column": target_column}
                },
                {
                    "name": "Recall (Weighted)",
                    "category": "model_performance",
                    "type": "recall_threshold",
                    "severity": "high",
                    "description": "Checks weighted recall to ensure false negatives are minimized.",
                    "config": {"threshold": 0.75, "operator": "gte", "dataset": "validation", "target_column": target_column}
                },
                {
                    "name": "F1 Score Macro",
                    "category": "model_performance",
                    "type": "f1_threshold",
                    "severity": "medium",
                    "description": "Evaluates the harmonic mean of precision and recall for balanced performance.",
                    "config": {"threshold": 0.70, "operator": "gte", "dataset": "validation", "target_column": target_column}
                }
            ],
            'performance': [
                 {
                    "name": "ROC AUC Score",
                    "category": "model_performance",
                    "type": "roc_auc_threshold",
                    "severity": "high",
                    "description": "Measures the model's ability to distinguish between classes (Area Under Curve).",
                    "config": {"threshold": 0.85, "operator": "gte", "dataset": "validation", "target_column": target_column}
                },
                {
                    "name": "Overfitting Gap Analysis",
                    "category": "model_performance",
                    "type": "overfitting_gap",
                    "severity": "critical",
                    "description": "Ensures model validation accuracy doesn't trail train accuracy by > 10%.",
                    "config": {"max_gap": 0.10, "dataset": "validation"}
                }
            ],
            'data_quality': [
                {
                    "name": "Schema Validation",
                    "category": "data_quality",
                    "type": "schema_validation",
                    "severity": "critical",
                    "description": "Verifies feature names, counts, and target existence.",
                    "config": {"target_column": target_column}
                },
                {
                    "name": "Dataset Statistical Profiling",
                    "category": "data_quality",
                    "type": "dataset_profiling",
                    "severity": "medium",
                    "description": "Profiles mean, std, skew, kurtosis, missing %, entropy.",
                    "config": {}
                },
                {
                    "name": "Missing Values (Val)",
                    "category": "data_quality",
                    "type": "missing_values",
                    "severity": "high",
                    "description": "Ensures validation data is clean and does not exceed 2% missing values.",
                    "config": {"threshold": 0.02, "dataset": "validation"}
                },
                {
                    "name": "Missing Values (Train)",
                    "category": "data_quality",
                    "type": "missing_values",
                    "severity": "medium",
                    "description": "Ensures training data is clean and does not exceed 5% missing values.",
                    "config": {"threshold": 0.05, "dataset": "training"}
                },
                {
                    "name": "Duplicate Rows Check",
                    "category": "data_quality",
                    "type": "duplicate_rows",
                    "severity": "medium",
                    "description": "Detects exact duplicate rows which can cause data leakage.",
                    "config": {"allow_duplicates": False, "dataset": "validation"}
                },
                {
                    "name": "Class Imbalance Ratio",
                    "category": "data_quality",
                    "type": "class_balance",
                    "severity": "high",
                    "description": "Checks if the target class distribution is balanced (ratio < 3.0).",
                    "config": {"max_imbalance_ratio": 3.0, "target_column": target_column, "dataset": "training"}
                }
            ],
            'bias': [
                {
                    "name": "Gender Parity Difference",
                    "category": "bias_fairness",
                    "type": "disparate_impact",
                    "severity": "critical",
                    "description": "Ensures model predictions are fair across gender groups (Disparate Impact).",
                    "config": {"protected_attribute": "gender", "threshold": 1.15, "dataset": "validation", "target_column": target_column}
                },
                {
                    "name": "Age Group Fairness",
                    "category": "bias_fairness",
                    "type": "disparate_impact",
                    "severity": "medium",
                    "description": "Checks for prediction bias across different age brackets.",
                    "config": {"protected_attribute": "age", "threshold": 1.25, "dataset": "validation", "target_column": target_column}
                }
            ],
            'drift': [
                {
                    "name": "Population Stability Index (PSI)",
                    "category": "statistical_stability",
                    "type": "psi_drift",
                    "severity": "critical",
                    "description": "Measures if the population distribution has shifted significantly (Drift).",
                    "config": {"psi_threshold": 0.1, "dataset": "validation"}
                },
                {
                    "name": "KS Test Drift Check",
                    "category": "statistical_stability",
                    "type": "ks_test",
                    "severity": "high",
                    "description": "Kolmogorov-Smirnov test to detect feature distribution changes.",
                    "config": {"p_value_threshold": 0.05}
                }
            ],
            'stability': [
                 {
                    "name": "Correlation Stability",
                    "category": "statistical_stability",
                    "type": "correlation_stability",
                    "severity": "medium",
                    "description": "Verifies that feature correlations with target remain stable.",
                    "config": {"threshold": 0.9}
                }
            ],
            'robustness': [
                {
                    "name": "Global Prediction Stability",
                    "category": "robustness",
                    "type": "prediction_stability",
                    "severity": "high",
                    "description": "Tests if model predictions remain stable under minor noise.",
                    "config": {"stability_threshold": 0.98, "noise_level": 0.005, "dataset": "validation"}
                }
            ],
            'stress_test': [
                {
                    "name": "Adversarial Noise Resistance",
                    "category": "robustness",
                    "type": "input_perturbation",
                    "severity": "high",
                    "description": "Stress tests the model against adversarial noise injection.",
                    "config": {"perturbation_factor": 0.05, "sensitivity_threshold": 0.05, "dataset": "validation"}
                },
                {
                    "name": "Extreme Value Stress Test",
                    "category": "robustness",
                    "type": "input_perturbation",
                    "severity": "medium",
                    "description": "Checks model behavior when inputs are pushed to extreme values.",
                    "config": {"perturbation_factor": 0.2, "sensitivity_threshold": 0.15, "dataset": "validation"}
                }
            ],
            'regression': [
                {
                    "name": "Accuracy Regression Check",
                    "category": "model_performance",
                    "type": "regression_check",
                    "severity": "critical",
                    "description": "Ensures new model does not regress in accuracy by more than 5% compared to baseline.",
                    "config": {"max_drop": -0.05, "target_column": target_column}
                }
            ]
        }

        selected_configs = []
        for cat in categories:
            if cat in full_library:
                selected_configs.extend(full_library[cat])

        return {
            "name": f"Strategic Scan: {', '.join(categories).title()}",
            "tests": selected_configs
        }

    def _build_default_suite(self, target_column: str = "target") -> Dict:
        """Returns a baseline production-readiness suite."""
        return self._build_suite_from_categories(['accuracy', 'data_quality', 'drift'], target_column)
