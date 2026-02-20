
import asyncio
from typing import List, Dict, Any, Optional
from app.domain.models.test_suite import TestConfig, TestResult, QualityGateResult, TestSuite
from app.domain.services.validation_engine import ValidationEngine
import uuid
from datetime import datetime
import structlog

logger = structlog.get_logger(__name__)

class TestOrchestrator:
    """
    Orchestrates the execution of a test suite.
    Now more feature-rich and aligned with production requirements.
    """
    def __init__(self):
        self.validation_engine = ValidationEngine()

    async def run_test_suite(
        self, 
        project_id: str, 
        model_version: str, 
        test_suite_name: str,
        model_artifact: Any = None,
        datasets: Dict[str, Any] = None,
        test_suite_config: Optional[Dict] = None,
        categories: Optional[List[str]] = None,
        target_column: str = "target"
    ) -> QualityGateResult:
        
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        logger.info("Starting test orchestration", run_id=run_id, categories=categories)
        
        # Build dynamic suite if config is missing but we have intent or categories
        if not test_suite_config:
            if categories:
                test_suite_config = self._build_suite_from_categories(categories, target_column)
            else:
                test_suite_config = self._build_default_suite(target_column)

        results = []
        tests_to_run = test_suite_config.get("tests", [])
        
        if not tests_to_run:
            logger.warning("No tests found in suite configuration")

        for test in tests_to_run:
            try:
                result = await self.validation_engine.run_test(test, model_artifact, datasets)
                results.append(TestResult(
                    test_id=result.test_id,
                    test_name=result.test_name,
                    category=result.category,
                    status=result.status,
                    severity=result.severity,
                    message=result.message,
                    execution_time_seconds=result.execution_time_seconds,
                    actual_value=result.actual_value,
                    threshold=result.threshold,
                    details=result.details,
                    description=test.get("description", "No description provided.")
                ))
            except Exception as e:
                logger.error("Test execution failed", test=test.get("name"), error=str(e))
                results.append(TestResult(
                    test_id=str(uuid.uuid4()),
                    test_name=test.get("name", "Unknown"),
                    category=test.get("category", "error"),
                    status="failed",
                    severity=test.get("severity", "medium"),
                    message=f"Execution error: {str(e)}",
                    execution_time_seconds=0,
                    description=test.get("description", "No description provided.")
                ))

        # Calculate a weighted Quality Index
        # Weights: Critical=10, High=5, Medium=2, Low=1
        weights = {"critical": 10, "high": 5, "medium": 2, "low": 1, "error": 5}
        total_possible_weight = sum(weights.get(r.severity, 1) for r in results)
        actual_weight = sum(weights.get(r.severity, 1) for r in results if r.status == "passed")
        
        score = (actual_weight / total_possible_weight) * 100 if total_possible_weight > 0 else 0

        # Strict Gate: No critical failures AND score must be >= 70
        critical_failures = [r for r in results if r.status == "failed" and r.severity == "critical"]
        deployment_allowed = len(critical_failures) == 0 and score >= 70

        logger.info("Orchestration complete", run_id=run_id, score=score, allowed=deployment_allowed)

        return QualityGateResult(
            run_id=run_id,
            project_id=project_id,
            model_version=model_version,
            test_suite=test_suite_name,
            score=score,
            deployment_allowed=deployment_allowed,
            results=results
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
                }
            ],
            'data_quality': [
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
