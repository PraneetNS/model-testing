"""
Test Orchestrator - Coordinates ML test execution
FireFlink-style test suite management and parallel execution
"""

import asyncio
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import structlog

from ..core.config import settings
from ..ml_testing.engine import MLTestEngine
from ..ml_testing.test_categories import (
    DataQualityTests,
    StatisticalStabilityTests,
    ModelPerformanceTests,
    RobustnessTests,
    BiasFairnessTests
)
from ..models.test_run import TestRun, TestResult
from ..storage.test_results import TestResultsStorage

logger = structlog.get_logger(__name__)

class TestOrchestrator:
    """Orchestrates ML test execution with FireFlink-style patterns."""

    def __init__(self):
        self.test_engine = MLTestEngine()
        self.results_storage = TestResultsStorage()
        self.executor = ThreadPoolExecutor(max_workers=settings.MAX_PARALLEL_TESTS)

    async def run_test_suite(
        self,
        project_id: str,
        model_version: str,
        test_suite_name: str,
        environment: str = "development",
        model_artifact: Any = None,
        datasets: Dict[str, Any] = None,
        test_suite_config: Dict[str, Any] = None
    ) -> TestRun:
        """
        Execute a complete test suite with parallel execution and proper orchestration.

        Args:
            project_id: FireFlink project identifier
            model_version: ML model version to test
            test_suite_name: Name of test suite to execute
            environment: Target environment (development/staging/production)

        Returns:
            TestRun: Complete test execution results
        """

        run_id = f"ml-run-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:8]}"
        start_time = datetime.utcnow()

        logger.info(
            "Starting ML test suite execution",
            run_id=run_id,
            project_id=project_id,
            model_version=model_version,
            test_suite=test_suite_name,
            environment=environment
        )

        # Load test suite configuration
        if test_suite_config is None:
            test_suite_config = await self._load_test_suite_config(
                project_id, test_suite_name
            )

            if not test_suite_config:
                raise ValueError(f"Test suite '{test_suite_name}' not found in project '{project_id}'")

        # Initialize test run
        test_run = TestRun(
            run_id=run_id,
            project_id=project_id,
            model_version=model_version,
            test_suite=test_suite_name,
            environment=environment,
            status="running",
            start_time=start_time,
            results=[]
        )

        try:
            # Use provided model and datasets, or load defaults
            if model_artifact is None or datasets is None:
                model_artifact, datasets = await self._load_model_and_datasets(
                    project_id, model_version
                )

            # Execute tests in parallel with proper orchestration
            test_results = await self._execute_test_suite_parallel(
                test_suite_config, model_artifact, datasets, run_id
            )

            # Update test run with results
            end_time = datetime.utcnow()
            test_run.results = test_results
            test_run.end_time = end_time
            test_run.execution_time_seconds = (end_time - start_time).total_seconds()
            test_run.status = "completed"

            # Store results
            await self.results_storage.save_test_run(test_run)

            logger.info(
                "ML test suite execution completed",
                run_id=run_id,
                total_tests=len(test_results),
                passed=len([r for r in test_results if r.status == "passed"]),
                failed=len([r for r in test_results if r.status == "failed"]),
                execution_time_seconds=test_run.execution_time_seconds
            )

            return test_run

        except Exception as e:
            logger.error(
                "ML test suite execution failed",
                run_id=run_id,
                error=str(e)
            )
            test_run.status = "failed"
            test_run.error_message = str(e)
            test_run.end_time = datetime.utcnow()
            await self.results_storage.save_test_run(test_run)
            raise

    async def _execute_test_suite_parallel(
        self,
        test_suite_config: Dict[str, Any],
        model_artifact: Any,
        datasets: Dict[str, Any],
        run_id: str
    ) -> List[TestResult]:
        """
        Execute tests in parallel with proper dependency management.

        Args:
            test_suite_config: Test suite configuration
            model_artifact: Loaded ML model
            datasets: Dictionary of loaded datasets
            run_id: Test run identifier

        Returns:
            List[TestResult]: All test results
        """

        all_results = []

        # Group tests by category for parallel execution
        test_groups = self._group_tests_by_category(test_suite_config)

        # Execute each category (some may have dependencies)
        for category_name, tests in test_groups.items():
            logger.debug(f"Executing {category_name} tests", test_count=len(tests))

            # Execute tests in this category in parallel
            category_results = await self._execute_category_tests_parallel(
                category_name, tests, model_artifact, datasets, run_id
            )

            all_results.extend(category_results)

        return all_results

    async def _execute_category_tests_parallel(
        self,
        category: str,
        tests: List[Dict[str, Any]],
        model_artifact: Any,
        datasets: Dict[str, Any],
        run_id: str
    ) -> List[TestResult]:
        """Execute all tests in a category in parallel."""

        async def execute_single_test(test_config: Dict[str, Any]) -> TestResult:
            """Execute a single test in a thread pool."""
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                self._execute_test_sync,
                test_config, model_artifact, datasets, run_id
            )

        # Create tasks for parallel execution
        tasks = [execute_single_test(test) for test in tests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions and convert to TestResult objects
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create failed test result for exceptions
                test_config = tests[i]
                final_results.append(TestResult(
                    test_id=f"{run_id}-{category}-{i}",
                    test_name=test_config.get("name", f"Test {i}"),
                    category=category,
                    status="failed",
                    severity=test_config.get("severity", "medium"),
                    message=f"Test execution failed: {str(result)}",
                    execution_time_seconds=0.0,
                    details={"error": str(result)}
                ))
            else:
                final_results.append(result)

        return final_results

    def _execute_test_sync(
        self,
        test_config: Dict[str, Any],
        model_artifact: Any,
        datasets: Dict[str, Any],
        run_id: str
    ) -> TestResult:
        """Execute a single test synchronously (called from thread pool)."""

        test_name = test_config.get("name", "Unnamed Test")
        category = test_config.get("category", "unknown")

        start_time = datetime.utcnow()

        try:
            # Route to appropriate test category engine
            if category == "data_quality":
                result = DataQualityTests().run_test(test_config, datasets)
            elif category == "statistical_stability":
                result = StatisticalStabilityTests().run_test(test_config, datasets)
            elif category == "model_performance":
                result = ModelPerformanceTests().run_test(test_config, model_artifact, datasets)
            elif category == "robustness":
                result = RobustnessTests().run_test(test_config, model_artifact, datasets)
            elif category == "bias_fairness":
                result = BiasFairnessTests().run_test(test_config, model_artifact, datasets)
            else:
                raise ValueError(f"Unknown test category: {category}")

            execution_time = (datetime.utcnow() - start_time).total_seconds()

            return TestResult(
                test_id=f"{run_id}-{category}-{hash(test_name) % 1000}",
                test_name=test_name,
                category=category,
                status=result.get("status", "failed"),
                severity=test_config.get("severity", "medium"),
                message=result.get("message", "Test completed"),
                execution_time_seconds=execution_time,
                details=result.get("details", {}),
                threshold=result.get("threshold"),
                actual_value=result.get("actual_value")
            )

        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(
                "Test execution failed",
                test_name=test_name,
                category=category,
                error=str(e)
            )

            return TestResult(
                test_id=f"{run_id}-{category}-{hash(test_name) % 1000}",
                test_name=test_name,
                category=category,
                status="failed",
                severity=test_config.get("severity", "high"),
                message=f"Test execution failed: {str(e)}",
                execution_time_seconds=execution_time,
                details={"error": str(e), "traceback": str(e.__traceback__)}
            )

    def _group_tests_by_category(self, test_suite_config: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Group tests by category for parallel execution."""
        groups = {}
        for test in test_suite_config.get("tests", []):
            category = test.get("category", "unknown")
            if category not in groups:
                groups[category] = []
            groups[category].append(test)
        return groups

    async def _load_test_suite_config(self, project_id: str, test_suite_name: str) -> Optional[Dict[str, Any]]:
        """Load test suite configuration from storage."""
        # In a real implementation, this would load from database/file storage
        # For now, return a mock configuration
        return self._get_default_test_suite_config(test_suite_name)

    async def _load_model_and_datasets(self, project_id: str, model_version: str) -> tuple:
        """Load model artifact and datasets."""
        # In a real implementation, this would load from model registry and storage
        # For now, return mock objects
        return None, {}

    def _get_default_test_suite_config(self, suite_name: str) -> Dict[str, Any]:
        """Get default test suite configuration for production readiness."""

        if suite_name == "production-readiness":
            return {
                "name": "Production Readiness",
                "description": "Comprehensive ML model validation for production deployment",
                "tests": [
                    # Data Quality Tests
                    {
                        "name": "Missing Values Check",
                        "category": "data_quality",
                        "severity": "high",
                        "config": {"threshold": 0.05}
                    },
                    {
                        "name": "Duplicate Rows Check",
                        "category": "data_quality",
                        "severity": "medium",
                        "config": {"allow_duplicates": False}
                    },
                    {
                        "name": "Class Balance Check",
                        "category": "data_quality",
                        "severity": "medium",
                        "config": {"max_imbalance_ratio": 10.0}
                    },

                    # Statistical Stability Tests
                    {
                        "name": "PSI Drift Check",
                        "category": "statistical_stability",
                        "severity": "high",
                        "config": {"psi_threshold": 0.1}
                    },
                    {
                        "name": "Feature Correlation Stability",
                        "category": "statistical_stability",
                        "severity": "medium",
                        "config": {"correlation_threshold": 0.1}
                    },

                    # Model Performance Tests
                    {
                        "name": "Accuracy Threshold",
                        "category": "model_performance",
                        "severity": "critical",
                        "config": {"metric": "accuracy", "threshold": 0.85, "operator": "gte"}
                    },
                    {
                        "name": "Precision Threshold",
                        "category": "model_performance",
                        "severity": "high",
                        "config": {"metric": "precision", "threshold": 0.80, "operator": "gte"}
                    },
                    {
                        "name": "Recall Threshold",
                        "category": "model_performance",
                        "severity": "high",
                        "config": {"metric": "recall", "threshold": 0.75, "operator": "gte"}
                    },

                    # Bias & Fairness Tests
                    {
                        "name": "Gender Bias Check",
                        "category": "bias_fairness",
                        "severity": "high",
                        "config": {
                            "protected_attribute": "gender",
                            "fairness_metric": "disparate_impact",
                            "threshold": 1.2
                        }
                    }
                ]
            }

        return None