"""
ML Test Engine - Main testing coordinator
"""

from typing import Dict, Any, List
import structlog

from .test_categories import (
    DataQualityTests,
    StatisticalStabilityTests,
    ModelPerformanceTests,
    RobustnessTests,
    BiasFairnessTests
)

logger = structlog.get_logger(__name__)

class MLTestEngine:
    """Main ML testing engine that coordinates all test categories."""

    def __init__(self):
        self.data_quality_tests = DataQualityTests()
        self.statistical_stability_tests = StatisticalStabilityTests()
        self.model_performance_tests = ModelPerformanceTests()
        self.robustness_tests = RobustnessTests()
        self.bias_fairness_tests = BiasFairnessTests()

    def run_test(
        self,
        test_config: Dict[str, Any],
        model_artifact: Any = None,
        datasets: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Run a single ML test based on configuration.

        Args:
            test_config: Test configuration dictionary
            model_artifact: Loaded ML model (if needed)
            datasets: Dictionary of datasets (if needed)

        Returns:
            Dict containing test results
        """

        category = test_config.get("category")
        test_name = test_config.get("name", "Unknown Test")

        logger.debug(
            "Running ML test",
            test_name=test_name,
            category=category
        )

        try:
            if category == "data_quality":
                return self.data_quality_tests.run_test(test_config, datasets or {})
            elif category == "statistical_stability":
                return self.statistical_stability_tests.run_test(test_config, datasets or {})
            elif category == "model_performance":
                return self.model_performance_tests.run_test(test_config, model_artifact, datasets or {})
            elif category == "robustness":
                return self.robustness_tests.run_test(test_config, model_artifact, datasets or {})
            elif category == "bias_fairness":
                return self.bias_fairness_tests.run_test(test_config, model_artifact, datasets or {})
            else:
                return {
                    "status": "failed",
                    "message": f"Unknown test category: {category}",
                    "details": {"error": f"Unsupported category: {category}"}
                }

        except Exception as e:
            logger.error(
                "ML test execution failed",
                test_name=test_name,
                category=category,
                error=str(e)
            )
            return {
                "status": "error",
                "message": f"Test execution failed: {str(e)}",
                "details": {"error": str(e), "traceback": str(e.__traceback__)}
            }

    def get_available_tests(self) -> Dict[str, List[str]]:
        """Get all available test types by category."""
        return {
            "data_quality": self.data_quality_tests.get_available_tests(),
            "statistical_stability": self.statistical_stability_tests.get_available_tests(),
            "model_performance": self.model_performance_tests.get_available_tests(),
            "robustness": self.robustness_tests.get_available_tests(),
            "bias_fairness": self.bias_fairness_tests.get_available_tests()
        }

    def validate_test_config(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a test configuration.

        Args:
            test_config: Test configuration to validate

        Returns:
            Dict with validation results
        """

        category = test_config.get("category")

        if not category:
            return {
                "valid": False,
                "error": "Test category is required"
            }

        try:
            if category == "data_quality":
                return self.data_quality_tests.validate_config(test_config)
            elif category == "statistical_stability":
                return self.statistical_stability_tests.validate_config(test_config)
            elif category == "model_performance":
                return self.model_performance_tests.validate_config(test_config)
            elif category == "robustness":
                return self.robustness_tests.validate_config(test_config)
            elif category == "bias_fairness":
                return self.bias_fairness_tests.validate_config(test_config)
            else:
                return {
                    "valid": False,
                    "error": f"Unknown test category: {category}"
                }

        except Exception as e:
            return {
                "valid": False,
                "error": f"Configuration validation failed: {str(e)}"
            }