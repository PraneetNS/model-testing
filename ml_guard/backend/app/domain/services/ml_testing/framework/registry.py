from typing import Dict, Type
from .base import MLTestCase
from .implementations import MissingValuesTest, AccuracyTest, PSIDriftTest, RegressionTest, SchemaValidationTest, DatasetProfilingTest, RobustnessTest, BiasDetectionTest, OverfittingGapTest

TEST_REGISTRY: Dict[str, Type[MLTestCase]] = {
    "missing_values": MissingValuesTest,
    "accuracy_threshold": AccuracyTest,
    "psi_drift": PSIDriftTest,
    "regression_check": RegressionTest,
    "schema_validation": SchemaValidationTest,
    "dataset_profiling": DatasetProfilingTest,
    "input_perturbation": RobustnessTest,
    "disparate_impact": BiasDetectionTest,
    "overfitting_gap": OverfittingGapTest
}

def get_test_class(test_type: str) -> Type[MLTestCase]:
    return TEST_REGISTRY.get(test_type)
