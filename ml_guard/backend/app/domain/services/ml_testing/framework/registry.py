from typing import Dict, Type
from .base import MLTestCase
from .implementations import MissingValuesTest, AccuracyTest, PSIDriftTest, RegressionTest

TEST_REGISTRY: Dict[str, Type[MLTestCase]] = {
    "missing_values": MissingValuesTest,
    "accuracy_threshold": AccuracyTest,
    "psi_drift": PSIDriftTest,
    "regression_check": RegressionTest
}

def get_test_class(test_type: str) -> Type[MLTestCase]:
    return TEST_REGISTRY.get(test_type)
