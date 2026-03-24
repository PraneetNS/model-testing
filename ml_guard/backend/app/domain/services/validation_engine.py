
from typing import Dict, Any, List
from .ml_testing.engine import MLTestEngine
from app.domain.models.test_suite import TestResult
import pandas as pd
import numpy as np
from datetime import datetime

class ValidationEngine:
    """
    Executes individual test types using the restored ML Testing Engine.
    """
    def __init__(self):
        self.engine = MLTestEngine()

    async def run_test(self, test_config: Dict, model: Any = None, datasets: Dict[str, Any] = None) -> TestResult:
        """
        Executes a test and converts the result to the domain model.
        """
        # Call the real engine logic ported from legacy
        raw_result = self.engine.run_test(test_config, model, datasets)
        
        return TestResult(
            test_id=raw_result.get("test_id", f"test-{datetime.now().timestamp()}"),
            test_name=test_config.get("name", "Unknown Test"),
            category=test_config.get("category", "unknown"),
            status=raw_result.get("status", "failed"),
            severity=test_config.get("severity", "medium"),
            message=raw_result.get("message", "Test executed"),
            execution_time_seconds=raw_result.get("execution_time_seconds", 0.0),
            actual_value=raw_result.get("actual_value"),
            threshold=raw_result.get("threshold"),
            details=raw_result.get("details", {})
        )
