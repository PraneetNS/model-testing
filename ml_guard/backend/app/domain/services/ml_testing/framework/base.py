from enum import Enum
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
import time

class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class TestStatus(str, Enum):
    PASS = "passed"
    FAIL = "failed"
    WARN = "warning"
    ERROR = "error"

class MLTestCaseResult(BaseModel):
    name: str
    description: str
    severity: Severity
    status: TestStatus
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    explanation: str
    remediation: str
    execution_time: float
    details: Dict[str, Any] = {}

class MLTestCase:
    """Base class for all ML Framework test cases."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get("name", self.__class__.__name__)
        self.description = config.get("description", "")
        self.severity = Severity(config.get("severity", "medium"))
        self.target_column = config.get("config", {}).get("target_column", "target")

    async def run(self, model: Any, datasets: Dict[str, Any], baseline_model: Any = None, baseline_datasets: Dict[str, Any] = None) -> MLTestCaseResult:
        start_time = time.time()
        try:
            return await self.execute(model, datasets, baseline_model, baseline_datasets, start_time)
        except Exception as e:
            return MLTestCaseResult(
                name=self.name,
                description=self.description,
                severity=self.severity,
                status=TestStatus.ERROR,
                explanation=f"Test execution failed: {str(e)}",
                remediation="Ensure artifacts and datasets are valid.",
                execution_time=time.time() - start_time
            )

    async def execute(self, model: Any, datasets: Dict[str, Any], baseline_model: Any, baseline_datasets: Dict[str, Any], start_time: float) -> MLTestCaseResult:
        raise NotImplementedError("Subclasses must implement execute")
