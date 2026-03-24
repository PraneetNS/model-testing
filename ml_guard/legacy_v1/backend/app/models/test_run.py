"""
Data models for test runs and results
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class TestResult(BaseModel):
    """Individual test execution result."""

    test_id: str = Field(..., description="Unique test identifier")
    test_name: str = Field(..., description="Human-readable test name")
    category: str = Field(
        ...,
        description="Test category",
        enum=[
            "data_quality",
            "statistical_stability",
            "model_performance",
            "robustness",
            "bias_fairness"
        ]
    )
    status: str = Field(
        ...,
        description="Test execution status",
        enum=["passed", "failed", "warning", "error"]
    )
    severity: str = Field(
        ...,
        description="Test failure severity",
        enum=["low", "medium", "high", "critical"]
    )
    message: str = Field(..., description="Human-readable result message")
    execution_time_seconds: float = Field(..., description="Test execution time")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional test execution details"
    )
    threshold: Optional[float] = Field(
        default=None,
        description="Expected threshold value"
    )
    actual_value: Optional[float] = Field(
        default=None,
        description="Actual measured value"
    )
    evidence: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Evidence supporting the test result"
    )

class TestRun(BaseModel):
    """Complete test suite execution."""

    run_id: str = Field(..., description="Unique test run identifier")
    project_id: str = Field(..., description="FireFlink project identifier")
    model_version: str = Field(..., description="ML model version tested")
    test_suite: str = Field(..., description="Test suite name")
    environment: str = Field(..., description="Execution environment")
    status: str = Field(
        ...,
        description="Overall test run status",
        enum=["running", "completed", "failed", "cancelled"]
    )
    start_time: datetime = Field(..., description="Test execution start time")
    end_time: Optional[datetime] = Field(
        default=None,
        description="Test execution end time"
    )
    execution_time_seconds: Optional[float] = Field(
        default=None,
        description="Total execution time in seconds"
    )
    results: List[TestResult] = Field(
        default_factory=list,
        description="Individual test results"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if execution failed"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional execution metadata"
    )

    @property
    def passed_tests(self) -> int:
        """Number of passed tests."""
        return len([r for r in self.results if r.status == "passed"])

    @property
    def failed_tests(self) -> int:
        """Number of failed tests."""
        return len([r for r in self.results if r.status == "failed"])

    @property
    def warning_tests(self) -> int:
        """Number of warning tests."""
        return len([r for r in self.results if r.status == "warning"])

    @property
    def critical_failures(self) -> List[TestResult]:
        """Critical test failures that should block deployment."""
        return [r for r in self.results if r.status == "failed" and r.severity == "critical"]

    @property
    def deployment_blocked(self) -> bool:
        """Whether this test run should block deployment."""
        return len(self.critical_failures) > 0