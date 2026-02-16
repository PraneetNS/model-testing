"""
Pydantic schemas for Quality Gate API
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class QualityGateStatus(str, Enum):
    """Quality gate evaluation status."""
    PASS = "PASS"
    FAIL = "FAIL"

class QualityGateRequest(BaseModel):
    """Request model for quality gate evaluation."""

    project_id: str = Field(..., description="FireFlink project identifier")
    model_version: str = Field(..., description="ML model version to evaluate")
    test_suite: str = Field(
        default="production-readiness",
        description="Test suite to execute"
    )
    environment: Optional[str] = Field(
        default="production",
        description="Target deployment environment"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata for the evaluation"
    )

    class Config:
        schema_extra = {
            "example": {
                "project_id": "ecommerce-ml-pipeline",
                "model_version": "v2.1.3",
                "test_suite": "production-readiness",
                "environment": "staging",
                "metadata": {
                    "triggered_by": "github-actions",
                    "commit_sha": "abc123",
                    "branch": "main"
                }
            }
        }

class TestFailure(BaseModel):
    """Details of a failed test."""

    test_name: str = Field(..., description="Name of the failed test")
    category: str = Field(..., description="Test category (data_quality, performance, etc.)")
    severity: str = Field(
        ...,
        description="Failure severity",
        enum=["low", "medium", "high", "critical"]
    )
    message: str = Field(..., description="Human-readable failure message")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional failure details and evidence"
    )

    class Config:
        schema_extra = {
            "example": {
                "test_name": "Model Accuracy Threshold",
                "category": "model_performance",
                "severity": "critical",
                "message": "Accuracy 0.82 below threshold 0.85",
                "details": {
                    "expected": 0.85,
                    "actual": 0.82,
                    "threshold_type": "minimum"
                }
            }
        }

class QualityGateResponse(BaseModel):
    """Response model for quality gate evaluation."""

    status: QualityGateStatus = Field(..., description="Overall quality gate status")
    deployment_allowed: bool = Field(..., description="Whether deployment is allowed")
    run_id: str = Field(..., description="Unique identifier for this evaluation run")
    project_id: str = Field(..., description="Project identifier")
    model_version: str = Field(..., description="Model version evaluated")
    test_suite: str = Field(..., description="Test suite executed")
    environment: str = Field(..., description="Target environment")
    summary: Dict[str, Any] = Field(..., description="Test execution summary")
    failures: Optional[List[TestFailure]] = Field(
        default=None,
        description="List of test failures (only included when status is FAIL)"
    )
    recommendations: Optional[List[str]] = Field(
        default=None,
        description="Actionable recommendations for fixing failures"
    )
    timestamp: datetime = Field(..., description="Evaluation timestamp")
    execution_details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional execution details"
    )

    class Config:
        schema_extra = {
            "example": {
                "status": "FAIL",
                "deployment_allowed": False,
                "run_id": "ml-run-20240115-001",
                "project_id": "ecommerce-ml-pipeline",
                "model_version": "v2.1.3",
                "test_suite": "production-readiness",
                "environment": "staging",
                "summary": {
                    "total_tests": 18,
                    "passed": 15,
                    "failed": 3,
                    "warnings": 0,
                    "execution_time_seconds": 45.2
                },
                "failures": [
                    {
                        "test_name": "Model Accuracy Threshold",
                        "category": "model_performance",
                        "severity": "critical",
                        "message": "Accuracy 0.82 below threshold 0.85",
                        "details": {
                            "expected": 0.85,
                            "actual": 0.82
                        }
                    }
                ],
                "recommendations": [
                    "Model retraining required - performance below acceptable thresholds",
                    "Review training data quality and distribution"
                ],
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }