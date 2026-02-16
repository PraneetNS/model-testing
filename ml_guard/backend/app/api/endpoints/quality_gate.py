"""
Quality Gate API - FireFlink-style deployment blocking/allowing
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import structlog

from ...core.config import settings
from ...services.test_orchestrator import TestOrchestrator
from ...services.model_registry import ModelRegistry
from ...schemas.quality_gate import (
    QualityGateRequest,
    QualityGateResponse,
    TestFailure,
    QualityGateStatus
)

logger = structlog.get_logger(__name__)

router = APIRouter()
test_orchestrator = TestOrchestrator()
model_registry = ModelRegistry()

@router.post("/", response_model=QualityGateResponse)
async def quality_gate_check(
    request: QualityGateRequest,
    background_tasks: BackgroundTasks
) -> QualityGateResponse:
    """
    FireFlink-style quality gate for ML model deployment.

    This endpoint evaluates ML model quality against predefined test suites
    and returns a PASS/FAIL decision for deployment control.

    Returns PASS if all tests pass, FAIL if critical tests fail.
    """

    try:
        logger.info(
            "Quality gate check requested",
            project_id=request.project_id,
            model_version=request.model_version,
            test_suite=request.test_suite
        )

        # Validate project and model exist
        if not await model_registry.project_exists(request.project_id):
            raise HTTPException(
                status_code=404,
                detail=f"Project '{request.project_id}' not found"
            )

        if not await model_registry.model_version_exists(
            request.project_id,
            request.model_version
        ):
            raise HTTPException(
                status_code=404,
                detail=f"Model version '{request.model_version}' not found in project '{request.project_id}'"
        )

        # Execute test suite
        test_run = await test_orchestrator.run_test_suite(
            project_id=request.project_id,
            model_version=request.model_version,
            test_suite_name=request.test_suite,
            environment=request.environment
        )

        # Determine quality gate status
        status = QualityGateStatus.PASS
        deployment_allowed = True
        critical_failures = []

        for result in test_run.results:
            if result.status == "failed" and result.severity == "critical":
                status = QualityGateStatus.FAIL
                deployment_allowed = False
                critical_failures.append(TestFailure(
                    test_name=result.test_name,
                    category=result.category,
                    severity=result.severity,
                    message=result.message,
                    details=result.details
                ))

        # Log quality gate decision
        logger.info(
            "Quality gate decision",
            status=status.value,
            deployment_allowed=deployment_allowed,
            total_tests=len(test_run.results),
            failed_tests=len([r for r in test_run.results if r.status == "failed"]),
            critical_failures=len(critical_failures)
        )

        # Prepare response
        response = QualityGateResponse(
            status=status,
            deployment_allowed=deployment_allowed,
            run_id=test_run.run_id,
            project_id=request.project_id,
            model_version=request.model_version,
            test_suite=request.test_suite,
            environment=request.environment,
            summary={
                "total_tests": len(test_run.results),
                "passed": len([r for r in test_run.results if r.status == "passed"]),
                "failed": len([r for r in test_run.results if r.status == "failed"]),
                "warnings": len([r for r in test_run.results if r.status == "warning"]),
                "execution_time_seconds": test_run.execution_time_seconds
            },
            failures=critical_failures if critical_failures else None,
            recommendations=_generate_recommendations(test_run) if critical_failures else None,
            timestamp=datetime.utcnow()
        )

        # Background task for additional processing (reporting, notifications, etc.)
        if not deployment_allowed:
            background_tasks.add_task(
                _handle_failed_quality_gate,
                request.project_id,
                request.model_version,
                critical_failures
            )

        return response

    except Exception as e:
        logger.error(
            "Quality gate check failed",
            error=str(e),
            project_id=request.project_id,
            model_version=request.model_version
        )
        raise HTTPException(status_code=500, detail=f"Quality gate check failed: {str(e)}")

@router.get("/status/{run_id}")
async def get_quality_gate_status(run_id: str):
    """Get the status of a quality gate run."""
    # In a real implementation, this would fetch from database/storage
    # For now, return a mock response
    return {
        "run_id": run_id,
        "status": "completed",
        "message": "Quality gate evaluation completed"
    }

def _generate_recommendations(test_run) -> List[str]:
    """Generate actionable recommendations based on test failures."""
    recommendations = []

    failure_categories = {}
    for result in test_run.results:
        if result.status == "failed":
            failure_categories.setdefault(result.category, []).append(result)

    # Data quality recommendations
    if "data_quality" in failure_categories:
        recommendations.append("Review and clean training data - address missing values and outliers")
        recommendations.append("Validate data schema consistency between train/validation/test sets")

    # Statistical stability recommendations
    if "statistical_stability" in failure_categories:
        recommendations.append("Investigate data drift - collect more recent training data")
        recommendations.append("Monitor feature distributions in production")

    # Model performance recommendations
    if "model_performance" in failure_categories:
        recommendations.append("Model retraining required - performance below acceptable thresholds")
        recommendations.append("Consider hyperparameter tuning or architecture changes")

    # Robustness recommendations
    if "robustness" in failure_categories:
        recommendations.append("Improve model robustness - test with adversarial inputs")
        recommendations.append("Implement input validation and sanitization")

    # Bias and fairness recommendations
    if "bias_fairness" in failure_categories:
        recommendations.append("Address bias in training data or model predictions")
        recommendations.append("Implement fairness-aware training techniques")
        recommendations.append("Conduct thorough bias audit with domain experts")

    if not recommendations:
        recommendations.append("Investigate test failures and address root causes before deployment")

    return recommendations[:5]  # Limit to top 5 recommendations

async def _handle_failed_quality_gate(
    project_id: str,
    model_version: str,
    failures: List[TestFailure]
):
    """Handle failed quality gate - send notifications, update dashboards, etc."""
    logger.warning(
        "Quality gate failed - deployment blocked",
        project_id=project_id,
        model_version=model_version,
        failure_count=len(failures)
    )

    # In a real implementation, this would:
    # - Send email/Slack notifications to team
    # - Update monitoring dashboards
    # - Create incident tickets
    # - Store failure analysis for reporting

    pass