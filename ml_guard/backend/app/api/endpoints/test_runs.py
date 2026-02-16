"""
Test run execution endpoints
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()

class TestRunRequest(BaseModel):
    project_id: str
    model_version: str
    test_suite_id: str
    environment: str = "development"

class TestRun(BaseModel):
    id: str
    project_id: str
    model_version: str
    test_suite_id: str
    environment: str
    status: str  # running, completed, failed
    start_time: str
    end_time: Optional[str] = None
    results: List[dict] = []
    summary: Optional[dict] = None

# Mock data for demonstration
MOCK_TEST_RUNS = [
    {
        "id": "run-1",
        "project_id": "ecommerce-ml",
        "model_version": "v2.1.3",
        "test_suite_id": "suite-1",
        "environment": "staging",
        "status": "completed",
        "start_time": "2024-01-15T10:00:00Z",
        "end_time": "2024-01-15T10:05:23Z",
        "results": [
            {
                "test_id": "test-1",
                "name": "Missing Values Check",
                "category": "data_quality",
                "status": "passed",
                "message": "Missing values rate: 0.023 (threshold: 0.05)"
            },
            {
                "test_id": "test-2",
                "name": "Model Accuracy Threshold",
                "category": "model_performance",
                "status": "failed",
                "message": "Accuracy 0.82 below threshold 0.85"
            }
        ],
        "summary": {
            "total_tests": 18,
            "passed": 16,
            "failed": 2,
            "execution_time_seconds": 283.5
        }
    }
]

@router.post("/", response_model=TestRun)
async def run_tests(request: TestRunRequest, background_tasks: BackgroundTasks):
    """Execute a test suite."""
    # In a real implementation, this would:
    # 1. Validate inputs
    # 2. Start test execution in background
    # 3. Return run ID immediately

    run_id = f"run-{len(MOCK_TEST_RUNS) + 1}"

    new_run = {
        "id": run_id,
        "project_id": request.project_id,
        "model_version": request.model_version,
        "test_suite_id": request.test_suite_id,
        "environment": request.environment,
        "status": "running",
        "start_time": datetime.utcnow().isoformat(),
        "results": [],
        "summary": None
    }

    MOCK_TEST_RUNS.append(new_run)

    # Simulate background execution
    background_tasks.add_task(simulate_test_execution, run_id)

    return new_run

@router.get("/", response_model=List[TestRun])
async def list_test_runs(project_id: Optional[str] = None, limit: int = 50):
    """List test runs, optionally filtered by project."""
    runs = MOCK_TEST_RUNS
    if project_id:
        runs = [run for run in runs if run["project_id"] == project_id]

    return runs[-limit:]  # Return most recent

@router.get("/{run_id}", response_model=TestRun)
async def get_test_run(run_id: str):
    """Get a specific test run."""
    for run in MOCK_TEST_RUNS:
        if run["id"] == run_id:
            return run
    raise HTTPException(status_code=404, detail="Test run not found")

async def simulate_test_execution(run_id: str):
    """Simulate test execution (for demonstration)."""
    # In a real implementation, this would execute actual tests
    import asyncio
    await asyncio.sleep(2)  # Simulate execution time

    # Update the mock run with results
    for run in MOCK_TEST_RUNS:
        if run["id"] == run_id:
            run["status"] = "completed"
            run["end_time"] = datetime.utcnow().isoformat()
            break