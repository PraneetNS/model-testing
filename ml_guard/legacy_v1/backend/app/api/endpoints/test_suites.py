"""
Test suite management endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel

router = APIRouter()

class TestSuite(BaseModel):
    id: str
    project_id: str
    name: str
    description: str = ""
    tests: List[dict]
    created_at: str
    updated_at: str

class TestCase(BaseModel):
    id: str
    suite_id: str
    name: str
    category: str
    config: dict
    severity: str = "medium"
    enabled: bool = True

# Mock data for demonstration
MOCK_TEST_SUITES = [
    {
        "id": "suite-1",
        "project_id": "ecommerce-ml",
        "name": "Production Readiness",
        "description": "Comprehensive validation for production deployment",
        "tests": [
            {
                "id": "test-1",
                "name": "Missing Values Check",
                "category": "data_quality",
                "severity": "high",
                "enabled": True
            },
            {
                "id": "test-2",
                "name": "Model Accuracy Threshold",
                "category": "model_performance",
                "severity": "critical",
                "enabled": True
            }
        ],
        "created_at": "2024-01-10T00:00:00Z",
        "updated_at": "2024-01-15T00:00:00Z"
    }
]

@router.get("/", response_model=List[TestSuite])
async def list_test_suites(project_id: Optional[str] = None):
    """List test suites, optionally filtered by project."""
    if project_id:
        return [suite for suite in MOCK_TEST_SUITES if suite["project_id"] == project_id]
    return MOCK_TEST_SUITES

@router.get("/{suite_id}", response_model=TestSuite)
async def get_test_suite(suite_id: str):
    """Get a specific test suite."""
    for suite in MOCK_TEST_SUITES:
        if suite["id"] == suite_id:
            return suite
    raise HTTPException(status_code=404, detail="Test suite not found")

@router.post("/", response_model=TestSuite)
async def create_test_suite(suite: TestSuite):
    """Create a new test suite."""
    # In a real implementation, this would save to database
    MOCK_TEST_SUITES.append(suite.dict())
    return suite