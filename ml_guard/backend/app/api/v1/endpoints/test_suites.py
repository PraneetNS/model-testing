
from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def list_test_suites():
    """List available test suites."""
    return [{"id": "drift_suite", "tests": ["psi", "kl_divergence"]}]
