
from fastapi import APIRouter
from typing import List

router = APIRouter()

@router.get("/")
async def list_models():
    """List all registered models."""
    return [{"id": "1", "name": "customer_churn_v1"}]

@router.post("/")
async def register_model():
    """Register a new model."""
    return {"status": "registered"}
