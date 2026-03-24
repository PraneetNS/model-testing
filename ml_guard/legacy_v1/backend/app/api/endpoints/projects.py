"""
Project management endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import List
from pydantic import BaseModel

router = APIRouter()

class Project(BaseModel):
    id: str
    name: str
    description: str = ""
    created_at: str
    updated_at: str

# Mock data for demonstration
MOCK_PROJECTS = [
    {
        "id": "ecommerce-ml",
        "name": "E-commerce ML Pipeline",
        "description": "Customer churn prediction and recommendation models",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-15T00:00:00Z"
    }
]

@router.get("/", response_model=List[Project])
async def list_projects():
    """List all projects."""
    return MOCK_PROJECTS

@router.get("/{project_id}", response_model=Project)
async def get_project(project_id: str):
    """Get a specific project."""
    for project in MOCK_PROJECTS:
        if project["id"] == project_id:
            return project
    raise HTTPException(status_code=404, detail="Project not found")

@router.post("/", response_model=Project)
async def create_project(project: Project):
    """Create a new project."""
    # In a real implementation, this would save to database
    MOCK_PROJECTS.append(project.dict())
    return project