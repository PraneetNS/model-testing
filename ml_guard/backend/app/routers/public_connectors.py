from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging
from .data_connectors import FetchRequest
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.core.auth import AuthContext, require_role
from ml_guard.plugins.data_connectors.factory import get_connector
from app.tasks.data_connectors import data_connector_fetch_task

logger = logging.getLogger(__name__)

kaggle_router = APIRouter()
openml_router = APIRouter()
roboflow_router = APIRouter()

@kaggle_router.post("/fetch")
async def fetch_kaggle_data(
    req: FetchRequest,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer"))
):
    try:
        connector = get_connector("kaggle")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    valid, errors = connector.validate_config(req.config)
    if not valid:
        raise HTTPException(status_code=400, detail={"errors": errors})

    # Trigger Celery task
    task = data_connector_fetch_task.delay(
        connector_type="kaggle",
        config=req.config,
        source_uri=req.source_uri
    )
    
    return {
        "task_id": task.id,
        "estimated_rows": None,
        "message": "Kaggle data ingestion task started in background",
        "connector": "kaggle",
        "source": req.source_uri
    }

@openml_router.get("/search")
async def search_openml(
    query: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("viewer"))
):
    import openml
    import time
    from filelock import FileLock
    import os
    import tempfile
    
    lock_path = "/tmp/openml_api.lock"
    if os.name == 'nt':
        lock_path = os.path.join(tempfile.gettempdir(), "openml_api.lock")

    try:
        with FileLock(lock_path, timeout=60):
            time.sleep(1) # Ensure 1-second delay
            datasets = openml.datasets.list_datasets(data_name=query, number_instances=">1")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    results = []
    for did, dinfo in list(datasets.items())[:limit]:
        results.append({
            "id": did,
            "name": dinfo.get("name"),
            "n_rows": dinfo.get("NumberOfInstances", 0),
            "n_cols": dinfo.get("NumberOfFeatures", 0),
            "n_classes": dinfo.get("NumberOfClasses", 0)
        })
    return results

@openml_router.post("/fetch")
async def fetch_openml_data(
    req: FetchRequest,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer"))
):
    try:
        connector = get_connector("openml")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    valid, errors = connector.validate_config(req.config)
    if not valid:
        raise HTTPException(status_code=400, detail={"errors": errors})

    # Trigger Celery task
    task = data_connector_fetch_task.delay(
        connector_type="openml",
        config=req.config,
        source_uri=req.source_uri
    )
    
    return {
        "task_id": task.id,
        "estimated_rows": None,
        "message": "OpenML data ingestion task started in background",
        "connector": "openml",
        "source": req.source_uri
    }

@roboflow_router.post("/fetch")
async def fetch_roboflow_data(
    req: FetchRequest,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer"))
):
    try:
        connector = get_connector("roboflow")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    valid, errors = connector.validate_config(req.config)
    if not valid:
        raise HTTPException(status_code=400, detail={"errors": errors})

    # Trigger Celery task
    task = data_connector_fetch_task.delay(
        connector_type="roboflow",
        config=req.config,
        source_uri=req.source_uri
    )
    
    return {
        "task_id": task.id,
        "estimated_rows": None,
        "message": "Roboflow data ingestion task started in background",
        "connector": "roboflow",
        "source": req.source_uri
    }
