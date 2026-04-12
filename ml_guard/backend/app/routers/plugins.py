from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import importlib
import sys
import os

from app.db.session import get_db

router = APIRouter()

class MLflowSyncRequest(BaseModel):
    model_id: str
    run_id: str
    tracking_uri: Optional[str] = None
    experiment_name: Optional[str] = None
    metric_map: Dict[str, str] = {}

class WandbSyncRequest(BaseModel):
    model_id: str
    run_id: str
    api_key: Optional[str] = None
    entity: Optional[str] = None
    project: Optional[str] = None
    metric_map: Dict[str, str] = {}

@router.post("/mlflow/sync")
async def sync_mlflow(request: MLflowSyncRequest, db: AsyncSession = Depends(get_db)):
    try:
        from ml_guard.plugins.mlflow_sync import MLflowSyncPlugin
        plugin = MLflowSyncPlugin({
            "tracking_uri": request.tracking_uri,
            "experiment_name": request.experiment_name,
            "metric_map": request.metric_map,
            "run_id": request.run_id
        })
        result = await plugin.sync_to_model(request.model_id, request.run_id, db)
        return result
    except ImportError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.post("/wandb/sync")
async def sync_wandb(request: WandbSyncRequest, db: AsyncSession = Depends(get_db)):
    try:
        from ml_guard.plugins.wandb_sync import WandbSyncPlugin
        plugin = WandbSyncPlugin({
            "api_key": request.api_key,
            "entity": request.entity,
            "project": request.project,
            "metric_map": request.metric_map,
            "run_id": request.run_id
        })
        result = await plugin.sync_to_model(request.model_id, request.run_id, db)
        return result
    except ImportError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.get("/available")
async def list_available_plugins():
    plugins = []
    try:
        importlib.import_module("mlflow")
        plugins.append("mlflow")
    except ImportError:
        pass
        
    try:
        importlib.import_module("wandb")
        plugins.append("wandb")
    except ImportError:
        pass
        
    return {"available_plugins": plugins}
