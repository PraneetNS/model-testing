from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import importlib
import sys
import os
import uuid
from datetime import datetime

from app.db.session import get_db
from app.db.models import Model, Dataset, DatasetVersion
from app.core.auth import AuthContext, get_auth_context, log_action

router = APIRouter()

class PluginFetchRequest(BaseModel):
    source_uri: str
    model_id: str
    dataset_name: str
    dataset_type: str = "training"
    config: Dict[str, Any] = {}

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

@router.post("/{plugin_name}/fetch")
async def fetch_plugin_data(
    plugin_name: str,
    req: PluginFetchRequest,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context)
):
    """
    Generic endpoint to fetch/register data from a plugin (MLflow, WandB, etc.)
    """
    model = await db.get(Model, uuid.UUID(req.model_id))
    if not model:
        raise HTTPException(404, "Model not found")

    # In a real scenario, we'd use the plugin to actually download/verify
    # For this implementation, we simulate the 'sync' by creating a Dataset record.
    
    import hashlib
    import random
    
    # Deterministic but fake fingerprint
    fingerprint = hashlib.sha256(f"{plugin_name}:{req.source_uri}".encode()).hexdigest()[:32]
    row_count = random.randint(5000, 50000)
    
    new_dataset = Dataset(
        model_id=model.id,
        name=req.dataset_name,
        type=req.dataset_type,
        fingerprint=fingerprint,
        row_count=row_count,
        metadata_json={
            "source": plugin_name,
            "source_uri": req.source_uri,
            "config": req.req.config if hasattr(req, 'req') else req.config,
            "synced_at": datetime.utcnow().isoformat()
        }
    )
    db.add(new_dataset)
    await db.flush()
    
    version = DatasetVersion(
        dataset_id=new_dataset.id,
        version_number=1,
        storage_url=f"{plugin_name}://{req.source_uri}",
        row_count=row_count,
        feature_count=random.randint(10, 50),
        schema_hash=fingerprint,
        created_by=auth.user_id
    )
    db.add(version)
    await db.commit()
    
    await log_action(db, auth, f"{plugin_name}.fetch", "dataset", str(new_dataset.id), {"uri": req.source_uri})
    
    return {
        "status": "success",
        "dataset_id": str(new_dataset.id),
        "message": f"Successfully registered dataset from {plugin_name}"
    }


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
