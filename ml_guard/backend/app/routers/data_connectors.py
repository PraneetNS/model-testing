"""
data_connectors.py — API Router for Enterprise Data Connectors
"""
import logging
from typing import Dict, Any, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.core.auth import AuthContext, require_role
from app.db.models import CredentialConfig

from ml_guard.plugins.data_connectors.factory import (
    get_connector, encrypt_config, decrypt_config
)
from app.tasks.data_connectors import data_connector_fetch_task

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Schemas ───────────────────────────────────────────────────────────────────

class FetchRequest(BaseModel):
    connector_type: Optional[str] = None # Optional for specific connector routers
    config: Dict[str, Any]
    source_uri: str
    save_config_label: Optional[str] = None
    model_id: Optional[str] = None
    dataset_type: Optional[str] = "training"
    dataset_name: Optional[str] = None

class ConfigListItem(BaseModel):
    id: UUID
    connector_type: str
    label: str

# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/fetch", tags=["data-connectors"])
async def fetch_data(
    req: FetchRequest,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer"))
):
    """
    Initiate a background data pull from a cloud source or warehouse.
    """
    try:
        connector = get_connector(req.connector_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 1. Validate config
    valid, errors = connector.validate_config(req.config)
    if not valid:
        raise HTTPException(status_code=400, detail={"errors": errors})

    # 2. Optionally encrypt and save config for future use
    if req.save_config_label:
        encrypted = encrypt_config(req.config)
        new_conf = CredentialConfig(
            connector_type=req.connector_type,
            label=req.save_config_label,
            encrypted_config=encrypted,
            created_by_key_id=auth.key_id if hasattr(auth, "key_id") else None
        )
        db.add(new_conf)
        await db.flush()

    # 3. Trigger Celery task
    # Mask config for logging
    masked = connector.mask_config(req.config)
    logger.info(f"User {auth.user_id} triggered {req.connector_type} fetch from {req.source_uri}. Config: {masked}")
    
    from app.core.celery_app import encrypt_task_payload
    payload = {
        "connector_type": req.connector_type,
        "config": req.config,
        "source_uri": req.source_uri
    }
    encrypted_payload = encrypt_task_payload(payload, ["config"])
    
    task = data_connector_fetch_task.delay(**encrypted_payload)
    
    return {
        "task_id": task.id,
        "estimated_rows": None,
        "message": "Data ingestion task started in background",
        "connector": req.connector_type,
        "source": req.source_uri
    }

@router.get("/configs", response_model=List[ConfigListItem], tags=["data-connectors"])
async def list_saved_configs(
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("viewer"))
):
    """
    List saved connection configurations (labels and types only).
    """
    result = await db.execute(select(CredentialConfig))
    configs = result.scalars().all()
    
    return [
        ConfigListItem(
            id=c.id,
            connector_type=c.connector_type,
            label=c.label
        ) for c in configs
    ]

@router.get("/configs/{config_id}", tags=["data-connectors"])
async def get_decrypted_config(
    config_id: UUID,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(require_role("ml_engineer"))
):
    """
    Retrieve one config, decrypted. Restricted to ML Engineers.
    """
    result = await db.execute(select(CredentialConfig).filter(CredentialConfig.id == config_id))
    conf = result.scalar_one_or_none()
    
    if not conf:
        raise HTTPException(status_code=404, detail="Config not found")
        
    decrypted = decrypt_config(conf.encrypted_config)
    return {
        "id": conf.id,
        "connector_type": conf.connector_type,
        "label": conf.label,
        "config": decrypted
    }
