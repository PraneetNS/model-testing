"""
data_connectors.py — Celery tasks for background data ingestion via connectors
"""
import os
import hashlib
import logging
import pandas as pd
from typing import Dict, Any

from app.core.celery_app import celery_app
from app.db.session import SessionLocal
from app.db.models import DatasetRegistry
from ml_guard.plugins.data_connectors.factory import get_connector

logger = logging.getLogger(__name__)

def _compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

@celery_app.task(name="tasks.data_connector_fetch", bind=True)
def data_connector_fetch_task(self, connector_type: str, config: Dict[str, Any], source_uri: str):
    """
    Background task to pull data, compute stats, and register in DB.
    """
    from app.core.celery_app import decrypt_task_payload
    decrypted = decrypt_task_payload({"config": config}, ["config"])
    config = decrypted["config"]
    
    logger.info(f"Starting background fetch for {connector_type} from {source_uri}")
    
    try:
        connector = get_connector(connector_type)
        local_path = connector.fetch(config)
        
        # Load briefly to get stats
        df = pd.read_csv(local_path)
        row_count = len(df)
        column_names = list(df.columns)
        sha256 = _compute_sha256(local_path)
        
        # Cleanup temp file? Actually the task description says "registered in a dataset_registry table".
        # Usually we'd move it to permanent storage, but the prompt says 
        # "base.py ... save_to_temp ... saves to /tmp/...".
        # I'll keep it there for now as indicated.
        
        with SessionLocal() as db:
            reg = DatasetRegistry(
                source_type=connector_type,
                source_uri=source_uri,
                row_count=row_count,
                column_names=column_names,
                sha256=sha256
            )
            db.add(reg)
            db.commit()
            db.refresh(reg)
            
            logger.info(f"Successfully registered dataset {reg.id} from {connector_type}")
            return {
                "registry_id": str(reg.id),
                "row_count": row_count,
                "sha256": sha256,
                "local_path": local_path
            }
            
    except Exception as e:
        logger.exception(f"Data connector fetch task failed for {connector_type}")
        raise
