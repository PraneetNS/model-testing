"""
azure_blob.py — Azure Blob Storage Data Connector for ML Guard
"""
import logging
import os
import tempfile
import pandas as pd
from typing import Dict, List, Any, Tuple
from azure.storage.blob import BlobServiceClient
from .base import DataConnector

logger = logging.getLogger(__name__)

class AzureBlobConnector(DataConnector):
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        required = ["account_name", "container", "blob_name"]
        errors = []
        for field in required:
            if not config.get(field):
                errors.append(f"Missing required field: {field}")
        
        if not (config.get("sas_token") or config.get("account_key")):
            errors.append("Must provide either sas_token or account_key")
            
        return len(errors) == 0, errors

    def fetch(self, config: Dict[str, Any]) -> str:
        masked = self.mask_config(config)
        logger.info(f"Azure Blob fetch initiated: {masked}")

        account_name = config["account_name"]
        container_name = config["container"]
        blob_name = config["blob_name"]
        
        if config.get("account_key"):
            account_url = f"https://{account_name}.blob.core.windows.net"
            client = BlobServiceClient(account_url, credential=config["account_key"])
        else:
            account_url = f"https://{account_name}.blob.core.windows.net?{config['sas_token']}"
            client = BlobServiceClient(account_url)

        blob_client = client.get_blob_client(container=container_name, blob=blob_name)
        
        suffix = os.path.splitext(blob_name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            with open(tmp.name, "wb") as f:
                data = blob_client.download_blob()
                data.readinto(f)
            local_raw_path = tmp.name

        try:
            if blob_name.endswith(".parquet"):
                df = pd.read_parquet(local_raw_path)
            else:
                df = pd.read_csv(local_raw_path)
            
            return self.save_to_temp(df)
        finally:
            if os.path.exists(local_raw_path):
                os.unlink(local_raw_path)
