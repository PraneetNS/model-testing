"""
gcs.py — Google Cloud Storage Data Connector for ML Guard
"""
import logging
import os
import tempfile
import base64
import json
import pandas as pd
from typing import Dict, List, Any, Tuple
from google.cloud import storage
from google.oauth2 import service_account
from .base import DataConnector

logger = logging.getLogger(__name__)

class GCSConnector(DataConnector):
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        required = ["bucket", "blob_name", "service_account_json"]
        errors = []
        for field in required:
            if not config.get(field):
                errors.append(f"Missing required field: {field}")
        return len(errors) == 0, errors

    def fetch(self, config: Dict[str, Any]) -> str:
        masked = self.mask_config(config)
        logger.info(f"GCS fetch initiated: {masked}")

        bucket_name = config["bucket"]
        blob_name = config["blob_name"]
        sa_json_b64 = config["service_account_json"]
        
        # Decode and load credentials
        sa_info = json.loads(base64.b64decode(sa_json_b64).decode("utf-8"))
        creds = service_account.Credentials.from_service_account_info(sa_info)
        
        client = storage.Client(credentials=creds, project=sa_info.get("project_id"))
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        suffix = os.path.splitext(blob_name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            blob.download_to_filename(tmp.name)
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
