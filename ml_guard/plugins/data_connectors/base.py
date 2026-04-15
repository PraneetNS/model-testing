"""
base.py — Abstract Base Class for ML Guard Data Connectors

Enables direct dataset ingestion from cloud storage (S3, GCS, Azure) 
and data warehouses (Snowflake, BigQuery).
"""
from __future__ import annotations

import os
import uuid
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

TEMP_DATASET_STORE = Path(os.getenv("MLGUARD_TEMP_DIR", "/tmp/mlguard_datasets"))

class DataConnector(ABC):
    """
    Abstract interface for all enterprise data connectors.
    """

    @abstractmethod
    def fetch(self, config: Dict[str, Any]) -> str:
        """
        Pull data from the source and return the local CSV path.
        
        Parameters
        ----------
        config : dict
            Connection and query/path configuration.
            Credentials should already be decrypted if retrieved from DB.
        
        Returns
        -------
        str
            The absolute path to the downloaded CSV file.
        """
        pass

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Check if the provided configuration contains all required fields.
        
        Returns
        -------
        bool
            True if valid, False otherwise.
        List[str]
            List of error messages if invalid.
        """
        pass

    def save_to_temp(self, df: pd.DataFrame) -> str:
        """
        Standard helper to save a DataFrame to a temporary CSV file.
        """
        # Ensure temp directory exists
        if not TEMP_DATASET_STORE.exists():
            try:
                TEMP_DATASET_STORE.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                # Fallback to current working directory if /tmp is not writable
                fallback = Path("./mlguard_datasets")
                fallback.mkdir(parents=True, exist_ok=True)
                logger.warning(f"Failed to create {TEMP_DATASET_STORE}, falling back to {fallback}. Error: {e}")
                dataset_path = fallback / f"{uuid.uuid4()}.csv"
                df.to_csv(dataset_path, index=False)
                return str(dataset_path.resolve())

        dataset_path = TEMP_DATASET_STORE / f"{uuid.uuid4()}.csv"
        df.to_csv(dataset_path, index=False)
        return str(dataset_path.resolve())

    def mask_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Helper to redact sensitive fields for logging.
        """
        sensitive_fields = {
            "aws_access_key_id", "aws_secret_access_key", 
            "service_account_json", "password", "account_key", "sas_token"
        }
        masked = config.copy()
        for field in masked:
            if field in sensitive_fields:
                masked[field] = "***"
        return masked
