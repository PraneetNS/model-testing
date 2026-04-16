"""
factory.py — Connector Factory and Encryption Utility for Data Connectors
"""
import json
import logging
from typing import Dict, Any, Type
from cryptography.fernet import Fernet
from app.core.config import settings

from .base import DataConnector
from .s3 import S3Connector
from .gcs import GCSConnector
from .azure_blob import AzureBlobConnector
from .snowflake import SnowflakeConnector
from .bigquery import BigQueryConnector
from .kaggle import KaggleConnector
from .openml import OpenMLConnector
from .roboflow import RoboflowConnector

logger = logging.getLogger(__name__)

# ── Encryption setup ──────────────────────────────────────────────────────────

def get_fernet() -> Fernet:
    """Initialize Fernet with the app's secret key."""
    # Ensure key is 32 base64-encoded bytes
    key = settings.SECRET_KEY
    if len(key) < 32:
        key = key.ljust(32, "0")
    import base64
    derived_key = base64.urlsafe_b64encode(key[:32].encode())
    return Fernet(derived_key)

def encrypt_config(config: Dict[str, Any]) -> str:
    """Encrypt a config dictionary into a string."""
    f = get_fernet()
    data = json.dumps(config).encode()
    return f.encrypt(data).decode()

def decrypt_config(token: str) -> Dict[str, Any]:
    """Decrypt a token back into a config dictionary."""
    f = get_fernet()
    data = f.decrypt(token.encode())
    return json.loads(data.decode())

# ── Connector Registry ────────────────────────────────────────────────────────

CONNECTOR_MAP: Dict[str, Type[DataConnector]] = {
    "s3": S3Connector,
    "gcs": GCSConnector,
    "azure": AzureBlobConnector,
    "snowflake": SnowflakeConnector,
    "bigquery": BigQueryConnector,
    "kaggle": KaggleConnector,
    "openml": OpenMLConnector,
    "roboflow": RoboflowConnector,
}

def get_connector(connector_type: str) -> DataConnector:
    """Get an instance of the requested connector."""
    cls = CONNECTOR_MAP.get(connector_type.lower())
    if not cls:
        raise ValueError(f"Unsupported connector type: {connector_type}")
    return cls()
