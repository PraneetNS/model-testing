"""
factory.py — Connector Factory and Encryption Utility for Data Connectors
"""
import json
import logging
from typing import Dict, Any, Type, Optional
from cryptography.fernet import Fernet
from app.core.config import settings

from .base import DataConnector

logger = logging.getLogger(__name__)


def _optional_connector(module: str, class_name: str) -> Optional[Type[DataConnector]]:
    try:
        mod = __import__(f"ml_guard.plugins.data_connectors.{module}", fromlist=[class_name])
        return getattr(mod, class_name)
    except ImportError as exc:
        logger.warning("data_connector_unavailable: %s (%s)", module, exc)
        return None


# ── Encryption setup ──────────────────────────────────────────────────────────

def get_fernet() -> Fernet:
    """Initialize Fernet with the app's secret key."""
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

_CONNECTOR_SPECS = {
    "s3": ("s3", "S3Connector"),
    "gcs": ("gcs", "GCSConnector"),
    "azure": ("azure_blob", "AzureBlobConnector"),
    "snowflake": ("snowflake", "SnowflakeConnector"),
    "bigquery": ("bigquery", "BigQueryConnector"),
    "kaggle": ("kaggle", "KaggleConnector"),
    "openml": ("openml", "OpenMLConnector"),
    "roboflow": ("roboflow", "RoboflowConnector"),
}

CONNECTOR_MAP: Dict[str, Type[DataConnector]] = {}
for _key, (_module, _class_name) in _CONNECTOR_SPECS.items():
    _cls = _optional_connector(_module, _class_name)
    if _cls is not None:
        CONNECTOR_MAP[_key] = _cls

def get_connector(connector_type: str) -> DataConnector:
    """Get an instance of the requested connector."""
    cls = CONNECTOR_MAP.get(connector_type.lower())
    if not cls:
        raise ValueError(f"Unsupported connector type: {connector_type}")
    return cls()
