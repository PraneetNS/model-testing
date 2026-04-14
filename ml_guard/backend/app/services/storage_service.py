"""
MinIO Object Storage Service.
S3-compatible API for uploading/downloading ML artifacts and datasets.

Environment variables:
    MINIO_ENDPOINT   - MinIO server endpoint (default: http://localhost:9000)
    MINIO_ACCESS_KEY - MinIO access key
    MINIO_SECRET_KEY - MinIO secret key
    MINIO_BUCKET     - Bucket name (default: mlguard-artifacts)
"""
import os
import uuid
import logging
import tempfile
from typing import Optional, BinaryIO, Union
from datetime import datetime, timezone

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError, EndpointConnectionError

from app.core.config import settings

logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────
MODELS_PREFIX = "models/"
DATASETS_PREFIX = "datasets/"
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB chunks for multipart upload
MAX_RETRIES = 3


def _get_s3_client():
    """Build a boto3 S3 client configured for MinIO."""
    return boto3.client(
        "s3",
        endpoint_url=settings.MINIO_ENDPOINT,
        aws_access_key_id=settings.MINIO_ACCESS_KEY,
        aws_secret_access_key=settings.MINIO_SECRET_KEY,
        region_name=settings.MINIO_REGION,
        config=BotoConfig(
            retries={"max_attempts": 0, "mode": "adaptive"}, # No retries for connectivity check
            s3={"addressing_style": "path"},
            connect_timeout=1,
            read_timeout=1,
        ),
    )


_STORAGE_MODE = None # 'minio' or 'local'
LOCAL_STORAGE_DIR = os.path.join(os.getcwd(), ".minio_mock")

def _get_storage_mode():
    global _STORAGE_MODE
    if _STORAGE_MODE is not None:
        return _STORAGE_MODE
    
    try:
        client = _get_s3_client()
        client.head_bucket(Bucket=settings.MINIO_BUCKET)
        _STORAGE_MODE = "minio"
        logger.info("Storage mode: MinIO (Connected)")
    except Exception:
        _STORAGE_MODE = "local"
        if not os.path.exists(LOCAL_STORAGE_DIR):
            os.makedirs(LOCAL_STORAGE_DIR, exist_ok=True)
        logger.warning(f"Storage mode: Local Fallback (MinIO unreachable). Data saved to {LOCAL_STORAGE_DIR}")
    
    return _STORAGE_MODE


def _ensure_bucket_exists(client):
    """Create the bucket if it doesn't exist (idempotent)."""
    try:
        client.head_bucket(Bucket=settings.MINIO_BUCKET)
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code in ("404", "NoSuchBucket"):
            client.create_bucket(Bucket=settings.MINIO_BUCKET)
            logger.info("Created MinIO bucket: %s", settings.MINIO_BUCKET)
        else:
            raise


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════


def upload_model(
    file_input: Union[bytes, BinaryIO],
    original_filename: str,
    model_id: Optional[str] = None,
    content_type: str = "application/octet-stream",
) -> dict:
    """Upload a model artifact. Supports bytes or streaming file-like object."""
    uid = model_id or str(uuid.uuid4())
    object_key = f"{MODELS_PREFIX}{uid}/{original_filename}"

    if _get_storage_mode() == "local":
        path = os.path.join(LOCAL_STORAGE_DIR, object_key.replace("/", os.sep))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = file_input if isinstance(file_input, bytes) else file_input.read()
        with open(path, "wb") as f:
            f.write(data)
        size = len(data)
        provider = "local"
        url = f"local://{object_key}"
    else:
        if isinstance(file_input, bytes):
            _upload_bytes(_get_s3_client(), object_key, file_input, content_type)
            size = len(file_input)
        else:
            res = upload_file_streaming(file_input, object_key, content_type)
            size = res.get("size", 0)
        provider = "minio"
        url = f"{settings.MINIO_ENDPOINT}/{settings.MINIO_BUCKET}/{object_key}"

    return {
        "object_key": object_key,
        "url": url,
        "size": size,
        "storage_provider": provider,
    }


def upload_dataset(
    file_input: Union[bytes, BinaryIO],
    original_filename: str,
    dataset_type: str = "training",
    scan_id: Optional[str] = None,
    content_type: str = "text/csv",
) -> dict:
    """Upload a dataset. Supports bytes or streaming file-like object."""
    uid = scan_id or str(uuid.uuid4())
    object_key = f"{DATASETS_PREFIX}{dataset_type}/{uid}/{original_filename}"

    if _get_storage_mode() == "local":
        path = os.path.join(LOCAL_STORAGE_DIR, object_key.replace("/", os.sep))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = file_input if isinstance(file_input, bytes) else file_input.read()
        with open(path, "wb") as f:
            f.write(data)
        size = len(data)
        provider = "local"
        url = f"local://{object_key}"
    else:
        if isinstance(file_input, bytes):
            _upload_bytes(_get_s3_client(), object_key, file_input, content_type)
            size = len(file_input)
        else:
            res = upload_file_streaming(file_input, object_key, content_type)
            size = res.get("size", 0)
        provider = "minio"
        url = f"{settings.MINIO_ENDPOINT}/{settings.MINIO_BUCKET}/{object_key}"

    return {
        "object_key": object_key,
        "url": url,
        "size": size,
        "storage_provider": provider,
    }


def download_artifact(object_key: str) -> bytes:
    """Download an artifact from MinIO. Returns raw bytes."""
    if _get_storage_mode() == "local":
        path = os.path.join(LOCAL_STORAGE_DIR, object_key.replace("/", os.sep))
        if os.path.exists(path):
            with open(path, "rb") as f:
                return f.read()
        raise FileNotFoundError(f"Local artifact not found: {object_key}")
        
    client = _get_s3_client()
    try:
        resp = client.get_object(
            Bucket=settings.MINIO_BUCKET,
            Key=object_key,
        )
        return resp["Body"].read()
    except ClientError as e:
        logger.error("Failed to download from MinIO %s: %s", object_key, e)
        raise


def download_from_url(url: str) -> bytes:
    """
    Download data from a minio:// or http URL.
    minio://bucket_name/object_key
    """
    if not url:
        return b""
        
    if url.startswith("minio://"):
        path = url.replace("minio://", "")
        parts = path.split("/", 1)
        if len(parts) < 2:
            # Fallback: maybe it's just the key and we use default bucket
            bucket = settings.MINIO_BUCKET
            key = path
        else:
            bucket, key = parts
            
        client = _get_s3_client()
        try:
            resp = client.get_object(Bucket=bucket, Key=key)
            return resp["Body"].read()
        except ClientError as e:
            logger.error(f"MinIO download failed for {url}: {e}")
            raise
    elif url.startswith(("http://", "https://")):
        import requests
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r.content
        except Exception as e:
            logger.error(f"HTTP download failed for {url}: {e}")
            raise
    else:
        # Assume it's a local path for dev convenience
        if os.path.exists(url):
            with open(url, "rb") as f:
                return f.read()
        raise ValueError(f"Unsupported URL or path: {url}")


def delete_artifact(object_key: str) -> bool:
    """Delete an artifact from MinIO."""
    if _get_storage_mode() == "local":
        path = os.path.join(LOCAL_STORAGE_DIR, object_key.replace("/", os.sep))
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    client = _get_s3_client()
    try:
        client.delete_object(
            Bucket=settings.MINIO_BUCKET,
            Key=object_key,
        )
        logger.info("Deleted MinIO object: %s", object_key)
        return True
    except ClientError as e:
        logger.error("Failed to delete MinIO object %s: %s", object_key, e)
        return False


def generate_signed_url(object_key: str, expires_in: int = 3600) -> str:
    """Generate a pre-signed URL for MinIO."""
    client = _get_s3_client()
    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.MINIO_BUCKET, "Key": object_key},
        ExpiresIn=expires_in,
    )
    return url


def upload_file_streaming(
    file_obj: BinaryIO,
    object_key: str,
    content_type: str = "application/octet-stream",
) -> dict:
    """Stream-upload files using boto3's high-level API."""
    client = _get_s3_client()
    _ensure_bucket_exists(client)

    try:
        # Ensure we are at the beginning of the file stream
        if hasattr(file_obj, "seek"):
            file_obj.seek(0)
            
        client.upload_fileobj(
            file_obj,
            Bucket=settings.MINIO_BUCKET,
            Key=object_key,
            ExtraArgs={'ContentType': content_type}
        )
        
        # Determine size if possible
        size = 0
        if hasattr(file_obj, "tell"):
            size = file_obj.tell()
            
        return {
            "object_key": object_key,
            "url": f"{settings.MINIO_ENDPOINT}/{settings.MINIO_BUCKET}/{object_key}",
            "size": size,
            "storage_provider": "minio",
        }
    except Exception as e:
        logger.error(f"Streaming upload failed: {e}")
        raise


def check_storage_health() -> dict:
    """Health check for MinIO connectivity."""
    mode = _get_storage_mode()
    if mode == "local":
        return {
            "status": "connected",
            "provider": "local_fallback",
            "detail": f"MinIO unreachable. Files are being saved to {LOCAL_STORAGE_DIR}",
        }
    try:
        client = _get_s3_client()
        _ensure_bucket_exists(client)
        client.head_bucket(Bucket=settings.MINIO_BUCKET)
        return {
            "status": "connected",
            "provider": "minio",
            "bucket": settings.MINIO_BUCKET,
        }
    except Exception as e:
        return {"status": "error", "provider": "minio", "detail": str(e)}


def _upload_bytes(client, object_key: str, data: bytes, content_type: str):
    """Internal helper to upload bytes."""
    if len(data) >= CHUNK_SIZE:
        import io
        upload_file_streaming(io.BytesIO(data), object_key, content_type)
    else:
        client.put_object(
            Bucket=settings.MINIO_BUCKET,
            Key=object_key,
            Body=data,
            ContentType=content_type,
        )
