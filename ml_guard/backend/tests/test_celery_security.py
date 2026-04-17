import pytest
import json
import base64
import functools
from celery.exceptions import NotRegistered
from app.core.celery_app import celery_app, ALLOWED_TASKS, encrypt_task_payload, decrypt_task_payload

def test_pickle_rejected():
    # If we try to send a pickle serialized task, celery should reject it
    # We can inspect the app configuration
    assert celery_app.conf.task_serializer == "json"
    assert "json" in celery_app.conf.accept_content
    assert "pickle" not in celery_app.conf.accept_content

def test_unallowed_task_prerun_signal():
    # Test that a task not in ALLOWED_TASKS raises an exception in prerun
    from celery.signals import task_prerun
    
    class MockTask:
        name = "evil_injected_task"
        
    class MockException(Exception):
        pass

    import structlog
    from unittest.mock import patch
    
    with patch("structlog.get_logger") as mock_logger:
        # Trigger the signal directly since we can't easily run the worker for an unregistered task
        responses = task_prerun.send(sender=MockTask, task_id="123", task=MockTask, args=(), kwargs={})
        
        exception_found = False
        for receiver, response in responses:
            if isinstance(response, Exception) and "is not on the allowlist" in str(response):
                exception_found = True
        
        assert exception_found, "The security exception was not raised by the signal receiver"       
        mock_logger.return_value.error.assert_called()

def test_payload_encryption():
    sensitive_data = {
        "model_path": "/safe/path/to/model",
        "dataset_path": "/fake/path/to/data",
        "hf_token": "hf_super_secret_token_123",
        "public_id": "abc-123"
    }
    sensitive_keys = ["model_path", "dataset_path", "hf_token"]
    
    encrypted = encrypt_task_payload(sensitive_data, sensitive_keys)
    
    # Assert values are changed
    for k in sensitive_keys:
        assert encrypted[k] != sensitive_data[k]
        # Fernet tokens start with gAAAA
        assert encrypted[k].startswith("gAAAA")
    
    # Assert non-sensitive values are not changed
    assert encrypted["public_id"] == sensitive_data["public_id"]
    
    # Decrypt and verify
    decrypted = decrypt_task_payload(encrypted, sensitive_keys)
    for k in sensitive_keys:
        assert decrypted[k] == sensitive_data[k]
        
    assert decrypted["public_id"] == sensitive_data["public_id"]

def test_complex_payload_encryption():
    sensitive_data = {
        "config": {"api_key": "sk-12345", "db_url": "postgresql://user:pass@localhost:5432/db"},
        "dataset_b64s": ["b64data1", "b64data2"],
        "normal_field": "public"
    }
    sensitive_keys = ["config", "dataset_b64s"]
    
    encrypted = encrypt_task_payload(sensitive_data, sensitive_keys)
    
    assert isinstance(encrypted["config"], str)
    assert encrypted["config"].startswith("gAAAA")
    assert isinstance(encrypted["dataset_b64s"], str)
    assert encrypted["dataset_b64s"].startswith("gAAAA")
    
    decrypted = decrypt_task_payload(encrypted, sensitive_keys)
    
    assert decrypted["config"] == sensitive_data["config"]
    assert isinstance(decrypted["config"], dict)
    assert decrypted["dataset_b64s"] == sensitive_data["dataset_b64s"]
    assert isinstance(decrypted["dataset_b64s"], list)
    assert decrypted["normal_field"] == "public"

