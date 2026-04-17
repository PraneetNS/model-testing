# Celery Worker Security Hardening

This document outlines the security measures implemented to protect the ML Guard Celery infrastructure against remote code execution (RCE) and data leakage.

## 1. Message Signing & Serialization
- **Pickle Disabled**: The use of `pickle` for task serialization is strictly forbidden to prevent RCE.
- **JSON Serialization**: `task_serializer` and `accept_content` are set to `json`.
- **HMAC Signing**: All task messages are signed using an HMAC-SHA256 key derived from the application `SECRET_KEY`. Any unsigned or improperly signed message will be rejected by the worker.

## 2. Payload Encryption
Sensitive task fields are encrypted using **Fernet (AES-128-CBC)** before being enqueued and decrypted only inside the task function.

### Protected Fields
- `model_path`
- `dataset_path`
- `train_path`, `test_path`, `val_path`
- `model_artifact_key`, `train_dataset_key`, `val_dataset_key`
- `hf_token` (HuggingFace API tokens)
- `config` (Data connector credentials)
- `model_b64`, `data_b64` (Base64 encoded file contents)

### Helpers
The following helpers in `app.core.celery_app` handle the encryption lifecycle:
- `encrypt_task_payload(data: dict, sensitive_keys: list)`
- `decrypt_task_payload(data: dict, sensitive_keys: list)`

## 3. Worker Isolation & Routing
Tasks are routed to dedicated queues to isolate high-risk analysis from routine observability.
- `drift`: For drift analysis.
- `audit`: For comprehensive scans and governance audits.
- `red_team`: For adversarial robustness testing.
- `default`: For routine maintenance and heartbeat.

## 4. Task Allowlist
A strict allowlist (`ALLOWED_TASKS`) is enforced using the `task_prerun` signal. If a task not on the list is received (e.g., via a compromised Redis instance), the worker logged a `SECURITY_ALERT` and aborts execution.

## 5. Result Backend Security
- **Expiration**: Task results expire after 3600 seconds (1 hour) to minimize the persistence of sensitive results in Redis.
- **Access Control**: Redis is expected to be configured with password authentication and isolated within the network.

## 6. Testing
Automated security tests are located in `backend/tests/test_celery_security.py`. These tests verify:
- Rejection of non-JSON content.
- Enforcement of the task allowlist.
- Correct encryption/decryption of complex payloads (dicts/lists).
