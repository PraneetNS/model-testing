"""
ml_guard/client.py — ML Guard Python SDK Client

Aligned with the actual v7.2 FastAPI router prefixes:
  POST /api/v1/ingest/predict
  POST /api/v1/ingest/batch
  POST /api/v1/ingest/label
  POST /api/v1/gate/evaluate
  GET  /api/v1/forecast/{model_id}
  WS   /api/v1/sentinel/stream/{model_id}
  GET  /api/v1/governance/{model_id}/score
  POST /api/v1/governance/{model_id}/gate
"""
from __future__ import annotations

import io
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class MLGuardClient:
    """
    Primary SDK client for the ML Guard v7.2 API.

    Usage:
        from ml_guard.client import MLGuardClient

        client = MLGuardClient(host="http://localhost:8000", api_key="mlg_xxx")

        # Log a prediction
        client.log(model_id="churn-v2", features={"age": 34}, prediction=1, proba=0.87)

        # CI gate check
        result = client.gate("churn-v2", policy_config={"min_accuracy": 0.85})
        if not result["passed"]:
            raise SystemExit(1)
    """

    def __init__(
        self,
        host: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 60,
    ):
        self.host = (host or os.getenv("MLGUARD_HOST", "http://localhost:8000")).rstrip("/")
        self.api_key = api_key or os.getenv("MLGUARD_API_KEY", "")
        self.timeout = timeout
        self._session = requests.Session()
        if self.api_key:
            self._session.headers["Authorization"] = f"Bearer {self.api_key}"
        self._session.headers["Content-Type"] = "application/json"
        self._session.headers["X-SDK-Version"] = "7.2.0"

    def _url(self, path: str) -> str:
        return f"{self.host}/api/v1/{path.lstrip('/')}"

    def _post(self, path: str, **kwargs: Any) -> Dict[str, Any]:
        url = self._url(path)
        try:
            r = self._session.post(url, timeout=self.timeout, **kwargs)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            raise RuntimeError(f"POST {url} failed [{e.response.status_code}]: {e.response.text}") from e

    def _get(self, path: str, **kwargs: Any) -> Dict[str, Any]:
        url = self._url(path)
        try:
            r = self._session.get(url, timeout=self.timeout, **kwargs)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            raise RuntimeError(f"GET {url} failed [{e.response.status_code}]: {e.response.text}") from e

    # ── Module 1: Prediction Ingestion ────────────────────────────────────────

    def log(
        self,
        model_id: str,
        features: Dict[str, Any],
        prediction: Any,
        proba: Optional[float] = None,
        latency_ms: Optional[float] = None,
        environment: str = "production",
        tags: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Log a single prediction to the observability pipeline.
        Non-blocking: backend accepts and writes asynchronously.
        Never raises — designed for fire-and-forget in production code.
        """
        try:
            payload = {
                "model_id": model_id,
                "features": features,
                "prediction": prediction,
                "prediction_proba": proba,
                "latency_ms": latency_ms,
                "data_source": "sdk",
                "environment": environment,
                "tags": tags or {},
            }
            return self._post("ingest/predict", json=payload)
        except Exception as e:
            logger.warning("mlguard_log_failed", extra={"model_id": model_id, "error": str(e)})
            return {"status": "error", "error": str(e), "log_id": None}

    def log_batch(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Log up to 10,000 predictions in a single call (dispatched via Celery)."""
        if len(rows) > 10_000:
            raise ValueError("Batch max is 10,000 rows.")
        return self._post("ingest/batch", json={"rows": rows})

    def add_labels(
        self,
        log_ids: List[str],
        ground_truths: List[Any],
    ) -> Dict[str, Any]:
        """Stitch ground truth labels onto logged predictions."""
        if len(log_ids) != len(ground_truths):
            raise ValueError("log_ids and ground_truths must have equal length.")
        return self._post("ingest/label", json={
            "log_ids": log_ids,
            "ground_truths": [str(g) for g in ground_truths],
        })

    # ── Module 2: CI/CD Gate ─────────────────────────────────────────────────

    def gate(
        self,
        model_id: str,
        policy_config: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Synchronous CI/CD governance gate check.
        Raises RuntimeError if gate fails so CI pipelines can detect failure.

        Example:
            try:
                result = client.gate("churn-v2", {"min_accuracy": 0.85})
                print(f"Gate passed — score {result['score']}")
            except RuntimeError as e:
                print(f"Gate FAILED: {e}")
                sys.exit(1)
        """
        try:
            return self._post(f"governance/{model_id}/gate", json={
                "policy_config": policy_config or {},
                "metrics": metrics or {},
            })
        except RuntimeError as e:
            if "422" in str(e):
                raise RuntimeError(f"Governance gate FAILED for model '{model_id}': {e}") from e
            raise

    def gate_evaluate(
        self,
        model_artifact_path: Optional[str] = None,
        inference_endpoint: Optional[str] = None,
        policy_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Legacy evaluate endpoint — wraps POST /api/v1/gate/evaluate.
        Accepts model artifact path or inference endpoint URL + policy config.
        """
        payload: Dict[str, Any] = {}
        if model_artifact_path:
            payload["model_artifact_path"] = model_artifact_path
        if inference_endpoint:
            payload["inference_endpoint"] = inference_endpoint
        if policy_config:
            payload["policy_config"] = policy_config
        return self._post("gate/evaluate", json=payload)

    # ── Module 3: Governance Score ────────────────────────────────────────────

    def get_score(self, model_id: str) -> Dict[str, Any]:
        """Get current governance score (audit + live decay)."""
        return self._get(f"governance/{model_id}/score")

    def get_live_score(self, model_id: str) -> Dict[str, Any]:
        """Get real-time live score (fastest — no full audit required)."""
        return self._get(f"governance/{model_id}/score/live")

    def certify(self, model_id: str) -> Dict[str, Any]:
        """Generate a compliance certificate. Returns cert_hash + verify URL."""
        return self._post(f"governance/{model_id}/certify", json={})

    def verify_cert(self, cert_hash: str) -> Dict[str, Any]:
        """Verify a compliance certificate (public — no auth)."""
        return self._get(f"governance/verify/{cert_hash}")

    # ── Module 4: Forecast ────────────────────────────────────────────────────

    def get_forecast(
        self,
        model_id: str,
        horizon_days: int = 30,
    ) -> Dict[str, Any]:
        """GET /api/v1/forecast/{model_id} — governance score forecast."""
        return self._get(f"forecast/{model_id}", params={"horizon_days": horizon_days})

    # ── Module 5: Observability ───────────────────────────────────────────────

    def get_drift_report(
        self,
        model_id: str,
        window_hours: int = 24,
        method: str = "ks",
    ) -> Dict[str, Any]:
        """GET /api/v1/observe/drift/{model_id}/report"""
        return self._get(
            f"observe/drift/{model_id}/report",
            params={"window_hours": window_hours, "method": method},
        )

    def get_performance(
        self,
        model_id: str,
        window_hours: int = 24,
    ) -> Dict[str, Any]:
        """GET /api/v1/observe/performance/{model_id}/live"""
        return self._get(
            f"observe/performance/{model_id}/live",
            params={"window_hours": window_hours},
        )

    # ── WebSocket Sentinel ────────────────────────────────────────────────────

    def get_sentinel_ws_url(self, model_id: str) -> str:
        """Returns the WebSocket URL for streaming drift from the Sentinel agent."""
        ws_host = self.host.replace("https://", "wss://").replace("http://", "ws://")
        return f"{ws_host}/api/v1/sentinel/stream/{model_id}"

    # ── Module 6: Data Profile Upload ─────────────────────────────────────────

    def upload_profile(self, profile_obj: Any) -> Dict[str, Any]:
        """
        Upload a DataProfile object to the ML Guard backend.
        The profile is serialized to JSON — no raw data is transmitted.

        Args:
            profile_obj: DataProfile instance from ml_guard.profile

        Returns:
            Backend acknowledgment with profile_id
        """
        return self._post("ingest/profile", json=profile_obj.to_dict())

    def compare_profiles(
        self,
        model_id: str,
        current_profile: Any,
        reference_profile: Any,
    ) -> Dict[str, Any]:
        """
        Upload two profiles for server-side comparison.
        Faster than local diff when profiles are already on the server.
        """
        return self._post(
            f"observe/profile/{model_id}/compare",
            json={
                "current": current_profile.to_dict(),
                "reference": reference_profile.to_dict(),
            },
        )

    # ── Module 7: Suite Reports ────────────────────────────────────────────────

    def upload_suite_report(
        self,
        model_id: str,
        report: Any,  # SuiteReport
    ) -> Dict[str, Any]:
        """
        Upload a SuiteReport to the ML Guard backend for historical tracking.
        Allows dashboard visualization of test suite trends.
        """
        try:
            return self._post(
                f"governance/{model_id}/suite-report",
                json=report.to_dict(),
            )
        except Exception:
            # Suite upload is best-effort — never block user code
            return {"status": "skipped", "reason": "upload_failed"}

    # ── Legacy Guard interface ────────────────────────────────────────────────

    def evaluate(
        self,
        model: Any,
        train_df: Any,
        val_df: Any,
        label_col: str = "target",
        model_name: str = "SDK-Uploaded-Model",
        selected_checks: List[str] = ["drift", "performance", "fairness", "security"],
        query: Optional[str] = None,
        timeout: int = 300,
        retries: int = 3,
    ) -> Dict[str, Any]:
        """
        Legacy full model evaluation upload (compatible with original Guard.evaluate).
        Uploads model + datasets to /api/v1/audit/run for a full audit scan.
        """
        import io
        import joblib
        import pandas as pd

        model_buffer = io.BytesIO()
        joblib.dump(model, model_buffer)
        model_buffer.seek(0)

        train_buffer = io.BytesIO()
        train_df.to_csv(train_buffer, index=False)
        train_buffer.seek(0)

        val_buffer = io.BytesIO()
        val_df.to_csv(val_buffer, index=False)
        val_buffer.seek(0)

        files = {
            "model_file": ("model.pkl", model_buffer, "application/octet-stream"),
            "train_file": ("train.csv",  train_buffer, "text/csv"),
            "val_file":   ("val.csv",   val_buffer,  "text/csv"),
        }
        data = {
            "model_name": model_name,
            "label_col": label_col,
            "selected": selected_checks,
            "query": query or "Automated SDK audit"
        }

        url = self._url("audit/run")
        for attempt in range(retries):
            try:
                r = requests.post(url, files=files, data=data, timeout=timeout)
                if r.status_code == 200:
                    return r.json()
                logger.warning(f"audit_attempt_{attempt+1}_failed", extra={"status": r.status_code})
            except Exception as e:
                logger.warning(f"audit_request_error", extra={"attempt": attempt, "error": str(e)})
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

        raise RuntimeError(f"Evaluation failed after {retries} attempts.")


# ── Backwards-compat alias ─────────────────────────────────────────────────
Guard = MLGuardClient
