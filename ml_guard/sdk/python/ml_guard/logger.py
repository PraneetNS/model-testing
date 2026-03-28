"""
ml_guard/logger.py — ML Guard SDK Prediction Logger

Wraps prediction ingestion to ensure the caller's application
NEVER crashes due to observability errors. Fire-and-forget by design.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, Optional


class JsonFormatter(logging.Formatter):
    """Structured JSON log formatter for ML Guard SDK output."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        }
        # Merge any extra fields attached to the record
        for key, val in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
            }:
                log_entry[key] = val
        return json.dumps(log_entry, default=str)


def setup_logger(name: str = "ml-guard-sdk", level: int = logging.INFO) -> logging.Logger:
    """Create or retrieve a JSON-formatted logger for the ML Guard SDK."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


class PredictionLogger:
    """
    Thread-safe, fire-and-forget prediction logger.
    Wraps the ML Guard ingest endpoint. Never raises.
    Designed for embedding directly inside inference code paths.

    Usage:
        plogger = PredictionLogger(host="http://localhost:8000", model_id="churn-v2")
        plogger.log(features={"age": 34}, prediction=1, proba=0.87)
    """

    def __init__(
        self,
        model_id: str,
        host: Optional[str] = None,
        api_key: Optional[str] = None,
        environment: str = "production",
        async_mode: bool = True,
    ):
        self.model_id = model_id
        self.environment = environment
        self.async_mode = async_mode
        self._logger = setup_logger()

        from ml_guard.client import MLGuardClient
        self._client = MLGuardClient(host=host, api_key=api_key)

    def log(
        self,
        features: Dict[str, Any],
        prediction: Any,
        proba: Optional[float] = None,
        latency_ms: Optional[float] = None,
        tags: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a prediction. If async_mode=True (default), dispatches in background thread.
        Never raises — connection errors are silently logged.
        """
        if self.async_mode:
            t = threading.Thread(
                target=self._safe_log,
                args=(features, prediction, proba, latency_ms, tags),
                daemon=True,
            )
            t.start()
        else:
            self._safe_log(features, prediction, proba, latency_ms, tags)

    def _safe_log(
        self,
        features: Dict[str, Any],
        prediction: Any,
        proba: Optional[float],
        latency_ms: Optional[float],
        tags: Optional[Dict[str, Any]],
    ) -> None:
        try:
            result = self._client.log(
                model_id=self.model_id,
                features=features,
                prediction=prediction,
                proba=proba,
                latency_ms=latency_ms,
                environment=self.environment,
                tags=tags,
            )
            self._logger.debug("prediction_logged", extra=result)
        except Exception as e:
            # NEVER re-raise — caller's inference must not be affected
            self._logger.warning("prediction_log_failed", extra={
                "model_id": self.model_id,
                "error": str(e),
            })
