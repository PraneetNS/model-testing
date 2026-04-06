"""
ml_guard/integrations/xgboost_integration.py — XGBoost Model Wrapper

Zero-code XGBoost instrumentation for ML Guard monitoring.

Usage:
    import xgboost as xgb
    from ml_guard.integrations import wrap_xgboost
    from ml_guard.client import MLGuardClient

    model = xgb.XGBClassifier().fit(X_train, y_train)
    client = MLGuardClient(host=\"http://localhost:8000\", api_key=\"mlg_xxx\")

    monitored = wrap_xgboost(model, model_id=\"fraud-v3\", client=client)
    preds = monitored.predict(X_test)  # auto-logged
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("mlguard.integrations.xgboost")


class MonitoredXGBoostModel:
    """ML Guard wrapper for XGBoost Booster or XGBClassifier/XGBRegressor."""

    def __init__(
        self,
        model,
        model_id: str,
        client,
        feature_names: Optional[List[str]] = None,
        environment: str = "production",
    ):
        self._model = model
        self.model_id = model_id
        self._client = client
        self.feature_names = feature_names
        self.environment = environment

    def _extract_features(self, X) -> List[Dict[str, Any]]:
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            names = X.columns.tolist()
            return X.to_dict(orient="records")
        arr = np.asarray(X)
        names = self.feature_names or (
            self._model.feature_names if hasattr(self._model, "feature_names") else
            [f"f{i}" for i in range(arr.shape[-1])]
        )
        if arr.ndim == 1:
            return [dict(zip(names, arr.tolist()))]
        return [dict(zip(names, row.tolist())) for row in arr]

    def _log_async(self, features_list, predictions, latency_ms=0.0):
        def _send():
            try:
                rows = [
                    {
                        "model_id": self.model_id,
                        "features": feat,
                        "prediction": str(pred),
                        "latency_ms": latency_ms / max(len(predictions), 1),
                        "data_source": "sdk",
                        "environment": self.environment,
                        "tags": {"wrapper": "xgboost"},
                    }
                    for feat, pred in zip(features_list, predictions)
                ]
                if len(rows) == 1:
                    self._client.log(**rows[0])
                elif rows:
                    self._client.log_batch(rows[:10_000])
            except Exception as e:
                logger.debug(f"xgboost_log_failed: {e}")

        threading.Thread(target=_send, daemon=True).start()

    def predict(self, X, **kwargs):
        start = time.perf_counter()
        result = self._model.predict(X, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        features_list = self._extract_features(X)
        self._log_async(features_list, np.asarray(result).tolist(), latency_ms)
        return result

    def predict_proba(self, X, **kwargs):
        start = time.perf_counter()
        probas = self._model.predict_proba(X, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        features_list = self._extract_features(X)
        preds = np.argmax(probas, axis=1).tolist()
        proba_max = probas.max(axis=1).tolist()
        self._log_async(features_list, preds, latency_ms)
        return probas

    def __getattr__(self, name):
        return getattr(self._model, name)

    def __repr__(self):
        return f"<MonitoredXGBoostModel model_id={self.model_id!r}>"


def wrap_xgboost(model, model_id: str, client, feature_names=None, environment="production"):
    """Wrap an XGBoost model with ML Guard monitoring."""
    return MonitoredXGBoostModel(
        model=model,
        model_id=model_id,
        client=client,
        feature_names=feature_names,
        environment=environment,
    )
