"""
ml_guard/integrations/sklearn_integration.py — Scikit-learn Model Wrapper

Zero-code drop-in replacement for sklearn pipelines that auto-logs
predictions, profiles input distributions, and enforces governance gates.

Unlike Evidently (manual report generation) and WhyLabs (separate profiling step),
ML Guard wraps the model object itself — monitoring is completely transparent.

Usage:
    from sklearn.ensemble import RandomForestClassifier
    from ml_guard.integrations import wrap_sklearn
    from ml_guard.client import MLGuardClient

    model = RandomForestClassifier().fit(X_train, y_train)
    client = MLGuardClient(host=\"http://localhost:8000\", api_key=\"mlg_xxx\")

    # Drop-in replacement — same API as original model
    monitored_model = wrap_sklearn(model, model_id=\"churn-v2\", client=client)

    # Predictions auto-logged to ML Guard
    predictions = monitored_model.predict(X_test)
    probas = monitored_model.predict_proba(X_test)  # also captured
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("mlguard.integrations.sklearn")


class MonitoredSklearnModel:
    """
    ML Guard-instrumented wrapper around any sklearn estimator or Pipeline.

    Maintains the full sklearn API (predict, predict_proba, score, transform)
    while automatically logging to ML Guard in background threads.
    """

    def __init__(
        self,
        model,
        model_id: str,
        client,
        feature_names: Optional[List[str]] = None,
        environment: str = "production",
        profile_every: int = 500,
        log_probas: bool = True,
    ):
        self._model = model
        self.model_id = model_id
        self._client = client
        self.feature_names = feature_names
        self.environment = environment
        self.profile_every = profile_every
        self.log_probas = log_probas
        self._call_count = 0
        self._lock = threading.Lock()

    def _extract_features(self, X) -> List[Dict[str, Any]]:
        """Convert numpy array or DataFrame to list of feature dicts."""
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            return X.to_dict(orient="records")

        arr = np.asarray(X)
        names = self.feature_names or [f"f{i}" for i in range(arr.shape[1] if arr.ndim > 1 else 1)]
        if arr.ndim == 1:
            return [{names[0]: float(arr[0])}]
        return [dict(zip(names, row.tolist())) for row in arr]

    def _log_async(
        self,
        features_list: List[Dict],
        predictions: List[Any],
        probas: Optional[List[float]] = None,
        latency_ms: float = 0.0,
    ) -> None:
        """Fire-and-forget background logging."""
        def _send():
            try:
                rows = []
                for i, (feat, pred) in enumerate(zip(features_list, predictions)):
                    row = {
                        "model_id": self.model_id,
                        "features": feat,
                        "prediction": str(pred),
                        "prediction_proba": float(probas[i]) if probas else None,
                        "latency_ms": latency_ms / max(len(predictions), 1),
                        "data_source": "sdk",
                        "environment": self.environment,
                        "tags": {"wrapper": "sklearn"},
                    }
                    rows.append(row)

                if len(rows) == 1:
                    self._client.log(**rows[0])
                elif rows:
                    self._client.log_batch(rows[:10_000])
            except Exception as e:
                logger.debug(f"sklearn_log_failed: {e}")

        threading.Thread(target=_send, daemon=True).start()

    def _maybe_profile(self, X) -> None:
        """Profile input data every N calls."""
        with self._lock:
            self._call_count += 1
            should_profile = self._call_count % self.profile_every == 0

        if should_profile:
            def _send_profile():
                try:
                    import pandas as pd
                    from ml_guard.profile import from_dataframe
                    if not isinstance(X, pd.DataFrame):
                        names = self.feature_names or [f"f{i}" for i in range(np.asarray(X).shape[-1])]
                        df = pd.DataFrame(np.asarray(X), columns=names)
                    else:
                        df = X

                    prof = from_dataframe(df, model_id=self.model_id,
                                         client=self._client)
                    self._client.upload_profile(prof)
                    logger.info(f"auto_profile_sent model_id={self.model_id} "
                               f"call={self._call_count} rows={len(df)}")
                except Exception as e:
                    logger.debug(f"auto_profile_failed: {e}")

            threading.Thread(target=_send_profile, daemon=True).start()

    def predict(self, X, **kwargs):
        """predict() with automatic ML Guard logging."""
        start = time.perf_counter()
        result = self._model.predict(X, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        features_list = self._extract_features(X)
        self._log_async(features_list, result.tolist(), latency_ms=latency_ms)
        self._maybe_profile(X)
        return result

    def predict_proba(self, X, **kwargs):
        """predict_proba() with automatic logging of confidence scores."""
        start = time.perf_counter()
        probas = self._model.predict_proba(X, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        predictions = np.argmax(probas, axis=1).tolist()
        proba_list = probas.max(axis=1).tolist()

        features_list = self._extract_features(X)
        self._log_async(features_list, predictions, probas=proba_list,
                       latency_ms=latency_ms)
        self._maybe_profile(X)
        return probas

    def score(self, X, y, **kwargs):
        """Passthrough to underlying model's score method."""
        return self._model.score(X, y, **kwargs)

    def transform(self, X, **kwargs):
        """Passthrough transform for sklearn Pipelines/transformers."""
        return self._model.transform(X, **kwargs)

    def fit(self, X, y=None, **kwargs):
        """Passthrough fit — original model training not affected."""
        return self._model.fit(X, y, **kwargs)

    # ── sklearn API compatibility ──────────────────────────────────────────────

    def __getattr__(self, name: str):
        """Delegate unknown attributes to the wrapped model."""
        return getattr(self._model, name)

    @property
    def classes_(self):
        return getattr(self._model, "classes_", None)

    @property
    def feature_importances_(self):
        return getattr(self._model, "feature_importances_", None)

    def __repr__(self) -> str:
        return (
            f"<MonitoredSklearnModel model_id={self.model_id!r} "
            f"base={type(self._model).__name__} calls={self._call_count}>"
        )


def wrap_sklearn(
    model,
    model_id: str,
    client,
    feature_names: Optional[List[str]] = None,
    environment: str = "production",
    profile_every: int = 500,
) -> MonitoredSklearnModel:
    """
    Wrap any scikit-learn estimator or Pipeline with ML Guard monitoring.

    Args:
        model: Any sklearn estimator or Pipeline
        model_id: ML Guard model identifier
        client: MLGuardClient instance
        feature_names: Optional feature names (auto-detected from DataFrame)
        environment: deployment environment tag
        profile_every: Send a data profile every N predict() calls

    Returns:
        MonitoredSklearnModel — drop-in replacement with same API
    """
    return MonitoredSklearnModel(
        model=model,
        model_id=model_id,
        client=client,
        feature_names=feature_names,
        environment=environment,
        profile_every=profile_every,
    )
